use pyo3::ffi as pyo3_ffi;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::Python;

// Import the unified debug system
use crate::debug_println;
use crate::converter::{ToPyObject};

// hack object to pass raw pointer for PyObject
#[derive(Clone, Debug)]
pub struct PyObjectPtr(pub *mut pyo3_ffi::PyObject);
unsafe impl Send for PyObjectPtr {}
unsafe impl Sync for PyObjectPtr {}

// Enum for worker results - either pre-converted PyObject or raw Rust type
#[derive(Debug)]
pub enum WorkerResult<T> {
    PyObject((usize, PyObjectPtr)),  // Pre-converted in worker thread
    RustType((usize, T)),           // Raw type for main thread conversion
}

unsafe impl<T: Send> Send for WorkerResult<T> {}

// Helper function to safely set list items with PyObjectPtr
#[inline(always)]
unsafe fn set_list_item(list_ptr: &PyObjectPtr, index: usize, item_ptr: PyObjectPtr) {
    pyo3_ffi::PyList_SetItem(list_ptr.0, index as isize, item_ptr.0);
}

// Memory management constants for bump allocator
const INITIAL_BUMP_CAPACITY: usize = 256 * 1024; // 256KB
const RESET_THRESHOLD: usize = 16 * 1024 * 1024; // 16MB 
const FREE_THRESHOLD: usize = RESET_THRESHOLD * 2; // 32MB

// Custom read function, to replace python's PyUnicode_AsUTF8AndSize
// PyUnicode_AsUTF8AndSize unfortunatly is not thread safe before python 3.13t
// this version does whole string conversion on rust side and kinda thread "safe"
// also, it uses bump allocator for performance optimization
fn make_string_unsafe<'a>(
    o: *mut pyo3_ffi::PyObject,
    bump: &'a bumpalo::Bump,
) -> bumpalo::collections::String<'a> {
    use std::slice;

    unsafe {
        // Asserts for sanity
        assert!(!o.is_null());
        assert!(pyo3_ffi::PyUnicode_Check(o) != 0);

        if pyo3_ffi::PyUnicode_READY(o) != 0 {
            panic!("PyUnicode_READY failed");
        }

        let len = pyo3_ffi::PyUnicode_GET_LENGTH(o) as usize;
        let kind = pyo3_ffi::PyUnicode_KIND(o);
        let data = pyo3_ffi::PyUnicode_DATA(o);

        let mut output = bumpalo::collections::Vec::with_capacity_in(len, bump);

        match kind {
            pyo3_ffi::PyUnicode_1BYTE_KIND => {
                let chars = slice::from_raw_parts(data as *const u8, len);
                for &ch in chars {
                    if ch < 0x80 {
                        output.push(ch);
                    } else {
                        let c = std::char::from_u32(ch as u32).unwrap();
                        let mut buf = [0u8; 4];
                        output.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
            pyo3_ffi::PyUnicode_2BYTE_KIND => {
                let chars = slice::from_raw_parts(data as *const u16, len);
                for &ch in chars {
                    let c = std::char::from_u32(ch as u32).unwrap();
                    let mut buf = [0u8; 4];
                    output.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
            pyo3_ffi::PyUnicode_4BYTE_KIND => {
                let chars = slice::from_raw_parts(data as *const u32, len);
                for &ch in chars {
                    let c = std::char::from_u32(ch).unwrap();
                    let mut buf = [0u8; 4];
                    output.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
                }
            }
            _ => panic!("Unknown Unicode kind"),
        }

        bumpalo::collections::String::from_utf8_unchecked(output)
    }
}

fn get_string_at_idx<'a>(
    list_ptr: &PyObjectPtr,
    idx: usize,
    bump: &'a bumpalo::Bump,
) -> bumpalo::collections::String<'a> {
    unsafe {
        let str_ptr = pyo3_ffi::PyList_GetItem(list_ptr.0, idx as isize);
        assert!(!str_ptr.is_null());
        make_string_unsafe(str_ptr, bump)
    }
}

fn make_range(len: usize, jobs: usize, i: usize) -> (usize, usize) {
    assert!(jobs > 0, "jobs must be > 0");
    assert!(
        i < jobs,
        "thread index {} is out of range (jobs = {})",
        i,
        jobs
    );

    let base = len / jobs;
    let rem = len % jobs;

    // Distribute the remainder to the first `rem` jobs
    let start = i * base + i.min(rem);
    let end = start + base + if i < rem { 1 } else { 0 };

    (start, end)
}

fn map_pylist_parallel<'py, 'a, T, F1, F2>(
    py: Python<'py>,
    list: &'a Bound<'py, PyList>,
    jobs: usize,
    inplace: bool,
    make_func: F1,
) -> PyResult<Py<PyList>>
where
    T: ToPyObject + Send + 'static,
    F1: Fn() -> F2 + Send + Sync,
    F2: Fn(&str) -> T + Send + 'static,
{
    let list_len = list.len();
    let input_list_ptr = PyObjectPtr(list.as_ptr());

    let real_jobs = jobs.min(list_len);
    debug_println!("parallel processing: jobs {}", real_jobs);

    // Create result list or use input list
    let target_list_ptr = if inplace {
        input_list_ptr.clone()
    } else {
        unsafe {
            let result_list = pyo3_ffi::PyList_New(list_len as isize);
            assert!(!result_list.is_null());
            PyObjectPtr(result_list)
        }
    };

    // Setup threading pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(real_jobs)
        .thread_name(|t| format!("worker_{}", t))
        .start_handler(|_t| {
            debug_println!("worker_{} init", _t);
        })
        .exit_handler(|_t| {
            debug_println!("worker_{} exit", _t);
        })
        .build()
        .unwrap();

    // Create channel for streaming results from workers to main thread
    let (sender, receiver) = crossbeam_channel::unbounded::<WorkerResult<T>>();
    
    for job_idx in 0..real_jobs {
        let (range_start, range_stop) = make_range(list_len, real_jobs, job_idx);
        let input_list_ptr = input_list_ptr.clone();
        let sender = sender.clone();
    
        let func = make_func();
        pool.spawn(move || {
            debug_println!(
                "thread {} started, range {}, {}",
                job_idx,
                range_start,
                range_stop
            );

            // Pre-allocate bump arena for this thread
            let mut bump = bumpalo::Bump::with_capacity(INITIAL_BUMP_CAPACITY);

            for i in range_start..range_stop {
                let current_size = bump.allocated_bytes();

                if current_size > FREE_THRESHOLD {
                    drop(bump);
                    bump = bumpalo::Bump::with_capacity(INITIAL_BUMP_CAPACITY);
                    debug_println!(
                        "Thread {}: freed arena at {}MB",
                        job_idx,
                        current_size / 1024 / 1024
                    );
                } else if current_size > RESET_THRESHOLD {
                    bump.reset();
                    debug_println!(
                        "Thread {}: reset arena at {}MB",
                        job_idx,
                        current_size / 1024 / 1024
                    );
                }

                // Extract string from input list
                let bump_string = get_string_at_idx(&input_list_ptr, i, &bump);
                
                // Process the string
                let result = func(bump_string.as_str());
                
                // Compile-time dispatch based on conversion strategy
                if T::THREAD_SAFE {
                    // Safe to convert in worker thread (String, bool)
                    unsafe {
                        let py_obj = result.to_py_object();
                        sender.send(WorkerResult::PyObject((i, py_obj))).unwrap();
                    }
                } else {
                    // Needs main thread conversion (Vec<T>)
                    sender.send(WorkerResult::RustType((i, result))).unwrap();
                }
            };
        

            debug_println!(
                "Thread {} finished, final arena size: {}MB",
                job_idx,
                bump.allocated_bytes() / 1024 / 1024
            );
        });
    };

    // Close sender side to signal when all workers are done
    drop(sender);

    // Main thread: apply results as they arrive (streaming updates)
    for worker_result in receiver {
        match worker_result {
            WorkerResult::PyObject((index, py_obj)) => {
                // Pre-converted in worker thread - just set
                unsafe {
                    set_list_item(&target_list_ptr, index, py_obj);
                }
            }
            WorkerResult::RustType((index, rust_obj)) => {
                // Convert in main thread with GIL
                unsafe {
                    let py_obj = rust_obj.to_py_object();
                    set_list_item(&target_list_ptr, index, py_obj);
                }
            }
        }
    }


    debug_println!("Passed the barrier");

    if inplace {
        Ok(list.clone().into())
    } else {
        unsafe {
            Ok(Py::from_owned_ptr(py, target_list_ptr.0))
        }
    }
}

// Sequential processing for jobs=1 or fallback
fn map_pylist_sequential<'py, 'a, T, F1, F2>(
    py: Python<'py>,
    list: &'a Bound<'py, PyList>,
    inplace: bool,
    make_func: F1,
) -> PyResult<Py<PyList>>
where
    T: ToPyObject + Send + 'static,
    F1: Fn() -> F2,
    F2: Fn(&str) -> T + Send + 'static,
{
    let list_len = list.len();
    let input_list_ptr = PyObjectPtr(list.as_ptr());
    let func = make_func();

    debug_println!("sequential processing, list length {}", list_len);

    // Use bump allocator for sequential processing too
    let mut bump = bumpalo::Bump::with_capacity(INITIAL_BUMP_CAPACITY);

    if inplace {
        // Modify existing list in place
        for i in 0..list_len {
            let current_size = bump.allocated_bytes();

            if current_size > FREE_THRESHOLD {
                drop(bump);
                bump = bumpalo::Bump::with_capacity(INITIAL_BUMP_CAPACITY);
                debug_println!(
                    "Sequential: freed arena at {}MB",
                    current_size / 1024 / 1024
                );
            } else if current_size > RESET_THRESHOLD {
                bump.reset();
                debug_println!(
                    "Sequential: reset arena at {}MB",
                    current_size / 1024 / 1024
                );
            }

            let bump_string = get_string_at_idx(&input_list_ptr, i, &bump);
            let result = func(bump_string.as_str());

            unsafe {
                let py_obj = result.to_py_object();
                set_list_item(&input_list_ptr, i, py_obj);
            }
        }
        Ok(list.clone().into())
    } else {
        unsafe {
            // Create new list with exact size
            let result_list = pyo3_ffi::PyList_New(list_len as isize);
            assert!(!result_list.is_null());

            for i in 0..list_len {
                let current_size = bump.allocated_bytes();

                if current_size > FREE_THRESHOLD {
                    drop(bump);
                    bump = bumpalo::Bump::with_capacity(INITIAL_BUMP_CAPACITY);
                    debug_println!(
                        "Sequential: freed arena at {}MB",
                        current_size / 1024 / 1024
                    );
                } else if current_size > RESET_THRESHOLD {
                    bump.reset();
                    debug_println!(
                        "Sequential: reset arena at {}MB",
                        current_size / 1024 / 1024
                    );
                }

                let bump_string = get_string_at_idx(&input_list_ptr, i, &bump);
                let result = func(bump_string.as_str());

                let py_obj = result.to_py_object();
                let result_list_ptr = PyObjectPtr(result_list);
                set_list_item(&result_list_ptr, i, py_obj);
            }

            let result_ptr = PyObjectPtr(result_list);
            Ok(Py::from_owned_ptr(py, result_ptr.0))
        }
    }
}

// Main entry point - simplified to just sequential vs parallel
pub fn map_pylist<'py, 'a, T, F1, F2>(
    py: Python<'py>,
    list: &'a Bound<'py, PyList>,
    jobs: usize,
    inplace: bool,
    make_func: F1,
) -> PyResult<Py<PyList>>
where
    T: ToPyObject + Send + 'static,
    F1: Fn() -> F2 + Send + Sync,
    F2: Fn(&str) -> T + Send + 'static,
{
    if jobs == 1 {
        map_pylist_sequential(py, list, inplace, make_func)
    } else {
        map_pylist_parallel(py, list, jobs, inplace, make_func)
    }
}
