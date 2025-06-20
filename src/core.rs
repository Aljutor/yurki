use pyo3::BoundObject;
use pyo3::ffi as pyo3_ffi;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::{PyObject, Python};

use crate::fast_string::make_string_fast;

// Memory management constants for bump allocator
const INITIAL_BUMP_CAPACITY: usize = 256 * 1024; // 256KB
const RESET_THRESHOLD: usize = 16 * 1024 * 1024; // 16MB 
const FREE_THRESHOLD: usize = RESET_THRESHOLD * 2; // 32MB

macro_rules! debug_println {
    ($($arg:tt)*) => {
        if std::env::var("DEBUG").is_ok() {
            eprintln!($($arg)*);
        }
    };
}
// hack object to pass raw pointer for PyObject
#[derive(Clone)]
struct PyObjectPtr(*mut pyo3_ffi::PyObject);
unsafe impl Send for PyObjectPtr {}
unsafe impl Sync for PyObjectPtr {}

fn get_string_at_idx<'a>(list_ptr: &PyObjectPtr, idx: usize, bump: &'a bumpalo::Bump) -> &'a str {
    unsafe {
        let str_ptr = pyo3_ffi::PyList_GetItem(list_ptr.0, idx as isize);
        assert!(!str_ptr.is_null());
        make_string_fast(str_ptr, bump)
    }
}

fn map_pylist_sequential<'py, 'a, T, F1, F2>(
    py: Python<'py>,
    list: &'a Bound<'py, PyList>,
    inplace: bool,
    make_func: F1,
) -> PyResult<Py<PyList>>
where
    T: pyo3::conversion::IntoPyObject<'py>
        + std::marker::Send
        + std::marker::Sync
        + std::clone::Clone
        + std::default::Default
        + 'static,
    T::Error: std::convert::Into<pyo3::PyErr> + std::fmt::Debug,
    F1: Fn() -> F2,
    F2: Fn(&str) -> T + std::marker::Send + 'static,
{
    let list_len = list.len();
    let list_ptr = PyObjectPtr(list.as_ptr());
    let func = make_func();

    debug_println!("sequential processing, list length {}", list_len);

    // Use bump allocator for sequential processing too
    let mut bump = bumpalo::Bump::with_capacity(INITIAL_BUMP_CAPACITY);

    if inplace {
        // Modify existing list in place using direct FFI
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

            let bump_string = get_string_at_idx(&list_ptr, i, &bump);
            let result = func(bump_string);
            let item: PyObject = result.into_pyobject(py).unwrap().into_any().unbind();

            unsafe {
                // Direct FFI call - no bounds checking, maximum performance
                pyo3_ffi::PyList_SetItem(list_ptr.0, i as isize, item.into_ptr());
            }
        }
        Ok(list.clone().into())
    } else {
        unsafe {
            // create list with exact size
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

                let bump_string = get_string_at_idx(&list_ptr, i, &bump);
                let result = func(bump_string);
                let item: PyObject = result.into_pyobject(py).unwrap().into_any().unbind();

                // direct set, PyList_SetItem steals the reference
                pyo3_ffi::PyList_SetItem(result_list, i as isize, item.into_ptr());
            }

            Ok(Py::from_owned_ptr(py, result_list))
        }
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

#[cfg(not(all(Py_3_13, py_sys_config = "Py_GIL_DISABLED")))]
fn map_pylist_parallel<'py, 'a, T, F1, F2>(
    py: Python<'py>,
    list: &'a Bound<'py, PyList>,
    jobs: usize,
    inplace: bool,
    make_func: F1,
) -> PyResult<Py<PyList>>
where
    T: pyo3::conversion::IntoPyObject<'py>
        + std::marker::Send
        + std::marker::Sync
        + std::clone::Clone
        + std::default::Default
        + 'static,
    T::Error: std::convert::Into<pyo3::PyErr> + std::fmt::Debug,
    F1: Fn() -> F2,
    F2: Fn(&str) -> T + std::marker::Send + 'static,
{
    let list_len = list.len();
    let list_ptr = PyObjectPtr(list.as_ptr());

    let real_jobs = jobs.min(list_len);
    debug_println!("parallel processing: jobs {}", real_jobs);

    // setup threading pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(real_jobs)
        .thread_name(|t| format!("worker_{}", t))
        .start_handler(|t| {
            debug_println!("worker_{} init", t);
        })
        .exit_handler(|t| {
            debug_println!("worker_{} exit", t);
        })
        .build()
        .unwrap();

    // channels to send task and receive results
    let (send_result, get_result) = crossbeam_channel::unbounded();

    // init all workers
    for job_idx in 0..real_jobs {
        let (range_start, range_stop) = make_range(list_len, real_jobs, job_idx);
        let send_result = send_result.clone();
        let list_ptr = list_ptr.clone();

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
                    // Drastic action: drop and recreate to free system memory
                    drop(bump);
                    bump = bumpalo::Bump::with_capacity(INITIAL_BUMP_CAPACITY);
                    debug_println!(
                        "Thread {}: freed arena at {}MB",
                        job_idx,
                        current_size / 1024 / 1024
                    );
                } else if current_size > RESET_THRESHOLD {
                    // Gentle reset: keeps allocated memory but resets pointer
                    bump.reset();
                    debug_println!(
                        "Thread {}: reset arena at {}MB",
                        job_idx,
                        current_size / 1024 / 1024
                    );
                }

                let bump_string = get_string_at_idx(&list_ptr, i, &bump);
                let result = func(bump_string);

                send_result.send((i, result)).unwrap();
            }

            debug_println!(
                "Thread {} finished, final arena size: {}MB",
                job_idx,
                bump.allocated_bytes() / 1024 / 1024
            );
        });
    }
    // we don't need this channel side after init of all workers
    drop(send_result);

    // collecting all remain results
    if inplace {
        get_result.iter().for_each(|(i, o)| {
            let item: PyObject = o.into_pyobject(py).unwrap().into_any().unbind();
            unsafe {
                pyo3_ffi::PyList_SetItem(list_ptr.0, i as isize, item.into_ptr());
            }
        });
        Ok(list.clone().into())
    } else {
        unsafe {
            // create list with exact size
            // it's imporant to fully init it
            let result_list = pyo3_ffi::PyList_New(list_len as isize);
            assert!(!result_list.is_null());

            get_result.iter().for_each(|(i, o)| {
                let item: PyObject = o.into_pyobject(py).unwrap().into_any().unbind();
                // Direct set, PyList_SetItem steals the reference
                pyo3_ffi::PyList_SetItem(result_list, i as isize, item.into_ptr());
            });

            Ok(Py::from_owned_ptr(py, result_list))
        }
    }
}

pub fn map_pylist<'py, 'a, T, F1, F2>(
    py: Python<'py>,
    list: &'a Bound<'py, PyList>,
    jobs: usize,
    inplace: bool,
    make_func: F1,
) -> PyResult<Py<PyList>>
where
    T: pyo3::conversion::IntoPyObject<'py>
        + std::marker::Send
        + std::marker::Sync
        + std::clone::Clone
        + std::default::Default
        + 'static,
    T::Error: std::convert::Into<pyo3::PyErr> + std::fmt::Debug,
    F1: Fn() -> F2,
    F2: Fn(&str) -> T + std::marker::Send + 'static,
{
    if jobs == 1 {
        map_pylist_sequential(py, list, inplace, make_func)
    } else {
        map_pylist_parallel(py, list, jobs, inplace, make_func)
    }
}
