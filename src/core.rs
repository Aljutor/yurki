use pyo3::ffi as pyo3_ffi;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::{PyObject, Python, ToPyObject};
// hack object to pass raw pointer for PyObject
#[derive(Clone)]
struct PyObjectPtr(*mut pyo3_ffi::PyObject);
unsafe impl Send for PyObjectPtr {}
unsafe impl Sync for PyObjectPtr {}

fn make_string_unsafe(o: *mut pyo3_ffi::PyObject) -> String {
    use std::alloc::{alloc, dealloc, Layout};
    use std::mem;
    use widestring::U32CStr;

    // it maybe possible to remove manual memory allocation here

    let t_align = mem::align_of::<pyo3_ffi::Py_UCS4>();
    let t_size = mem::size_of::<pyo3_ffi::Py_UCS4>();

    unsafe {
        // asserts for sanity
        assert!(!o.is_null());
        assert!(pyo3_ffi::PyUnicode_Check(o) != 0);

        // +1 cause we use null-terminated strings
        let length = pyo3_ffi::PyUnicode_GetLength(o) + 1;

        let layout = Layout::from_size_align(t_size * length as usize, t_align).unwrap();
        #[allow(clippy::cast_ptr_alignment)]
        let buffer = alloc(layout) as *mut pyo3_ffi::Py_UCS4;
        assert!(!buffer.is_null());

        // in good case PyUnicode_AsUCS4 falls into pure memcpy
        // and it does not mess with python gil (it does not use pymalloc)
        let result: *mut u32 = pyo3_ffi::PyUnicode_AsUCS4(o, buffer, length, 1);
        assert!(!result.is_null());

        // from_ptr_with_nul accepts len of string without null-terminator
        // in general case we should get valid utf-8 string from python

        #[allow(clippy::wrong_self_convention)]
        let string = U32CStr::from_ptr_with_nul(buffer, (length - 1) as usize)
            .to_string()
            .unwrap();

        dealloc(buffer as *mut u8, layout);
        string
    }
}

fn get_string_at_idx(list: *mut pyo3_ffi::PyObject, idx: usize) -> String {
    unsafe {
        let str_ptr = pyo3_ffi::PyList_GetItem(list, idx as isize);
        assert!(!str_ptr.is_null());
        make_string_unsafe(str_ptr)
    }
}

fn map_pylist_sequential<'a, T, F1, F2>(
    py: Python,
    list: &'a Bound<PyList>,
    inplace: bool,
    make_func: F1,
) -> PyResult<Py<PyList>>
where
    T: ToPyObject
        + std::marker::Send
        + std::marker::Sync
        + std::clone::Clone
        + std::default::Default
        + 'static,
    F1: Fn() -> F2,
    F2: Fn(&str) -> T + std::marker::Send + 'static,
{
    let list_len = list.len();
    let list_ptr = PyObjectPtr(list.as_ptr());
    let func = make_func();

    eprintln!("sequential processing, list length {}", list_len);

    if inplace {
        // Modify existing list in place using direct FFI
        for i in 0..list_len {
            let string = get_string_at_idx(list_ptr.0, i);
            let result = func(&string);
            let item: PyObject = result.to_object(py);
            
            unsafe {
                // Direct FFI call - no bounds checking, maximum performance
                pyo3_ffi::PyList_SetItem(list_ptr.0, i as isize, item.into_ptr());
            }
        }
        Ok(list.clone().into())
    } else {
        unsafe {
            // Create list with exact size - no reallocation needed
            let result_list = pyo3_ffi::PyList_New(list_len as isize);
            assert!(!result_list.is_null());

            for i in 0..list_len {
                let string = get_string_at_idx(list_ptr.0, i);
                let result = func(&string);
                let item: PyObject = result.to_object(py);
                
                // Direct set - PyList_SetItem steals the reference
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

fn map_pylist_parallel<'a, T, F1, F2>(
    py: Python,
    list: &'a Bound<PyList>,
    jobs: usize,
    inplace: bool,
    make_func: F1,
) -> PyResult<Py<PyList>>
where
    T: ToPyObject
        + std::marker::Send
        + std::marker::Sync
        + std::clone::Clone
        + std::default::Default
        + 'static,
    F1: Fn() -> F2,
    F2: Fn(&str) -> T + std::marker::Send + 'static,
{
    let list_len = list.len();
    let list_ptr = PyObjectPtr(list.as_ptr());

    let real_jobs = jobs.min(list_len);
    eprintln!("parallel processing: jobs {}", real_jobs);

    // setup threading pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(real_jobs)
        .thread_name(|t| format!("worker_{}", t))
        .start_handler(|t| {
            eprintln!("thread{} init", t);
        })
        .exit_handler(|t| {
            eprintln!("thread{} exit", t);
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
            eprintln!(
                "thread {} started, range {}, {}",
                job_idx, range_start, range_stop
            );
            for i in range_start..range_stop {
                let string = get_string_at_idx(list_ptr.0, i);
                let result = func(&string);

                send_result.send((i, result)).unwrap();
            }
        });
    }
    // we don't need this channel side after init of all workers
    drop(send_result);

    // collecting all remain results
    if inplace {
        // Direct modification of existing list using FFI
        get_result.iter().for_each(|(i, o)| {
            let item: PyObject = o.to_object(py);
            unsafe {
                pyo3_ffi::PyList_SetItem(list_ptr.0, i as isize, item.into_ptr());
            }
        });
        Ok(list.clone().into())
    } else {
        unsafe {
            // Create list with exact size - no reallocation needed
            let result_list = pyo3_ffi::PyList_New(list_len as isize);
            assert!(!result_list.is_null());

            // Set results directly at their indices using FFI
            get_result.iter().for_each(|(i, o)| {
                let item: PyObject = o.to_object(py);
                // Direct set - PyList_SetItem steals the reference
                pyo3_ffi::PyList_SetItem(result_list, i as isize, item.into_ptr());
            });

            Ok(Py::from_owned_ptr(py, result_list))
        }
    }
}

pub fn map_pylist<'a, T, F1, F2>(
    py: Python,
    list: &'a Bound<PyList>,
    jobs: usize,
    inplace: bool,
    make_func: F1,
) -> PyResult<Py<PyList>>
where
    T: ToPyObject
        + std::marker::Send
        + std::marker::Sync
        + std::clone::Clone
        + std::default::Default
        + 'static,
    F1: Fn() -> F2,
    F2: Fn(&str) -> T + std::marker::Send + 'static,
{
    if jobs == 1 {
        map_pylist_sequential(py, list, inplace, make_func)
    } else {
        map_pylist_parallel(py, list, jobs, inplace, make_func)
    }
}
