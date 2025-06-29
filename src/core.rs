use pyo3::Python;
use pyo3::ffi as pyo3_ffi;
use pyo3::prelude::*;
use pyo3::types::PyList;

// Import the unified debug system
use crate::debug_println;
use crate::object::{create_list_empty, list_set_item_transfer, make_string_fast};

// hack object to pass raw pointer for PyObject
#[derive(Clone, Debug)]
pub struct PyObjectPtr(pub *mut pyo3_ffi::PyObject);
unsafe impl Send for PyObjectPtr {}
unsafe impl Sync for PyObjectPtr {}
impl Copy for PyObjectPtr {}

// Enum for worker results - either pre-converted PyObject or raw Rust type
#[derive(Debug)]
pub enum WorkerResult {
    PyObject((usize, PyObjectPtr)),
}

unsafe impl Send for WorkerResult {}

// Helper function to safely set list items with PyObjectPtr
#[inline(always)]
unsafe fn set_list_item(list_ptr: &PyObjectPtr, index: usize, item_ptr: PyObjectPtr) {
    list_set_item_transfer(list_ptr.0, index as isize, item_ptr.0);
}

// Bump allocator manager to prevent code duplication
pub struct BumpAllocatorManager {
    pub name: String,
    pub bump: bumpalo::Bump,
}

const MANAGEMENT_BATCH_SIZE: usize = 100;

impl BumpAllocatorManager {
    // Memory management constants
    const INITIAL_CAPACITY: usize = 256 * 1024; // 256KB
    const RESET_THRESHOLD: usize = 16 * 1024 * 1024; // 16MB 
    const FREE_THRESHOLD: usize = Self::RESET_THRESHOLD * 2; // 32MB

    // Constructor with custom name for threading/context
    pub fn new(name: String) -> Self {
        Self {
            name,
            bump: bumpalo::Bump::with_capacity(Self::INITIAL_CAPACITY),
        }
    }

    // Main memory management method
    pub fn manage_memory(&mut self) {
        let current_size = self.bump.allocated_bytes();

        if current_size > Self::FREE_THRESHOLD {
            self.bump = bumpalo::Bump::with_capacity(Self::INITIAL_CAPACITY);
            debug_println!(
                "{}: freed arena at {}MB",
                self.name,
                current_size / 1024 / 1024
            );
        } else if current_size > Self::RESET_THRESHOLD {
            self.bump.reset();
            debug_println!(
                "{}: reset arena at {}MB",
                self.name,
                current_size / 1024 / 1024
            );
        }
    }

    // Get reference to the bump allocator
    pub fn bump(&self) -> &bumpalo::Bump {
        &self.bump
    }
}

fn get_string_at_idx<'a>(list_ptr: &PyObjectPtr, idx: usize, bump: &'a bumpalo::Bump) -> &'a str {
    unsafe {
        let str_ptr = pyo3_ffi::PyList_GetItem(list_ptr.0, idx as isize);
        assert!(!str_ptr.is_null());
        make_string_fast(str_ptr, bump)
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

fn map_pylist_parallel<'py, F1, F2>(
    py: Python<'py>,
    list: &Bound<'py, PyList>,
    jobs: usize,
    inplace: bool,
    make_func: F1,
) -> PyResult<PyObject>
where
    F1: Fn() -> F2 + Send + Sync,
    F2: for<'a> Fn(&'a str) -> PyObjectPtr + Send + 'static,
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
            let result_list = create_list_empty(list_len as isize);
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
    let (sender, receiver) = crossbeam_channel::unbounded::<WorkerResult>();

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
            let mut bump_manager = BumpAllocatorManager::new(format!("Thread {}", job_idx));

            for i in range_start..range_stop {
                // Extract string from input list
                let bump_string = get_string_at_idx(&input_list_ptr, i, bump_manager.bump());

                let py_obj = func(bump_string);
                if inplace {
                    sender.send(WorkerResult::PyObject((i, py_obj))).unwrap();
                } else {
                    unsafe {set_list_item(&target_list_ptr, i, py_obj)};
                }

                if (i - range_start) % MANAGEMENT_BATCH_SIZE == 0 {
                    bump_manager.manage_memory();
                }
            }

            debug_println!(
                "Thread {} finished, final arena size: {}MB",
                job_idx,
                bump_manager.bump().allocated_bytes() / 1024 / 1024
            );
        });
    }

    // Close sender side to signal when all workers are done
    drop(sender);

    // Main thread: apply results as they arrive (streaming updates)
    for result in receiver {
        match result {
            WorkerResult::PyObject((index, py_obj)) => {
                // Pre-converted in worker thread - just set
                unsafe {
                    set_list_item(&target_list_ptr, index, py_obj);
                }
            }
        }
    }

    debug_println!("Passed the barrier");

    if inplace {
        Ok(list.clone().into())
    } else {
        unsafe { Ok(Py::from_owned_ptr(py, target_list_ptr.0)) }
    }
}

// Sequential processing for jobs=1 or fallback
fn map_pylist_sequential<'py, F1, F2>(
    py: Python<'py>,
    list: &Bound<'py, PyList>,
    inplace: bool,
    make_func: F1,
) -> PyResult<PyObject>
where
    F1: Fn() -> F2,
    F2: for<'a> Fn(&'a str) -> PyObjectPtr,
{
    let list_len = list.len();
    let input_list_ptr = PyObjectPtr(list.as_ptr());
    let func = make_func();

    debug_println!("sequential processing, list length {}", list_len);

    // Use bump allocator manager for sequential processing too
    let mut bump_manager = BumpAllocatorManager::new("Sequential".to_string());

    if inplace {
        // Modify existing list in place
        for i in 0..list_len {
            let bump_string = get_string_at_idx(&input_list_ptr, i, bump_manager.bump());
            let py_obj = func(bump_string);

            unsafe {
                set_list_item(&input_list_ptr, i, py_obj);
            }

            if i % MANAGEMENT_BATCH_SIZE == 0 {
                bump_manager.manage_memory();
            }
        }
        Ok(list.clone().into())
    } else {
        unsafe {
            // Create new list with exact size
            let result_list = create_list_empty(list_len as isize);
            assert!(!result_list.is_null());
            let result_list_ptr = PyObjectPtr(result_list);

            for i in 0..list_len {
                let bump_string = get_string_at_idx(&input_list_ptr, i, bump_manager.bump());
                let py_obj = func(bump_string);
                set_list_item(&result_list_ptr, i, py_obj);
                
                if i % MANAGEMENT_BATCH_SIZE == 0 {
                    bump_manager.manage_memory();
                }
            }

            Ok(Py::from_owned_ptr(py, result_list))
        }
    }
}

// Main entry point - simplified to just sequential vs parallel
pub fn map_pylist<'py, F1, F2>(
    py: Python<'py>,
    list: &Bound<'py, PyList>,
    jobs: usize,
    inplace: bool,
    make_func: F1,
) -> PyResult<PyObject>
where
    F1: Fn() -> F2 + Send + Sync,
    F2: for<'a> Fn(&'a str) -> PyObjectPtr + Send + 'static,
{
    if jobs == 1 {
        map_pylist_sequential(py, list, inplace, make_func)
    } else {
        map_pylist_parallel(py, list, jobs, inplace, make_func)
    }
}
