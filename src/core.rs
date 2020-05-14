use cpython::{PyList, Python, PythonObject, ToPyObject};

// hack object to pass raw pointer for PyObject
#[derive(Clone)]
struct PyObjectPtr(*mut python3_sys::PyObject);
unsafe impl Send for PyObjectPtr {}
unsafe impl Sync for PyObjectPtr {}

fn make_string_unsafe(o: *mut python3_sys::PyObject) -> String {
    use std::alloc::{alloc, dealloc, Layout};
    use std::mem;
    use widestring::U32CStr;

    let t_align = mem::align_of::<python3_sys::Py_UCS4>();
    let t_size = mem::size_of::<python3_sys::Py_UCS4>();

    unsafe {
        // +1 cause we use null-terminated strings
        let length = python3_sys::PyUnicode_GetLength(o) + 1;

        let layout = Layout::from_size_align(t_size * length as usize, t_align).unwrap();
        #[allow(clippy::cast_ptr_alignment)]
        let buffer = alloc(layout) as *mut python3_sys::Py_UCS4;
        assert!(!buffer.is_null());

        // in good case PyUnicode_AsUCS4 falls into pure memcpy
        // and it does not mess with python gil (cause not use pymalloc)
        let r = python3_sys::PyUnicode_AsUCS4(o, buffer, length, 1);
        assert!(!r.is_null());

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

fn get_string_at_idx(list: *mut python3_sys::PyObject, idx: usize) -> String {
    unsafe {
        let str_ptr = python3_sys::PyList_GetItem(list, idx as isize);
        assert!(!str_ptr.is_null());
        make_string_unsafe(str_ptr)
    }
}

fn make_range(len: usize, chunk_size: usize, i: usize) -> (usize, usize) {
    let range_start = i * chunk_size;
    let range_stop = std::cmp::min(range_start + chunk_size, len);

    (range_start, range_stop)
}

pub fn map_pylist<'a, T: 'static, F1, F2: 'static>(
    py: Python,
    list: PyList,
    jobs: usize,
    inplace: bool,
    make_func: F1,
) -> PyList
where
    T: ToPyObject
        + std::marker::Send
        + std::marker::Sync
        + std::clone::Clone
        + std::default::Default,
    F1: Fn() -> F2,
    F2: Fn(&str) -> T + std::marker::Send,
{
    let jobs = if jobs < 1 { 1 } else { jobs };
    let chunk_size = (list.len(py) / jobs) + 1;
    let list_len = list.len(py);
    let list_ptr = PyObjectPtr(list.as_object().as_ptr());

    eprintln!("jobs {}, chunk size {}", jobs, chunk_size);

    // setup threading pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .thread_name(|t| format!("worker_{}", t))
        .start_handler(|t| {
            eprintln!("worker_{} start", t);
        })
        .exit_handler(|t| {
            eprintln!("worker_{} exit", t);
        })
        .build()
        .unwrap();

    // channels to send task and receive results
    let (send_result, get_result) = crossbeam_channel::unbounded();

    // init all workers
    for t in 0..jobs {
        let (range_start, range_stop) = make_range(list_len, chunk_size, t);
        let send_result = send_result.clone();
        let list_ptr = list_ptr.clone();

        let func = make_func();
        pool.spawn(move || {
            eprintln!(
                "worker_{} started, range {}, {}",
                t, range_start, range_stop
            );
            for i in range_start..range_stop {
                let string = get_string_at_idx(list_ptr.0, i);
                let result = func(&string);

                send_result.send((i, result)).unwrap();
            }
        });
    }
    // we don't need this channel after init of all workers
    drop(send_result);
    // collecting all remain results
    return if inplace {
        get_result.iter().for_each(|(i, o)| {
            let item = o.into_py_object(py).into_object();
            list.set_item(py, i, item);
        });
        list
    } else {
        let mut tmp_vec = vec![T::default(); list.len(py)];
        get_result.iter().for_each(|(i, o)| {
            tmp_vec[i] = o;
        });
        tmp_vec.into_py_object(py)
    };
}
