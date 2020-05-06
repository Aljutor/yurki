use cpython::{PyList, PyObject, Python, PythonObject, ToPyObject};

#[inline]
fn make_string_fast(_py: Python, o: PyObject) -> String {
    unsafe {
        let mut size: python3_sys::Py_ssize_t = 0;
        let data = python3_sys::PyUnicode_AsUTF8AndSize(o.as_ptr(), &mut size) as *const u8;
        std::str::from_utf8(std::slice::from_raw_parts(data, (size) as usize))
            .unwrap()
            .to_string()
    }
}

pub fn map_pylist_inplace<'a, T: ToPyObject>(
    py: Python,
    list: &'a PyList,
    func: fn(&str) -> T,
) -> &'a PyList {
    for (i, item) in list.iter(py).enumerate() {
        let string = make_string_fast(py, item);
        let result = func(&string);

        let item = result.into_py_object(py).into_object();
        list.set_item(py, i, item);
    }

    return list;
}

pub fn map_pylist<T: ToPyObject>(py: Python, list: &PyList, func: fn(&str) -> T) -> PyList {
    let vec: Vec<PyObject> = list
        .iter(py)
        .map(|o| {
            let string = make_string_fast(py, o);
            func(&string).to_py_object(py).into_object()
        })
        .collect::<Vec<PyObject>>();

    PyList::new(py, &vec)
}

pub fn map_pylist_inplace_parallel<
    'a,
    T: 'static + ToPyObject + std::marker::Send + std::marker::Sync,
>(
    py: Python,
    list: &'a PyList,
    jobs: usize,
    func: fn(String) -> T,
) -> &'a PyList {
    // better to use un parallel version if jobs < 2
    let jobs = if jobs < 1 { 1 } else { jobs };

    // setup threading pool
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs)
        .build()
        .unwrap();

    // channels to send task and receive results
    let (send_task, get_task) = crossbeam_channel::unbounded();
    let (send_result, get_result) = crossbeam_channel::unbounded();

    // init all workers
    for _ in 0..jobs {
        let get_task = get_task.clone();
        let send_result = send_result.clone();

        pool.spawn(move || {
            get_task.iter().for_each(|(i, s): (usize, String)| {
                send_result.send((i, func(s))).unwrap();
            });
        })
    }
    // we don't need this channel after init of all workers
    drop(send_result);

    // converting strings and sending them to workers
    for (i, item) in list.iter(py).enumerate() {
        let string = make_string_fast(py, item).to_string();
        send_task.send((i, string)).unwrap();

        // try get some results if we already have them
        if let Ok((i, o)) = get_result.try_recv() {
            list.set_item(py, i, o.into_py_object(py).into_object());
        }
    }
    //we don't need this channel after sending all tasks
    drop(send_task);
    drop(pool);
    // collecting all remain results
    get_result.iter().for_each(|(i, o)| {
        let item = o.into_py_object(py).into_object();
        list.set_item(py, i, item);
    });

    return list;
}

#[cfg(test)]
mod tests {
    use super::*;
    use cpython::{PyList, Python};
    use rstest::rstest;

    #[rstest(input, case::ascii("hello"), case::utf8("привет"), case::empty(""))]

    fn test_make_string(input: &str) {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let o = py
            .eval(format!("'{}'", input).as_str(), None, None)
            .unwrap();
        let string = make_string_fast(py, o);

        assert_eq!(string, input.to_string())
    }

    #[test]
    fn test_map_pylist() {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let expected = "HELLO_ПРИВЕТ";

        let l = py
            .eval("['hello_привет' for _ in range(10)]", None, None)
            .unwrap()
            .cast_into::<PyList>(py)
            .unwrap();

        let list = map_pylist(py, &l, |s| s.to_uppercase());

        assert_eq!(list.len(py), 10);

        list.iter(py).for_each(|o| {
            let s = make_string_fast(py, o);
            assert_eq!(s, expected)
        })
    }

    #[test]
    fn test_map_pylist_inplace() {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let expected = "HELLO_ПРИВЕТ";

        let l = py
            .eval("['hello_привет' for _ in range(10)]", None, None)
            .unwrap()
            .cast_into::<PyList>(py)
            .unwrap();

        map_pylist_inplace(py, &l, |s| s.to_uppercase());

        assert_eq!(l.len(py), 10);

        l.iter(py).for_each(|o| {
            let s = make_string_fast(py, o);
            assert_eq!(s, expected)
        })
    }

    #[test]
    fn test_map_pylist_inplace_parallel() {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let expected = "HELLO_ПРИВЕТ";

        let l = py
            .eval("['hello_привет' for _ in range(10)]", None, None)
            .unwrap()
            .cast_into::<PyList>(py)
            .unwrap();

        map_pylist_inplace_parallel(py, &l, 2, |s| s.to_uppercase());

        assert_eq!(l.len(py), 10);
        l.iter(py).for_each(|o| {
            let s = make_string_fast(py, o);
            assert_eq!(s, expected)
        })
    }

    #[test]
    fn test_map_pylist_inplace_parallel_zero_jobs() {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let expected = "HELLO_ПРИВЕТ";

        let l = py
            .eval("['hello_привет' for _ in range(10)]", None, None)
            .unwrap()
            .cast_into::<PyList>(py)
            .unwrap();

        map_pylist_inplace_parallel(py, &l, 0, |s| s.to_uppercase());

        assert_eq!(l.len(py), 10);
        l.iter(py).for_each(|o| {
            let s = make_string_fast(py, o);
            assert_eq!(s, expected)
        })
    }
}
