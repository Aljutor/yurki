use cpython::{py_fn, py_module_initializer, PyList, PyResult, PyString, Python};
use regex::Regex;

pub mod core;
pub mod text;

py_module_initializer!(yurki, |py, m| {
    m.add(py, "__doc__", "Fast NLP tools")?;
    m.add(
        py,
        "find_in_string",
        py_fn!(
            py,
            find_in_string(list: PyList, pattern: PyString, jobs: usize, inplace: bool)
        ),
    )?;
    Ok(())
});

pub fn find_in_string(
    py: Python,
    list: PyList,
    pattern: PyString,
    jobs: usize,
    inplace: bool,
) -> PyResult<PyList> {
    let pattern = Regex::new(&pattern.to_string(py).unwrap()).unwrap();

    let make_func = move || {
        let pattern = pattern.clone();
        move |s: &str| text::find_in_string(s, &pattern)
    };

    let list = core::map_pylist(py, list, jobs, inplace, make_func);
    Ok(list)
}
