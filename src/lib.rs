use cpython::{py_fn, py_module_initializer, PyList, PyResult, PyString, Python};

pub mod core;
pub mod text;

use regex::Regex;

py_module_initializer!(yurki, |py, m| {
    m.add(py, "__doc__", "Fast NLP tools")?;
    m.add(
        py,
        "find_in_string",
        py_fn!(
            py,
            find_in_string(list: PyList, pattern: PyString, jobs: usize)
        ),
    )?;
    Ok(())
});

fn _find_in_string(string: &str, pattern: &Regex) -> String {
    let mat = pattern.find(string);
    return mat.map(|x| x.as_str()).unwrap_or("").to_string();
}

pub fn find_in_string(
    py: Python,
    list: PyList,
    pattern: PyString,
    jobs: usize,
) -> PyResult<PyList> {
    let pattern = Regex::new(&pattern.to_string(py).unwrap()).unwrap();

    let make_func = move || {
        let pattern = pattern.clone();
        return move |s: &str| _find_in_string(s, &pattern);
    };

    core::map_pylist_inplace_par(py, &list, jobs, make_func);

    return Ok(list);
}
