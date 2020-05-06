use cpython::{
    PyList, PyResult, Python, py_fn, py_module_initializer
};

pub mod core;
pub mod text;

py_module_initializer!(yurki, |py, m| {
    m.add(py, "__doc__", "Fast NLP tools")?;
    m.add(py, "to_uppercase", py_fn!(py, to_uppercase(list: PyList)))?;
    Ok(())
});

pub fn to_uppercase(py: Python, list: PyList) -> PyResult<PyList>{
    core::map_pylist_inplace_parallel(py, &list, 4,|s|{
        s.to_uppercase()
    });

    return Ok(list)
}
