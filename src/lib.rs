use cpython::{py_fn, py_module_initializer, PyResult, Python};

py_module_initializer!(yurki, |py, m| {
    m.add(py, "__doc__", "Fast NLP tools")?;
    m.add(py, "stub", py_fn!(py, stub()))?;
    Ok(())
});

fn stub(_: Python) -> PyResult<String> {
    let out = "stub".to_string();
    Ok(out)
}
