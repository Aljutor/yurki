use cpython::{
    py_fn, py_module_initializer, PyList, PyResult, PyString, PyTuple, Python, ToPyObject,
};
use regex::Regex;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

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
    m.add(
        py,
        "tokenize_string",
        py_fn!(
            py,
            tokenize_string(
                list: PyList,
                ngrams: (usize, usize),
                jobs: usize,
                inplace: bool
            )
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

pub fn tokenize_string(
    py: Python,
    list: PyList,
    ngrams: (usize, usize),
    jobs: usize,
    inplace: bool,
) -> PyResult<PyTuple> {
    let vocab = Arc::new(RwLock::new(HashMap::<String, usize>::new()));

    let make_func = || {
        let vocab = Arc::clone(&vocab);

        move |s: &str| {
            let mut counter = HashMap::<usize, usize>::new();

            text::tokenize_word_bound(s, ngrams).iter().for_each(|t| {
                // TODO somehow make this look nice and fast
                let key = {
                    let r_vocab = vocab.read().unwrap();
                    let key = r_vocab.get(t);

                    if key.is_some() {
                        key.unwrap().to_owned()
                    } else {
                        drop(r_vocab);
                        let mut w_vocab = vocab.write().unwrap();

                        let key = w_vocab.get(t);
                        let key = if key.is_some() {
                            key.unwrap().to_owned()
                        } else {
                            let key = w_vocab.len();
                            w_vocab.insert(t.to_owned(), key);
                            key
                        };
                        key
                    }
                };
                counter.entry(key).and_modify(|e| *e += 1).or_insert(1);
            });

            let mut column = Vec::<usize>::with_capacity(counter.len());
            let mut values = Vec::<usize>::with_capacity(counter.len());

            counter.iter().for_each(|(c, v)| {
                column.push(c.to_owned());
                values.push(v.to_owned());
            });

            (column, values)
        }
    };

    let list = core::map_pylist(py, list, jobs, inplace, make_func);
    let vocab = Arc::try_unwrap(vocab).unwrap().into_inner().unwrap();
    Ok((list, vocab).into_py_object(py))
}
