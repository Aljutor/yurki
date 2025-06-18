#![allow(unsafe_op_in_unsafe_fn)]

use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use regex::RegexBuilder;

pub mod core;
pub mod text;
pub mod v2;

#[pymodule(gil_used = false)]
mod yurki {
    use super::*;

    #[pymodule(gil_used = false)]
    mod internal {
        use super::*;

        #[pyfunction]
        fn copy_string_list_v2(
            py: Python,
            list: &Bound<PyList>,
            jobs: usize,
        ) -> PyResult<Py<PyList>> {
            let list = list.clone().unbind();
            py.allow_threads(|| v2::copy_string_list(list, jobs))
        }

        #[pyfunction]
        fn copy_string_list(py: Python, list: &Bound<PyList>, jobs: usize) -> PyResult<Py<PyList>> {
            let make_func = move || move |s: &str| text::copy_string(s);

            let list = core::map_pylist(py, list, jobs, false, make_func)?;
            Ok(list)
        }

        #[pyfunction]
        fn find_regex_in_string(
            py: Python,
            list: &Bound<PyList>,
            pattern: &Bound<PyString>,
            case: bool,
            jobs: usize,
            inplace: bool,
        ) -> PyResult<Py<PyList>> {
            let pattern = RegexBuilder::new(&pattern.to_string())
                .case_insensitive(case)
                .build()
                .unwrap();

            let pattern = &*Box::leak(Box::new(pattern));

            let make_func = move || {
                let pattern = pattern;
                move |s: &str| text::find_in_string(s, &pattern)
            };

            let list = core::map_pylist(py, list, jobs, inplace, make_func)?;
            Ok(list)
        }

        #[pyfunction]
        fn is_match_regex_in_string(
            py: Python,
            list: &Bound<PyList>,
            pattern: &Bound<PyString>,
            case: bool,
            jobs: usize,
            inplace: bool,
        ) -> PyResult<Py<PyList>> {
            let pattern = RegexBuilder::new(&pattern.to_string())
                .case_insensitive(case)
                .build()
                .unwrap();

            let pattern = &*Box::leak(Box::new(pattern));

            let make_func = move || {
                let pattern = pattern;
                move |s: &str| text::is_match_in_string(s, &pattern)
            };

            let list = core::map_pylist(py, list, jobs, inplace, make_func)?;
            Ok(list)
        }

        #[pyfunction]
        fn capture_regex_in_string(
            py: Python,
            list: &Bound<PyList>,
            pattern: &Bound<PyString>,
            case: bool,
            jobs: usize,
            inplace: bool,
        ) -> PyResult<Py<PyList>> {
            let pattern = RegexBuilder::new(&pattern.to_string())
                .case_insensitive(case)
                .build()
                .unwrap();

            let pattern = &*Box::leak(Box::new(pattern));

            let make_func = move || {
                let pattern = pattern;
                move |s: &str| text::capture_regex_in_string(s, &pattern)
            };

            let list = core::map_pylist(py, list, jobs, inplace, make_func)?;
            Ok(list)
        }

        #[pyfunction]
        fn split_by_regexp_string(
            py: Python,
            list: &Bound<PyList>,
            pattern: &Bound<PyString>,
            case: bool,
            jobs: usize,
            inplace: bool,
        ) -> PyResult<Py<PyList>> {
            let pattern = RegexBuilder::new(&pattern.to_string())
                .case_insensitive(case)
                .build()
                .unwrap();

            let pattern = &*Box::leak(Box::new(pattern));

            let make_func = move || {
                let pattern = pattern;
                move |s: &str| text::split_by_regexp_string(s, &pattern)
            };

            let list = core::map_pylist(py, list, jobs, inplace, make_func)?;
            Ok(list)
        }

        #[pyfunction]
        fn replace_regexp_in_string(
            py: Python,
            list: &Bound<PyList>,
            pattern: &Bound<PyString>,
            replacement: &Bound<PyString>,
            count: usize,
            case: bool,
            jobs: usize,
            inplace: bool,
        ) -> PyResult<Py<PyList>> {
            let pattern = RegexBuilder::new(&pattern.to_string())
                .case_insensitive(case)
                .build()
                .unwrap();
            let pattern = &*Box::leak(Box::new(pattern));

            let replacement_str = replacement.to_string();
            let replacement_str = &*Box::leak(Box::new(replacement_str));

            let make_func = move || {
                let pattern = pattern;
                let replacement = replacement_str;
                move |s: &str| text::replace_regexp_in_string(s, &pattern, &replacement, count)
            };

            let list = core::map_pylist(py, list, jobs, inplace, make_func)?;
            Ok(list)
        }

        /// Hack: workaround for https://github.com/PyO3/pyo3/issues/759
        #[pymodule_init]
        fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
            Python::with_gil(|py| {
                Python::import(py, "sys")?
                    .getattr("modules")?
                    .set_item("yurki.internal", m)
            })
        }
    }
}
