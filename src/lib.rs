#![allow(unsafe_op_in_unsafe_fn)]
#![feature(portable_simd)]
#![feature(min_specialization)]

use crate::converter::ToPyObject;
use mimalloc::MiMalloc;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use regex::RegexBuilder;

// Let's globaly use mimmaloc as allocator
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

// Unified Macro for Debug Message
#[cfg(feature = "debug-yurki-internal")]
macro_rules! debug_println {
    ($($arg:tt)*) => { eprintln!($($arg)*) };
}

#[cfg(not(feature = "debug-yurki-internal"))]
macro_rules! debug_println {
    ($($arg:tt)*) => {};
}

// Export the macro so it can be used in other modules
pub(crate) use debug_println;

pub mod converter;
pub mod core;
pub mod object;
pub mod simd;
pub mod text;

#[pymodule(gil_used = false)]
mod yurki {
    use super::*;

    #[pymodule(gil_used = false)]
    mod internal {
        use super::*;

        #[pyfunction]
        fn find_regex_in_string(
            py: Python,
            list: &Bound<PyList>,
            pattern: &Bound<PyString>,
            case: bool,
            jobs: usize,
            inplace: bool,
        ) -> PyResult<PyObject> {
            let pattern = RegexBuilder::new(&pattern.to_string())
                .case_insensitive(case)
                .build()
                .unwrap();

            let make_func = {
                let pattern = pattern.clone();
                move || unsafe {
                    let pattern = pattern.clone();
                    move |s: &str| text::find_in_string(s, &pattern).to_py_object()
                }
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
        ) -> PyResult<PyObject> {
            let pattern = RegexBuilder::new(&pattern.to_string())
                .case_insensitive(case)
                .build()
                .unwrap();

            let make_func = move || unsafe {
                let pattern = pattern.clone();
                move |s: &str| text::is_match_in_string(s, &pattern).to_py_object()
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
        ) -> PyResult<PyObject> {
            let pattern = RegexBuilder::new(&pattern.to_string())
                .case_insensitive(case)
                .build()
                .unwrap();

            let make_func = move || unsafe {
                let pattern = pattern.clone();
                move |s: &str| text::capture_regex_in_string(s, &pattern).to_py_object()
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
        ) -> PyResult<PyObject> {
            let pattern = RegexBuilder::new(&pattern.to_string())
                .case_insensitive(case)
                .build()
                .unwrap();

            let make_func = move || unsafe {
                let pattern = pattern.clone();
                move |s: &str| text::split_by_regexp_string(s, &pattern).to_py_object()
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
        ) -> PyResult<PyObject> {
            let pattern = RegexBuilder::new(&pattern.to_string())
                .case_insensitive(case)
                .build()
                .unwrap();

            let replacement_str = replacement.to_string();

            let make_func = move || unsafe {
                let pattern = pattern.clone();
                let replacement = replacement_str.clone();
                move |s: &str| {
                    text::replace_regexp_in_string(s, &pattern, &replacement, count).to_py_object()
                }
            };

            let list = core::map_pylist(py, list, jobs, inplace, make_func)?;
            Ok(list)
        }

        /// Hack: workaround for https://github.com/PyO3/pyo3/issues/759
        #[pymodule_init]
        fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
            let _ = Python::with_gil(|py| {
                Python::import(py, "sys")?
                    .getattr("modules")?
                    .set_item("yurki.internal", m)
            });

            unsafe {
                object::init_string_type(m.as_ptr())?;
                object::init_list_type(m.as_ptr())?;
                Ok(())
            }
        }
    }
}
