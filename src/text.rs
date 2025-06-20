use regex::Regex;
use pyo3::ffi;
use pyo3::{prelude::*, Bound};
use crate::pystring::make_compact_unicode;

// Wrapper types for performance optimization
#[derive(Clone, Debug)]
pub struct FastString(String);

#[derive(Clone, Debug)]
pub struct FastStringVec(Vec<String>);

#[derive(Clone, Debug)]
pub struct FastBool(bool);

// Default implementations
impl Default for FastString {
    fn default() -> Self {
        FastString(String::new())
    }
}

impl Default for FastStringVec {
    fn default() -> Self {
        FastStringVec(Vec::new())
    }
}

impl Default for FastBool {
    fn default() -> Self {
        FastBool(false)
    }
}

// Send + Sync are automatically derived for these types since String and Vec<String> implement them

// IntoPyObject implementations using manual allocation
impl<'py> pyo3::conversion::IntoPyObject<'py> for FastString {
    type Target = pyo3::PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        unsafe {
            let ptr = make_compact_unicode(&self.0);
            Ok(Bound::from_owned_ptr(py, ptr))
        }
    }
}

impl<'py> pyo3::conversion::IntoPyObject<'py> for FastBool {
    type Target = pyo3::PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        unsafe {
            let ptr = if self.0 {
                ffi::Py_True()
            } else {
                ffi::Py_False()
            };
            ffi::Py_INCREF(ptr);
            Ok(Bound::from_owned_ptr(py, ptr))
        }
    }
}

impl<'py> pyo3::conversion::IntoPyObject<'py> for FastStringVec {
    type Target = pyo3::PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        unsafe {
            let list_ptr = ffi::PyList_New(self.0.len() as isize);
            if list_ptr.is_null() {
                return Err(pyo3::exceptions::PyMemoryError::new_err("Failed to create list"));
            }
            
            for (i, s) in self.0.iter().enumerate() {
                let str_ptr = make_compact_unicode(s);
                ffi::PyList_SetItem(list_ptr, i as isize, str_ptr);
            }
            Ok(Bound::from_owned_ptr(py, list_ptr))
        }
    }
}

pub fn find_in_string(string: &str, pattern: &Regex) -> FastString {
    let mat = pattern.find(string);
    FastString(mat.map(|x| x.as_str()).unwrap_or("").to_string())
}

pub fn is_match_in_string(string: &str, pattern: &Regex) -> FastBool {
    FastBool(pattern.is_match(string))
}

pub fn capture_regex_in_string(string: &str, pattern: &Regex) -> FastStringVec {
    let result = pattern
        .captures(string)
        .map(|caps| {
            caps.iter()
                .map(|m| m.map(|m| m.as_str()).unwrap_or(""))
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_else(Vec::new);
    FastStringVec(result)
}

pub fn split_by_regexp_string(string: &str, pattern: &Regex) -> FastStringVec {
    let result = pattern.split(string).map(|s| s.to_string()).collect();
    FastStringVec(result)
}

pub fn replace_regexp_in_string(
    string: &str,
    pattern: &Regex,
    replacement: &str,
    count: usize,
) -> FastString {
    let result = if count == 0 {
        pattern.replace_all(string, replacement).to_string()
    } else {
        pattern.replacen(string, count, replacement).to_string()
    };
    FastString(result)
}
