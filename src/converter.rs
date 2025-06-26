#![allow(dead_code)]

use crate::core::PyObjectPtr;
use crate::pylist::{create_fast_list_empty, fast_list_set_item_transfer};
use crate::pystring::create_fast_string;
use parking_lot::Mutex;
use pyo3::ffi as pyo3_ffi;
use std::borrow::Cow;

// Global mutex for Python FFI operations
pub static PYTHON_FFI_MUTEX: Mutex<()> = Mutex::new(());

/// Trait to determine conversion strategy at compile time
pub trait ConversionStrategy {
    const THREAD_SAFE: bool;
}

/// Trait for converting Rust types to Python objects in worker threads
pub trait ToPyObject: ConversionStrategy {
    unsafe fn to_py_object(self) -> PyObjectPtr;
}

// String implementations
#[cfg(not(feature = "disable-fast-string"))]
impl ConversionStrategy for String {
    const THREAD_SAFE: bool = true;
}

#[cfg(feature = "disable-fast-string")]
impl ConversionStrategy for String {
    const THREAD_SAFE: bool = false; 
}

impl ToPyObject for String {
    #[cfg(not(feature = "disable-fast-string"))]
    unsafe fn to_py_object(self) -> PyObjectPtr {
        PyObjectPtr(create_fast_string(&self))
    }

    #[cfg(feature = "disable-fast-string")]
    unsafe fn to_py_object(self) -> PyObjectPtr {
        PyObjectPtr(pyo3_ffi::PyUnicode_FromStringAndSize(
            self.as_ptr() as *const i8,
            self.len() as isize,
        ))
    }
}

// Bool implementations
impl ConversionStrategy for bool {
    const THREAD_SAFE: bool = true; // Safe to convert in worker thread
}

impl ToPyObject for bool {
    unsafe fn to_py_object(self) -> PyObjectPtr {
        let ptr = if self {
            pyo3_ffi::Py_True()
        } else {
            pyo3_ffi::Py_False()
        };
        PyObjectPtr(ptr)
    }
}

// Vec implementations - use streaming approach with FastList
impl<T> ConversionStrategy for Vec<T>
where
    T: ToPyObject,
{
    const THREAD_SAFE: bool = false; // Needs main thread with GIL
}

impl<T> ToPyObject for Vec<T>
where
    T: ToPyObject,
{
    unsafe fn to_py_object(self) -> PyObjectPtr {
        let len = self.len();
        
        // Pre-allocate FastList with exact size
        let list = create_fast_list_empty(len as isize);
        if list.is_null() {
            return PyObjectPtr(std::ptr::null_mut());
        }
        
        // Stream items directly into FastList
        for (index, item) in self.into_iter().enumerate() {
            let py_obj = item.to_py_object();
            fast_list_set_item_transfer(list, index as isize, py_obj.0);
        }
        
        PyObjectPtr(list)
    }
}

// &str implementations
#[cfg(not(feature = "disable-fast-string"))]
impl ConversionStrategy for &str {
    const THREAD_SAFE: bool = true;
}

#[cfg(feature = "disable-fast-string")]
impl ConversionStrategy for &str {
    const THREAD_SAFE: bool = false;
}

impl ToPyObject for &str {
    #[cfg(not(feature = "disable-fast-string"))]
    unsafe fn to_py_object(self) -> PyObjectPtr {
        PyObjectPtr(create_fast_string(self)) // No mutex - FastString path
    }

    #[cfg(feature = "disable-fast-string")]
    unsafe fn to_py_object(self) -> PyObjectPtr {
        PyObjectPtr(pyo3_ffi::PyUnicode_FromStringAndSize(
            self.as_ptr() as *const i8,
            self.len() as isize,
        ))
    }
}

// Cow<str> implementations
impl ConversionStrategy for Cow<'_, str> {
    const THREAD_SAFE: bool = true; // Safe to convert in worker thread
}

impl ToPyObject for Cow<'_, str> {
    unsafe fn to_py_object(self) -> PyObjectPtr {
        // Convert to &str and reuse that logic
        let s: &str = self.as_ref();
        s.to_py_object()
    }
}

impl ConversionStrategy for Vec<PyObjectPtr> {
    const THREAD_SAFE: bool = false;
}

impl ToPyObject for Vec<PyObjectPtr> {
    unsafe fn to_py_object(self) -> PyObjectPtr {
        let len = self.len();
        
        // Pre-allocate FastList with exact size
        let list = create_fast_list_empty(len as isize);
        if list.is_null() {
            return PyObjectPtr(std::ptr::null_mut());
        }
        
        // Stream PyObjectPtr items directly into FastList
        for (index, py_obj_ptr) in self.into_iter().enumerate() {
            fast_list_set_item_transfer(list, index as isize, py_obj_ptr.0);
        }
        
        PyObjectPtr(list)
    }
}
