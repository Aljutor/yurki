use pyo3::ffi as pyo3_ffi;
use std::slice;
use crate::fast_string::{ucs1_to_utf8, ucs2_to_utf8, ucs4_to_utf8};

/// Main public interface - SIMD-accelerated Python string conversion with bumpalo integration
pub fn make_string_fast<'a>(
    o: *mut pyo3_ffi::PyObject,
    bump: &'a bumpalo::Bump,
) -> bumpalo::collections::String<'a> {
    unsafe {
        // Asserts for sanity
        assert!(!o.is_null());
        assert!(pyo3_ffi::PyUnicode_Check(o) != 0);

        if pyo3_ffi::PyUnicode_READY(o) != 0 {
            panic!("PyUnicode_READY failed");
        }

        let len = pyo3_ffi::PyUnicode_GET_LENGTH(o) as usize;
        let kind = pyo3_ffi::PyUnicode_KIND(o);
        let data = pyo3_ffi::PyUnicode_DATA(o);

        // Use the new portable SIMD functions and allocate into bump
        let utf8_bytes = match kind {
            pyo3_ffi::PyUnicode_1BYTE_KIND => {
                let chars = slice::from_raw_parts(data as *const u8, len);
                ucs1_to_utf8(chars)
            }
            pyo3_ffi::PyUnicode_2BYTE_KIND => {
                let chars = slice::from_raw_parts(data as *const u16, len);
                ucs2_to_utf8(chars)
            }
            pyo3_ffi::PyUnicode_4BYTE_KIND => {
                let chars = slice::from_raw_parts(data as *const u32, len);
                ucs4_to_utf8(chars)
            }
            _ => panic!("Unknown Unicode kind"),
        };

        // Move the UTF-8 bytes into the bump allocator
        let mut bump_vec = bumpalo::collections::Vec::with_capacity_in(utf8_bytes.len(), bump);
        bump_vec.extend_from_slice(&utf8_bytes);
        
        bumpalo::collections::String::from_utf8_unchecked(bump_vec)
    }
}
