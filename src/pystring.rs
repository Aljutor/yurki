// src/lib.rs
#![allow(clippy::cast_possible_truncation, non_camel_case_types)]

use pyo3::ffi;
use std::{
    alloc::{alloc_zeroed, handle_alloc_error, Layout},
    mem, ptr,
};

/// Build a *compact* Unicode (`str`) object manually, using Rust’s global
/// allocator.  Works on CPython 3.12 **and** 3.13-dev.
///
/// # Safety
/// * Returns a `*mut PyObject` with refcount = 1.  Wrap it exactly **once**
///   with `Py::from_owned_ptr` or call `Py_DECREF` manually.
/// * Relies on public PEP-393 structs; once CPython makes them opaque this
///   trick will break.
pub unsafe fn make_compact_unicode(text: &str) -> *mut ffi::PyObject {
    // ── decide element size / PyUnicode_KIND ──────────────────────────────
    let (kind_bits, elem_sz) = match text.chars().map(|c| c as u32).max().unwrap_or(0) {
        0x0000..=0x00FF => (ffi::PyUnicode_1BYTE_KIND as u32, 1),
        0x0100..=0xFFFF => (ffi::PyUnicode_2BYTE_KIND as u32, 2),
        _               => (ffi::PyUnicode_4BYTE_KIND as u32, 4),
    };
    let char_len = text.chars().count();

    // ── allocate header + payload with Rust allocator ─────────────────────
    let header = mem::size_of::<ffi::PyCompactUnicodeObject>();
    let total  = header + (char_len + 1) * elem_sz;               // +NUL
    let layout = Layout::from_size_align(total, mem::align_of::<usize>()).unwrap();
    let raw = alloc_zeroed(layout) as *mut ffi::PyCompactUnicodeObject;
    if raw.is_null() {
        handle_alloc_error(layout);
    }

    // ── initialise the PyObject header portably (sets refcnt & ob_type) ──
    ffi::PyObject_Init(raw.cast(), core::ptr::addr_of_mut!(ffi::PyUnicode_Type));

    // convenience aliases
    let ascii   = &mut (*raw)._base;            // embedded PyASCIIObject
    let payload = (raw as *mut u8).add(header); // canonical code-point data

    // ── fill remaining PyASCIIObject fields ───────────────────────────────
    ascii.length = char_len as ffi::Py_ssize_t;
    ascii.hash   = -1;                          // hash not computed

    // bit-field: interned(2) | kind(3) | compact(1) | ascii(1) | ready(1)
    let flags: u32 = (kind_bits << 2) | (1 << 5) | /*compact*/ (0 << 6) | (1 << 7);
    ptr::write(&mut ascii.state as *mut _ as *mut u32, flags);

    // ── PyCompactUnicodeObject extras (same 3.12 → 3.13) ──────────────────
    (*raw).utf8_length = 0;
    (*raw).utf8        = ptr::null_mut();

    // ── copy canonical data right after the header ────────────────────────
    match elem_sz {
        1 => {
            for (i, b) in text.bytes().enumerate()     { *payload.add(i) = b; }
            *payload.add(char_len) = 0;
        }
        2 => {
            let dst = payload as *mut u16;
            for (i, u) in text.encode_utf16().enumerate() { *dst.add(i) = u; }
            *dst.add(char_len) = 0;
        }
        4 => {
            let dst = payload as *mut u32;
            for (i, ch) in text.chars().enumerate() { *dst.add(i) = ch as u32; }
            *dst.add(char_len) = 0;
        }
        _ => unreachable!(),
    }

    raw.cast()
}

// optional: very small smoke-test
#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::{prelude::*, Py};

    #[test]
    fn round_trip() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| unsafe {
            let raw = make_compact_unicode("Γειά σου Κόσμε");
            let s: Py<PyAny> = Py::from_owned_ptr(py, raw);
            assert_eq!(s.extract::<String>(py).unwrap(), "Γειά σου Κόσμε");
        });
    }
}
