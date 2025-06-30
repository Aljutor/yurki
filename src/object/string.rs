use pyo3::{ffi, prelude::*};
use std::{alloc, mem, ptr};

use crate::simd;
use crate::debug_println;

/// Allocate bytes with usize alignment.
#[inline(always)]
unsafe fn internal_alloc_bytes(size: usize) -> *mut u8 {
    let layout =
        alloc::Layout::from_size_align(size, mem::align_of::<usize>()).expect("invalid layout");
    alloc::alloc(layout)
}

/// Free block with original size for layout consistency.
#[inline(always)]
unsafe fn internal_free_bytes(ptr: *mut std::ffi::c_void, size: usize) {
    let layout =
        alloc::Layout::from_size_align(size, mem::align_of::<usize>()).expect("invalid layout");
    alloc::dealloc(ptr as *mut u8, layout)
}

// String type definition

static mut STRING_TYPE: *mut ffi::PyTypeObject = std::ptr::null_mut();

unsafe extern "C" fn string_alloc(
    type_object: *mut ffi::PyTypeObject,
    item_count: ffi::Py_ssize_t,
) -> *mut ffi::PyObject {
    let size = ((*type_object).tp_basicsize as isize
        + item_count * (*type_object).tp_itemsize as isize) as usize;
    let p = internal_alloc_bytes(size) as *mut ffi::PyObject;
    if p.is_null() {
        ffi::PyErr_NoMemory();
    }
    p
}
/// tp_dealloc runs before tp_free
unsafe extern "C" fn string_dealloc(obj: *mut ffi::PyObject) {
    debug_println!("string_dealloc ▶ {:?}", obj);
    // Nothing special to clean for a plain str
    ffi::Py_TYPE(obj).as_ref().unwrap().tp_free.unwrap()(obj as _);
    debug_println!("string_dealloc ◀");
}

/// tp_free for yurki.String with debug tracing
unsafe extern "C" fn string_free(obj: *mut std::ffi::c_void) {
    debug_println!("string_free ▶ called with obj {:p}", obj);
    if obj.is_null() {
        panic!("string_free: obj is NULL");
    }

    // Header & sanity check
    let py_object = obj as *mut ffi::PyObject;
    let string_type = ptr::read(ptr::addr_of!(STRING_TYPE));
    debug_println!(
        "  ob_type = {:p}  STRING_TYPE = {:p}",
        (*py_object).ob_type,
        string_type
    );

    if string_type.is_null() || (*py_object).ob_type != string_type {
        debug_println!("  not a String instance – returning");
        return;
    }

    let _refcnt = ptr::read(&(*py_object).ob_refcnt as *const _ as *const ffi::Py_ssize_t);
    debug_println!("  refcnt  = {}", _refcnt);

    // Decode string layout
    let ascii = obj as *mut ffi::PyASCIIObject;
    let character_count = (*ascii).length as usize;
    let flags = ptr::read(&(*ascii).state as *const _ as *const u32);
    let element_size = match (flags >> 2) & 0b111 {
        1 => 1,
        2 => 2,
        _ => 4,
    };
    let _is_ascii = ((flags >> 5) & 1) == 1;

    debug_println!(
        "  character_count = {character_count}, element_size = {element_size}, flags = 0x{flags:x}, is_ascii={_is_ascii}"
    );

    // Compute total allocation size
    let header_size = (*(*ascii).ob_base.ob_type).tp_basicsize as usize;
    let total_size = header_size + (character_count + 1) * element_size;

    debug_println!("  header_size (tp_basicsize) = {header_size}");
    debug_println!("  total_size to free         = {total_size}");

    if total_size == 0 || total_size > 10_000_000 {
        panic!("string_free: suspicious total_size = {total_size}");
    }

    // Free memory
    debug_println!("  calling internal_free_bytes …");
    internal_free_bytes(obj, total_size);
    debug_println!("string_free ◀ finished (freed {:p})", obj);
}

/// Initialize String type for module.
pub unsafe fn init_string_type(m: *mut ffi::PyObject) -> PyResult<()> {
    let mut slots = [
        ffi::PyType_Slot {
            slot: ffi::Py_tp_base as i32,
            pfunc: &raw mut ffi::PyUnicode_Type as *mut _ as *mut _,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_new as i32,
            pfunc: std::ptr::null_mut(),
        }, // Prevent external instantiation
        ffi::PyType_Slot {
            slot: ffi::Py_tp_alloc as i32,
            pfunc: string_alloc as *mut _,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_dealloc as i32,
            pfunc: string_dealloc as *mut _,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_free as i32,
            pfunc: string_free as *mut _,
        },
        ffi::PyType_Slot {
            slot: 0,
            pfunc: std::ptr::null_mut(),
        },
    ];

    let base_size = unsafe { ffi::PyUnicode_Type.tp_basicsize };

    let mut spec = ffi::PyType_Spec {
        name: b"yurki.String\0".as_ptr() as *const _,
        basicsize: base_size as i32,
        itemsize: 0,
        flags: (ffi::Py_TPFLAGS_DEFAULT
            | ffi::Py_TPFLAGS_UNICODE_SUBCLASS
            | ffi::Py_TPFLAGS_BASETYPE) as u32,
        slots: slots.as_mut_ptr(),
    };

    let typ = ffi::PyType_FromSpec(&mut spec as *mut _) as *mut ffi::PyTypeObject;
    if typ.is_null() {
        return Err(PyErr::fetch(Python::assume_gil_acquired()));
    }

    STRING_TYPE = typ;
    ffi::PyModule_AddObject(m, b"String\0".as_ptr() as *const _ as *mut _, typ as _);
    Ok(())
}

// String creation

/// Create a yurki.String from UTF-8 text.
/// Safety: caller must hold the GIL and `text` must be valid UTF-8.
pub unsafe fn create_fast_string(text: &str) -> *mut ffi::PyObject {
    debug_println!("create_fast_string: input {:?}", text);

    // SIMD-accelerated analysis: get max codepoint and length in one pass
    let (character_count, max_codepoint) = simd::analyze_utf8_simd(text.as_bytes());

    // Choose internal kind / element size
    let (unicode_kind, element_size) = match max_codepoint {
        0x0000..=0x00FF => (ffi::PyUnicode_1BYTE_KIND as u32, 1),
        0x0100..=0xFFFF => (ffi::PyUnicode_2BYTE_KIND as u32, 2),
        _ => (ffi::PyUnicode_4BYTE_KIND as u32, 4),
    };

    // Calculate sizes
    let header_actual = if max_codepoint < 0x80 {
        std::mem::size_of::<ffi::PyASCIIObject>()
    } else {
        std::mem::size_of::<ffi::PyCompactUnicodeObject>()
    };
    let header_padded = (*STRING_TYPE).tp_basicsize as usize;
    let total_bytes = header_padded + (character_count + 1) * element_size;

    // Allocate memory
    let raw = internal_alloc_bytes(total_bytes) as *mut u8;
    if raw.is_null() {
        ffi::PyErr_NoMemory();
        return std::ptr::null_mut();
    }
    std::ptr::write_bytes(raw, 0, total_bytes);
    debug_println!("  alloc {:p}, total_bytes={total_bytes}", raw);

    // PyObject header
    let py_object = raw as *mut ffi::PyVarObject;
    std::ptr::write(
        &mut (*py_object).ob_base.ob_refcnt as *mut _ as *mut ffi::Py_ssize_t,
        1,
    );
    (*py_object).ob_base.ob_type = STRING_TYPE;

    // PyASCII fields
    let ascii_header = &mut *(raw as *mut ffi::PyASCIIObject);
    ascii_header.length = character_count as ffi::Py_ssize_t;
    ascii_header.hash = -1;

    // Bit layout: interned(2) | kind(3) | compact(1) | ascii(1) | ready(1)
    let is_ascii = if max_codepoint < 0x80 { 1 } else { 0 };
    let flags: u32 = (unicode_kind << 2)        // bits 2-4  (1/2/4-BYTE)
                   | (1 << 5)                   // compact = 1 (always)
                   | ((is_ascii as u32) << 6)   // ascii = 0 or 1
                   | (1 << 7); // ready  = 1

    std::ptr::write(&mut ascii_header.state as *mut _ as *mut u32, flags);
    debug_println!("  flags = 0x{flags:x} (is_ascii={is_ascii})");

    // Compact-unicode extras
    if is_ascii == 0 {
        let compact_unicode = &mut *(raw as *mut ffi::PyCompactUnicodeObject);
        compact_unicode.utf8_length = 0;
        compact_unicode.utf8 = std::ptr::null_mut();
    }

    // Copy canonical data just after real header using SIMD
    let payload = raw.add(header_actual);
    match element_size {
        1 => {
            let dst_slice = std::slice::from_raw_parts_mut(payload, character_count);
            let actual_len = simd::utf8_to_ucs1_simd(text.as_bytes(), dst_slice);
            debug_assert_eq!(actual_len, character_count);
        }
        2 => {
            let dst = payload as *mut u16;
            let dst_slice = std::slice::from_raw_parts_mut(dst, character_count);
            let actual_len = simd::utf8_to_ucs2_simd(text.as_bytes(), dst_slice);
            debug_assert_eq!(actual_len, character_count);
        }
        4 => {
            let dst = payload as *mut u32;
            let dst_slice = std::slice::from_raw_parts_mut(dst, character_count);
            let actual_len = simd::utf8_to_ucs4_simd(text.as_bytes(), dst_slice);
            debug_assert_eq!(actual_len, character_count);
        }
        _ => unreachable!(),
    }
    debug_println!("  payload copied @ {:p}", payload);

    raw as *mut ffi::PyObject
}
