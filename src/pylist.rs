//! yurki::fast_list  —  immutable list with custom allocator
use mimalloc::MiMalloc;
use pyo3::{ffi, prelude::*};
use std::{
    alloc::{GlobalAlloc, Layout},
    mem, ptr,
    os::raw::c_int
};

//-------------------------------------------
// Debug helper (same one you already have)
use crate::debug_println;
//-------------------------------------------

// ───────────────────────────────────────────
//  Allocation helpers (MiMalloc + GlobalAlloc)
// ───────────────────────────────────────────
static FAST_LIST_ALLOCATOR: MiMalloc = MiMalloc;

/// Allocate `size` bytes aligned to `usize`.
#[inline(always)]
unsafe fn internal_alloc_bytes(size: usize) -> *mut u8 {
    let layout = Layout::from_size_align(size, mem::align_of::<usize>())
        .expect("FastList: invalid layout");
    GlobalAlloc::alloc(&FAST_LIST_ALLOCATOR, layout)
}

/// Free a previously-allocated block.
#[inline(always)]
unsafe fn internal_free_bytes(ptr: *mut std::ffi::c_void, size: usize) {
    let layout = Layout::from_size_align(size, mem::align_of::<usize>())
        .expect("FastList: invalid layout");
    GlobalAlloc::dealloc(&FAST_LIST_ALLOCATOR, ptr as *mut u8, layout)
}

// ───────────────────────────────────────────
//  FastList C-level layout
// ───────────────────────────────────────────
/// Exact copy of `PyListObject` so we can treat
/// it as a true list subclass.
#[repr(C)]
struct PyFastList {
    ob_base: ffi::PyVarObject,      // ob_refcnt / ob_type / ob_size
    ob_item: *mut *mut ffi::PyObject,
    allocated: ffi::Py_ssize_t,
}

// ───────────────────────────────────────────
//  Type object slot implementations
// ───────────────────────────────────────────
static mut FASTLIST_TYPE: *mut ffi::PyTypeObject = ptr::null_mut();

/// Custom tp_alloc — one shot for header + elements.
unsafe extern "C" fn fastlist_alloc(
    subtype: *mut ffi::PyTypeObject,
    item_count: ffi::Py_ssize_t,
) -> *mut ffi::PyObject {
    debug_println!("fastlist_alloc ▶ subtype={:p} items={item_count}", subtype);

    let header = (*subtype).tp_basicsize as usize;
    let elements = if item_count < 0 { 0 } else { item_count as usize };
    let total_size = header + elements * mem::size_of::<*mut ffi::PyObject>();

    let raw = internal_alloc_bytes(total_size) as *mut PyFastList;
    if raw.is_null() {
        ffi::PyErr_NoMemory();
        return ptr::null_mut();
    }
    ptr::write_bytes(raw as *mut u8, 0, total_size);

    // Initialise ob_refcnt / ob_type / ob_size
    let var = &mut (*raw).ob_base;
    std::ptr::write(
    &mut (*var).ob_base.ob_refcnt as *mut _ as *mut ffi::Py_ssize_t,
    1,
    );
    var.ob_base.ob_type = subtype;
    var.ob_size = item_count;

    // Data area immediately after the struct
    if elements > 0 {
        (*raw).ob_item = (raw as *mut u8).add(header) as *mut *mut ffi::PyObject;
        (*raw).allocated = item_count;
    } else {
        (*raw).ob_item = ptr::null_mut();
        (*raw).allocated = 0;
    }

    debug_println!(
        "fastlist_alloc ◀ raw={:p}, header={header}, total={total_size}",
        raw
    );
    raw as *mut ffi::PyObject
}

/// tp_dealloc – decref each element, then call tp_free.
unsafe extern "C" fn fastlist_dealloc(obj: *mut ffi::PyObject) {
    debug_println!("fastlist_dealloc ▶ obj={:p}", obj);
    let fl = obj as *mut PyFastList;
    let n = (*fl).ob_base.ob_size;
    for i in 0..n {
        let it_ptr = *(*fl).ob_item.add(i as usize);
        if !it_ptr.is_null() {
            ffi::Py_DECREF(it_ptr);
        }
    }
    // Delegate to tp_free (our custom free)
    ffi::Py_TYPE(obj).as_ref().unwrap().tp_free.unwrap()(obj as _);
    debug_println!("fastlist_dealloc ◀");
}

/// tp_free – actual memory release through mimalloc.
unsafe extern "C" fn fastlist_free(ptr_: *mut std::ffi::c_void) {
    // Reconstruct size to free
    let fl = ptr_ as *mut PyFastList;
    let header = (*(*fl).ob_base.ob_base.ob_type).tp_basicsize as usize;
    let items = (*fl).ob_base.ob_size as usize;
    let total = header + items * mem::size_of::<*mut ffi::PyObject>();

    debug_println!(
        "fastlist_free ▶ ptr={:p} header={header} items={items} total={total}",
        ptr_
    );
    internal_free_bytes(ptr_, total);
    debug_println!("fastlist_free ◀");
}

// ───────────────────────────────────────────
//  Type initialisation
// ───────────────────────────────────────────
pub unsafe fn init_fastlist_type(m: *mut ffi::PyObject) -> PyResult<()> {
    // Slots table
    let mut slots = [
        ffi::PyType_Slot {
            slot: ffi::Py_tp_base as c_int,
            pfunc: &raw mut ffi::PyList_Type as *mut _ as *mut _,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_new as c_int,
            pfunc: ptr::null_mut(), // block Python-side instantiation
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_alloc as c_int,
            pfunc: fastlist_alloc as *mut _,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_dealloc as c_int,
            pfunc: fastlist_dealloc as *mut _,
        },
        ffi::PyType_Slot {
            slot: ffi::Py_tp_free as c_int,
            pfunc: fastlist_free as *mut _,
        },
        ffi::PyType_Slot {
            slot: 0,
            pfunc: ptr::null_mut(),
        },
    ];

    // Build type spec
    let mut spec = ffi::PyType_Spec {
        name: b"yurki.FastList\0".as_ptr() as *const _,
        basicsize: mem::size_of::<PyFastList>() as c_int,
        itemsize: mem::size_of::<*mut ffi::PyObject>() as c_int,
        flags: (ffi::Py_TPFLAGS_DEFAULT
            | ffi::Py_TPFLAGS_LIST_SUBCLASS
            | ffi::Py_TPFLAGS_BASETYPE) as u32,
        slots: slots.as_mut_ptr(),
    };

    let typ = ffi::PyType_FromSpec(&mut spec) as *mut ffi::PyTypeObject;
    if typ.is_null() {
        return Err(PyErr::fetch(Python::assume_gil_acquired()));
    }
    FASTLIST_TYPE = typ;
    ffi::PyModule_AddObject(m, b"FastList\0".as_ptr() as *const _ as *mut _, typ as _);
    Ok(())
}

// ───────────────────────────────────────────
//  Rust-side constructor (GIL **not** required)
// ───────────────────────────────────────────
/// Create a `yurki.FastList` from a slice of `*mut PyObject`.
///
/// ⚠️  Safety:
/// * Caller must **eventually** hold the GIL before handing the
///   resulting object to Python code.
/// * Every element in `items` must be a valid (live) `PyObject*`.
pub unsafe fn create_fast_list(items: &[*mut ffi::PyObject]) -> *mut ffi::PyObject {
    debug_println!("create_fast_list ▶ len={}", items.len());

    // Allocate FastList object
    let obj = fastlist_alloc(FASTLIST_TYPE, items.len() as ffi::Py_ssize_t);
    if obj.is_null() {
        return ptr::null_mut();
    }
    let fl = obj as *mut PyFastList;

    // Copy pointers + INCREF (needs GIL, so we do it only if we have the GIL)
    if ffi::PyGILState_Check() != 0 {
        for (i, &it) in items.iter().enumerate() {
            ffi::Py_INCREF(it);
            *(*fl).ob_item.add(i) = it;
        }
    } else {
        // GIL not held; just copy raw pointers – caller must keep them alive.
        ptr::copy_nonoverlapping(
            items.as_ptr(),
            (*fl).ob_item,
            items.len(),
        );
    }

    debug_println!("create_fast_list ◀ obj={:p}", obj);
    obj
}

/// Create empty FastList with pre-allocated space (like PyList_New)
pub unsafe fn create_fast_list_empty(size: isize) -> *mut ffi::PyObject {
    debug_println!("create_fast_list_empty ▶ size={}", size);
    
    if size <= 0 {
        return create_fast_list(&[]); // Empty list
    }
    
    let obj = fastlist_alloc(FASTLIST_TYPE, size);
    if obj.is_null() {
        return ptr::null_mut();
    }
    
    debug_println!("create_fast_list_empty ◀ obj={:p}", obj);
    obj
}

/// Set item at index with ownership transfer (no INCREF)
pub unsafe fn fast_list_set_item_transfer(
    list: *mut ffi::PyObject, 
    index: isize, 
    item: *mut ffi::PyObject
) {
    debug_println!("fast_list_set_item_transfer ▶ list={:p} index={} item={:p}", list, index, item);
    let fl = list as *mut PyFastList;
    *(*fl).ob_item.add(index as usize) = item;
    debug_println!("fast_list_set_item_transfer ◀");
}
