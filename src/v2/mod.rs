use core::num;
use itertools::Itertools;
use pyo3::{
    Bound, Py, PyObject, PyResult, PyTypeInfo, Python,
    types::{PyAnyMethods, PyList, PyListMethods, PyString},
};
use smallvec::SmallVec;
use std::borrow::{Borrow, Cow};

use std::iter;

type StringTx<'a> = crossbeam_channel::Sender<(usize, PtrRef<'a>)>;
type StringRx<'a> = crossbeam_channel::Receiver<(usize, PtrRef<'a>)>;

pub fn copy_string_list(list: Py<PyList>, threads: usize) -> PyResult<Py<PyList>> {
    // Aquire GIL for the duration of the operation
    let list = OwnedPyList::from(list);

    // Handle empty list case, as chunks iterator will panic on empty list
    if list.len() == 0 {
        return Ok(Python::with_gil(|py| PyList::empty(py).unbind()));
    }

    let chunks = list.chunks(threads);

    let (result_tx, result_rx) = crossbeam_channel::bounded(list.len().max(1024));

    // Aquire GIL for the duration of the operation
    // This avoids potential corruption, if Python interpreter runs in other threads
    Python::with_gil(|_py| {
        if threads > 1 {
            with_pool(threads, |s| copy_string_list_impl(s, chunks, result_tx));
        } else {
            copy_string_list_worker(list.chunks(1).next().unwrap(), result_tx);
        }
    });

    let collected: Py<PyList> = Python::with_gil(|py| {
        // Collecting into Vec seems unavoidable, if we want to
        // benefit from known-size list creation
        //
        // Potential optimization: atomic counter for resulting list length and custom KnownSizeIterator
        let received = result_rx
            .into_iter()
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .map(|(_, obj)| unsafe { PyObject::from_owned_ptr(py, obj.into_owned().inner) })
            .collect::<Vec<_>>();

        PyList::new(py, received).unwrap().unbind()
    });

    Ok(collected)
}

#[inline]
fn copy_string_list_impl<'scope>(
    pool: &rayon::Scope<'scope>,
    chunks: impl Iterator<Item = BorrowedPyList<'scope>>,
    result_tx: StringTx<'scope>,
) {
    // Critical section: only read-only operations should be performed inside
    for chunk in chunks {
        let result_tx = result_tx.clone();
        pool.spawn(|s| copy_string_list_worker(chunk, result_tx));
    }
}

// Critical section: only read-only operations should be performed inside
#[inline]
fn copy_string_list_worker<'a>(chunk: BorrowedPyList<'a>, results: StringTx<'a>) {
    for (idx, item) in chunk.iter() {
        results.send((idx, item)).unwrap();
    }
}

#[inline]
fn with_pool<'a, F, R>(threads: usize, f: F) -> R
where
    F: FnOnce(&rayon::Scope<'a>) -> R + Send,
    R: Send,
{
    use rayon::ThreadPoolBuilder;

    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to create thread pool");

    pool.scope(|s| f(s))
}

struct Ptr<'a> {
    inner: *mut pyo3::ffi::PyObject,
    phantom: std::marker::PhantomData<&'a ()>,
}

unsafe impl Send for Ptr<'_> {}
unsafe impl Sync for Ptr<'_> {}

#[derive(Clone, Copy)]
struct PtrRef<'a> {
    inner: *mut pyo3::ffi::PyObject,
    phantom: std::marker::PhantomData<&'a ()>,
}

unsafe impl Send for PtrRef<'_> {}

impl<'a> PtrRef<'a> {
    #[inline]
    pub fn as_ptr(&self) -> Ptr<'a> {
        Ptr {
            inner: self.inner,
            phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub unsafe fn into_owned(self) -> Ptr<'static> {
        Ptr {
            inner: self.inner,
            phantom: std::marker::PhantomData,
        }
    }
}

struct OwnedPyList {
    items: SmallVec<[Ptr<'static>; 32]>,
}

impl From<Py<PyList>> for OwnedPyList {
    fn from(list: Py<PyList>) -> Self {
        Python::with_gil(|py| {
            let list = list.bind(py);
            let items = list
                .iter()
                .map(|item| Ptr {
                    inner: item.unbind().into_ptr(),
                    phantom: std::marker::PhantomData,
                })
                .collect();
            OwnedPyList { items }
        })
    }
}

impl Into<Py<PyList>> for OwnedPyList {
    #[inline]
    fn into(self) -> Py<PyList> {
        Python::with_gil(|py| {
            let items = self
                .items
                .into_iter()
                .map(|item| unsafe { PyObject::from_owned_ptr(py, item.inner) });
            PyList::new(py, items).unwrap().unbind()
        })
    }
}

impl OwnedPyList {
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn chunks<'a>(
        &'a self,
        num_chunks: usize,
    ) -> impl Iterator<Item = BorrowedPyList<'a>> + 'a {
        let len = self.len();
        if len == 0 {
            panic!("Cannot chunk an empty list");
        }

        let num_chunks = num_chunks.min(len);
        let chunk_size = len / num_chunks;
        (0..num_chunks).map(move |i| {
            let start = i * chunk_size;
            let end = if i + 1 >= num_chunks {
                len
            } else {
                start + chunk_size
            };
            let items: &'a [Ptr<'a>] = unsafe { std::mem::transmute(&self.items[start..end]) };
            BorrowedPyList {
                start_idx: start,
                items,
            }
        })
    }
}

struct BorrowedPyList<'a> {
    items: &'a [Ptr<'a>],
    start_idx: usize,
}

impl<'a> BorrowedPyList<'a> {
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (usize, PtrRef<'a>)> {
        self.items.iter().enumerate().map(|(idx, item)| {
            (
                self.start_idx + idx,
                PtrRef {
                    inner: item.inner,
                    phantom: std::marker::PhantomData,
                },
            )
        })
    }
}
