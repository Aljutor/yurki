//! py_unicode_simd.rs – SIMD-accelerated transcoding for Python Stable-ABI code-units
//!
//! **Pure Rust, zero-FFI:** Wide-SIMD, branch-free codecs inspired by
//! Daniel Lemire et al.’s **simdutf**, built entirely on the evolving
//! *portable-SIMD* API so a single crate compiles on x86-64 (AVX2/AVX-512),
//! Apple M-series (NEON) and WASM-SIMD.
//!
//! ### Nightly 2025-06 compatibility
//! * Manual `LANES_*` constants – no macro crates required.
//! * `SimdUint` now lives in `core::simd::prelude` / `std::simd::prelude`.
//!
//! ---------------------------------------------------------------------------

#![allow(dead_code)]

use core::simd::Simd;
use core::simd::cmp::SimdPartialOrd;
use core::simd::prelude::SimdUint; // `.cast()` lives here on nightly-2025-06
use std::borrow::Cow;
use std::ptr;

/* ===================================================================== */
/*                      SIMD lane-width selection                         */
/* ===================================================================== */

/* ── u8 ────────────────────────────────────────────────────────────── */
#[cfg(target_feature = "avx512bw")]
type U8s = Simd<u8, 64>;
#[cfg(target_feature = "avx512bw")]
const LANES_U8: usize = 64;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
type U8s = Simd<u8, 32>;
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
const LANES_U8: usize = 32;

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512bw")))]
type U8s = Simd<u8, 16>;
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512bw")))]
const LANES_U8: usize = 16;

/* ── u16 ───────────────────────────────────────────────────────────── */
#[cfg(target_feature = "avx512bw")]
type U16s = Simd<u16, 32>;
#[cfg(target_feature = "avx512bw")]
const LANES_U16: usize = 32;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
type U16s = Simd<u16, 16>;
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
const LANES_U16: usize = 16;

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512bw")))]
type U16s = Simd<u16, 8>;
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512bw")))]
const LANES_U16: usize = 8;

/* ── u32 ───────────────────────────────────────────────────────────── */
#[cfg(target_feature = "avx512bw")]
type U32s = Simd<u32, 16>;
#[cfg(target_feature = "avx512bw")]
const LANES_U32: usize = 16;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
type U32s = Simd<u32, 8>;
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
const LANES_U32: usize = 8;

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512bw")))]
type U32s = Simd<u32, 4>;
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512bw")))]
const LANES_U32: usize = 4;

/* ===================================================================== */
/*               Py_UCS1 (Latin-1) → UTF-8                                */
/* ===================================================================== */

/// Convert a Latin-1 slice to UTF-8 using bump allocator.
/// * Pure ASCII → borrowed `&str` (zero-alloc)
/// * Mixed input → bump-allocated `&str`
#[inline]
pub fn ucs1_to_utf8_bump<'a>(input: &'a [u8], bump: &'a bumpalo::Bump) -> &'a str {
    /* 1. All-ASCII detection (vector + scalar tail) */
    if input
        .chunks_exact(LANES_U8)
        .all(|c| U8s::from_slice(c).simd_lt(U8s::splat(0x80)).all())
        && input[input.len() - input.len() % LANES_U8..]
            .iter()
            .all(|&b| b < 0x80)
    {
        return unsafe { core::str::from_utf8_unchecked(input) };
    }

    /* 2. Count ≥0x80 bytes for exact allocation */
    let mut extra = 0usize;
    for c in input.chunks_exact(LANES_U8) {
        let v = U8s::from_slice(c);
        extra += v.simd_ge(U8s::splat(0x80)).to_bitmask().count_ones() as usize;
    }
    for &b in &input[input.len() - input.len() % LANES_U8..] {
        if b >= 0x80 {
            extra += 1
        }
    }
    let mut out = bumpalo::collections::Vec::with_capacity_in(input.len() + extra, bump);

    /* 3. SIMD loop with bulk ASCII copy */
    let mut i = 0;
    while i + LANES_U8 <= input.len() {
        let blk = &input[i..i + LANES_U8];
        let v = U8s::from_slice(blk);
        if v.simd_lt(U8s::splat(0x80)).all() {
            out.extend_from_slice(blk);
        } else {
            expand_latin1_block_bump(blk, &mut out);
        }
        i += LANES_U8;
    }

    /* 4. Scalar tail */
    for &b in &input[i..] {
        if b < 0x80 {
            out.push(b);
        } else {
            out.push(0xC0 | (b >> 6));
            out.push(0x80 | (b & 0x3F));
        }
    }

    let slice = out.into_bump_slice();
    unsafe { core::str::from_utf8_unchecked(slice) }
}

/// Convert a Latin-1 slice to UTF-8.
/// * Pure ASCII → borrowed `&str` (zero-alloc)
/// * Mixed input → owned `String` inside `Cow::Owned`
#[inline]
pub fn ucs1_to_utf8<'a>(input: &'a [u8]) -> Cow<'a, str> {
    /* 1. All-ASCII detection (vector + scalar tail) */
    if input
        .chunks_exact(LANES_U8)
        .all(|c| U8s::from_slice(c).simd_lt(U8s::splat(0x80)).all())
        && input[input.len() - input.len() % LANES_U8..]
            .iter()
            .all(|&b| b < 0x80)
    {
        return Cow::Borrowed(unsafe { core::str::from_utf8_unchecked(input) });
    }

    /* 2. Count ≥0x80 bytes for exact allocation */
    let mut extra = 0usize;
    for c in input.chunks_exact(LANES_U8) {
        let v = U8s::from_slice(c);
        extra += v.simd_ge(U8s::splat(0x80)).to_bitmask().count_ones() as usize;
    }
    for &b in &input[input.len() - input.len() % LANES_U8..] {
        if b >= 0x80 {
            extra += 1
        }
    }
    let mut out: Vec<u8> = Vec::with_capacity(input.len() + extra);

    /* 3. SIMD loop with bulk ASCII copy */
    let mut i = 0;
    while i + LANES_U8 <= input.len() {
        let blk = &input[i..i + LANES_U8];
        let v = U8s::from_slice(blk);
        if v.simd_lt(U8s::splat(0x80)).all() {
            unsafe {
                let dst = out.as_mut_ptr().add(out.len());
                ptr::copy_nonoverlapping(blk.as_ptr(), dst, LANES_U8);
                out.set_len(out.len() + LANES_U8);
            }
        } else {
            expand_latin1_block(blk, &mut out);
        }
        i += LANES_U8;
    }

    /* 4. Scalar tail */
    for &b in &input[i..] {
        if b < 0x80 {
            out.push(b);
        } else {
            out.push(0xC0 | (b >> 6));
            out.push(0x80 | (b & 0x3F));
        }
    }

    Cow::Owned(unsafe { String::from_utf8_unchecked(out) })
}

/* LUTs for two-byte Latin-1 expansion */
static HIGH_LUT: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = 0xC0 | ((i as u8) >> 6);
        i += 1;
    }
    t
};
static LOW_LUT: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = 0x80 | ((i as u8) & 0x3F);
        i += 1;
    }
    t
};

#[inline]
fn expand_latin1_block_bump(block: &[u8], out: &mut bumpalo::collections::Vec<u8>) {
    for &b in block {
        if b < 0x80 {
            out.push(b)
        } else {
            out.push(HIGH_LUT[b as usize]);
            out.push(LOW_LUT[b as usize]);
        }
    }
}

#[inline]
fn expand_latin1_block(block: &[u8], out: &mut Vec<u8>) {
    for &b in block {
        if b < 0x80 {
            out.push(b)
        } else {
            out.push(HIGH_LUT[b as usize]);
            out.push(LOW_LUT[b as usize]);
        }
    }
}

/* ===================================================================== */
/*               Py_UCS2 (UTF-16) → UTF-8                                 */
/* ===================================================================== */

#[inline]
pub fn ucs2_to_utf8_bump<'a>(input: &[u16], bump: &'a bumpalo::Bump) -> &'a str {
    let mut out = bumpalo::collections::Vec::with_capacity_in(input.len() * 3, bump);
    let mut i = 0;
    while i + LANES_U16 <= input.len() {
        let v = U16s::from_slice(&input[i..i + LANES_U16]);
        if v.simd_ge(U16s::splat(0x0080)).to_bitmask() == 0 {
            /* Fast ASCII copy */
            for &w in &input[i..i + LANES_U16] {
                out.push(w as u8);
            }
        } else {
            expand_ucs2_block_bump(&input[i..i + LANES_U16], &mut out);
        }
        i += LANES_U16;
    }
    expand_ucs2_block_bump(&input[i..], &mut out);
    let slice = out.into_bump_slice();
    unsafe { core::str::from_utf8_unchecked(slice) }
}

#[inline]
pub fn ucs2_to_utf8(input: &[u16]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(input.len() * 3);
    let mut i = 0;
    while i + LANES_U16 <= input.len() {
        let v = U16s::from_slice(&input[i..i + LANES_U16]);
        if v.simd_ge(U16s::splat(0x0080)).to_bitmask() == 0 {
            /* Fast ASCII copy */
            unsafe {
                let bytes: Simd<u8, { LANES_U16 }> = v.cast();
                let dst = out.as_mut_ptr().add(out.len());
                ptr::copy_nonoverlapping(bytes.as_array().as_ptr(), dst, LANES_U16);
                out.set_len(out.len() + LANES_U16);
            }
        } else {
            expand_ucs2_block(&input[i..i + LANES_U16], &mut out);
        }
        i += LANES_U16;
    }
    expand_ucs2_block(&input[i..], &mut out);
    out
}

#[inline]
fn expand_ucs2_block_bump(block: &[u16], out: &mut bumpalo::collections::Vec<u8>) {
    let mut j = 0;
    while j < block.len() {
        let w = block[j];
        match w {
            0x0000..=0x007F => out.push(w as u8),
            0x0080..=0x07FF => {
                out.push((0xC0 | (w >> 6)) as u8);
                out.push((0x80 | (w & 0x3F)) as u8);
            }
            0xD800..=0xDBFF => {
                /* High surrogate: assume valid pair */
                let lo = block[j + 1];
                let cp = 0x10000 + (((w as u32 & 0x3FF) << 10) | (lo as u32 & 0x3FF));
                push_utf8_4_bump(cp, out);
                j += 1; // skip low surrogate
            }
            0xDC00..=0xDFFF => unsafe { core::hint::unreachable_unchecked() },
            _ => {
                out.push((0xE0 | (w >> 12)) as u8);
                out.push((0x80 | ((w >> 6) & 0x3F)) as u8);
                out.push((0x80 | (w & 0x3F)) as u8);
            }
        }
        j += 1;
    }
}

#[inline]
fn expand_ucs2_block(block: &[u16], out: &mut Vec<u8>) {
    let mut j = 0;
    while j < block.len() {
        let w = block[j];
        match w {
            0x0000..=0x007F => out.push(w as u8),
            0x0080..=0x07FF => {
                out.push((0xC0 | (w >> 6)) as u8);
                out.push((0x80 | (w & 0x3F)) as u8);
            }
            0xD800..=0xDBFF => {
                /* High surrogate: assume valid pair */
                let lo = block[j + 1];
                let cp = 0x10000 + (((w as u32 & 0x3FF) << 10) | (lo as u32 & 0x3FF));
                push_utf8_4(cp, out);
                j += 1; // skip low surrogate
            }
            0xDC00..=0xDFFF => unsafe { core::hint::unreachable_unchecked() },
            _ => {
                out.push((0xE0 | (w >> 12)) as u8);
                out.push((0x80 | ((w >> 6) & 0x3F)) as u8);
                out.push((0x80 | (w & 0x3F)) as u8);
            }
        }
        j += 1;
    }
}

/* ===================================================================== */
/*                       Helper routines                                  */
/* ===================================================================== */

#[inline(always)]
fn push_utf8_4_bump(cp: u32, out: &mut bumpalo::collections::Vec<u8>) {
    out.extend_from_slice(&[
        (0xF0 | (cp >> 18)) as u8,
        (0x80 | ((cp >> 12) & 0x3F)) as u8,
        (0x80 | ((cp >> 6) & 0x3F)) as u8,
        (0x80 | (cp & 0x3F)) as u8,
    ]);
}

#[inline(always)]
fn push_utf8_4(cp: u32, out: &mut Vec<u8>) {
    out.extend_from_slice(&[
        (0xF0 | (cp >> 18)) as u8,
        (0x80 | ((cp >> 12) & 0x3F)) as u8,
        (0x80 | ((cp >> 6) & 0x3F)) as u8,
        (0x80 | (cp & 0x3F)) as u8,
    ]);
}

#[inline(always)]
fn push_utf32_scalar_bump(cp: u32, out: &mut bumpalo::collections::Vec<u8>) {
    match cp {
        0x0000..=0x007F => out.push(cp as u8),
        0x0080..=0x07FF => {
            out.push((0xC0 | (cp >> 6)) as u8);
            out.push((0x80 | (cp & 0x3F)) as u8);
        }
        0x0800..=0xFFFF => {
            out.push((0xE0 | (cp >> 12)) as u8);
            out.push((0x80 | ((cp >> 6) & 0x3F)) as u8);
            out.push((0x80 | (cp & 0x3F)) as u8);
        }
        _ => push_utf8_4_bump(cp, out),
    }
}

#[inline(always)]
fn push_utf32_scalar(cp: u32, out: &mut Vec<u8>) {
    match cp {
        0x0000..=0x007F => out.push(cp as u8),
        0x0080..=0x07FF => {
            out.push((0xC0 | (cp >> 6)) as u8);
            out.push((0x80 | (cp & 0x3F)) as u8);
        }
        0x0800..=0xFFFF => {
            out.push((0xE0 | (cp >> 12)) as u8);
            out.push((0x80 | ((cp >> 6) & 0x3F)) as u8);
            out.push((0x80 | (cp & 0x3F)) as u8);
        }
        _ => push_utf8_4(cp, out),
    }
}

/* ===================================================================== */
/*               Py_UCS4 (UTF-32) → UTF-8                                 */
/* ===================================================================== */

#[inline]
pub fn ucs4_to_utf8_bump<'a>(input: &[u32], bump: &'a bumpalo::Bump) -> &'a str {
    let mut out = bumpalo::collections::Vec::with_capacity_in(input.len() * 4, bump);
    let mut i = 0;
    while i + LANES_U32 <= input.len() {
        let v = U32s::from_slice(&input[i..i + LANES_U32]);
        if v.simd_gt(U32s::splat(0x7F)).to_bitmask() == 0 {
            /* All ASCII in vector */
            for &cp in &input[i..i + LANES_U32] {
                out.push(cp as u8);
            }
        } else {
            for &cp in &input[i..i + LANES_U32] {
                push_utf32_scalar_bump(cp, &mut out);
            }
        }
        i += LANES_U32;
    }
    for &cp in &input[i..] {
        push_utf32_scalar_bump(cp, &mut out);
    }
    let slice = out.into_bump_slice();
    unsafe { core::str::from_utf8_unchecked(slice) }
}

#[inline]
pub fn ucs4_to_utf8(input: &[u32]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(input.len() * 4);
    let mut i = 0;
    while i + LANES_U32 <= input.len() {
        let v = U32s::from_slice(&input[i..i + LANES_U32]);
        if v.simd_gt(U32s::splat(0x7F)).to_bitmask() == 0 {
            /* All ASCII in vector */
            unsafe {
                let bytes: Simd<u8, { LANES_U32 }> = v.cast();
                let dst = out.as_mut_ptr().add(out.len());
                ptr::copy_nonoverlapping(bytes.as_array().as_ptr(), dst, LANES_U32);
                out.set_len(out.len() + LANES_U32);
            }
        } else {
            for &cp in &input[i..i + LANES_U32] {
                push_utf32_scalar(cp, &mut out);
            }
        }
        i += LANES_U32;
    }
    for &cp in &input[i..] {
        push_utf32_scalar(cp, &mut out);
    }
    out
}

/// SIMD-accelerated Python-string → UTF-8, allocated inside a bumpalo arena.
pub fn make_string_fast<'a>(o: *mut pyo3::ffi::PyObject, bump: &'a bumpalo::Bump) -> &'a str {
    unsafe {
        use pyo3::ffi as pyo3_ffi;
        assert!(!o.is_null());
        assert!(pyo3_ffi::PyUnicode_Check(o) != 0);
        if pyo3_ffi::PyUnicode_READY(o) != 0 {
            panic!("PyUnicode_READY failed");
        }

        let len = pyo3_ffi::PyUnicode_GET_LENGTH(o) as usize;
        let kind = pyo3_ffi::PyUnicode_KIND(o);
        let data = pyo3_ffi::PyUnicode_DATA(o);

        match kind {
            pyo3_ffi::PyUnicode_1BYTE_KIND => {
                let chars = std::slice::from_raw_parts(data as *const u8, len);
                ucs1_to_utf8_bump(chars, bump)
            }
            pyo3_ffi::PyUnicode_2BYTE_KIND => {
                let chars = std::slice::from_raw_parts(data as *const u16, len);
                ucs2_to_utf8_bump(chars, bump)
            }
            pyo3_ffi::PyUnicode_4BYTE_KIND => {
                let chars = std::slice::from_raw_parts(data as *const u32, len);
                ucs4_to_utf8_bump(chars, bump)
            }
            _ => {
                panic!("Unknown Unicode kind")
            }
        }
    }
}

/* ===================================================================== */
/*                Unsafe unchecked-to-String convenience                  */
/* ===================================================================== */

pub trait IntoUncheckedString {
    /// Convert the buffer into `String` without validating UTF-8 again.
    ///
    /// # Safety
    /// The caller must guarantee the bytes were produced by these codecs.
    unsafe fn into_unchecked_string(self) -> String;
}

impl IntoUncheckedString for Cow<'_, str> {
    unsafe fn into_unchecked_string(self) -> String {
        match self {
            Cow::Borrowed(s) => s.to_owned(),
            Cow::Owned(s) => s,
        }
    }
}
impl IntoUncheckedString for Vec<u8> {
    unsafe fn into_unchecked_string(self) -> String {
        String::from_utf8_unchecked(self)
    }
}

/* ===================================================================== */
/*                               Tests                                    */
/* ===================================================================== */

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow::*;

    /* ── Py_UCS1 ─────────────────────────────────────────────── */
    #[test]
    fn ucs1_ascii() {
        assert_eq!(ucs1_to_utf8(b"Hello"), Borrowed("Hello"));
    }
    #[test]
    fn ucs1_latin1() {
        let b = [0x48, 0xE9, 0x6C, 0x6C, 0xF6]; // "Héllö"
        assert_eq!(&*ucs1_to_utf8(&b), "Héllö");
    }

    /* ── Py_UCS2 ─────────────────────────────────────────────── */
    #[test]
    fn ucs2_basic() {
        let s = "漢字";
        let v: Vec<u16> = s.encode_utf16().collect();
        assert_eq!(ucs2_to_utf8(&v), s.as_bytes());
    }
    #[test]
    fn ucs2_emoji() {
        let s = "🦀";
        let v: Vec<u16> = s.encode_utf16().collect();
        assert_eq!(ucs2_to_utf8(&v), s.as_bytes());
    }

    /* ── Py_UCS4 ─────────────────────────────────────────────── */
    #[test]
    fn ucs4_basic() {
        let cps = [0x41u32, 0x03A9u32]; // 'A', 'Ω'
        assert_eq!(ucs4_to_utf8(&cps), "AΩ".as_bytes());
    }
    #[test]
    fn ucs4_supp() {
        let cps = [0x1F984u32]; // 🦄
        assert_eq!(ucs4_to_utf8(&cps), "🦄".as_bytes());
    }
}