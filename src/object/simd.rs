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

/* ==================================================================== */
/*                      SIMD lane-width selection                       */
/* ==================================================================== */

/* ── u8 ───────────────────────────────────────────────────────────────*/
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

/* ── u16 ──────────────────────────────────────────────────────────────*/
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

/* ── u32 ──────────────────────────────────────────────────────────────*/
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
/*               Py_UCS1 (Latin-1) → UTF-8                               */
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
/*               Py_UCS2 (UTF-16) → UTF-8                                */
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
/*                       Helper routines                                 */
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
/*               Py_UCS4 (UTF-32) → UTF-8                                */
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
pub fn convert_pystring<'a>(o: *mut pyo3::ffi::PyObject, bump: &'a bumpalo::Bump) -> &'a str {
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
/*                               Tests                                   */
/* ===================================================================== */

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow::*;

    /* ── UCS1 (Latin-1) Tests ─────────────────────────────────────────── */
    
    #[test]
    fn ucs1_empty() {
        assert_eq!(ucs1_to_utf8(b""), Borrowed(""));
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs1_to_utf8_bump(b"", &bump), "");
    }

    #[test]
    fn ucs1_ascii() {
        assert_eq!(ucs1_to_utf8(b"Hello"), Borrowed("Hello"));
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs1_to_utf8_bump(b"Hello, World!", &bump), "Hello, World!");
    }

    #[test]
    fn ucs1_single_char() {
        assert_eq!(ucs1_to_utf8(b"A"), Borrowed("A"));
        assert_eq!(&*ucs1_to_utf8(&[0xFF]), "ÿ");
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs1_to_utf8_bump(b"Z", &bump), "Z");
        assert_eq!(ucs1_to_utf8_bump(&[0xA9], &bump), "©"); // Copyright symbol
    }

    #[test]
    fn ucs1_latin1() {
        let b = [0x48, 0xE9, 0x6C, 0x6C, 0xF6]; // "Héllö"
        assert_eq!(&*ucs1_to_utf8(&b), "Héllö");
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs1_to_utf8_bump(&b, &bump), "Héllö");
    }

    #[test]
    fn ucs1_mixed_content() {
        let mixed = b"Hello \xE9\xE8\xEA world \xFF!";
        let result = ucs1_to_utf8(mixed);
        assert!(matches!(result, Owned(_)));
        assert_eq!(&*result, "Hello éèê world ÿ!");
    }

    #[test]
    fn ucs1_large_ascii() {
        let large_ascii = "A".repeat(1000);
        let bytes = large_ascii.as_bytes();
        assert_eq!(ucs1_to_utf8(bytes), Borrowed(&large_ascii));
    }

    #[test]
    fn ucs1_large_mixed() {
        let mut input = Vec::new();
        for i in 0..1000 {
            input.push(if i % 4 == 0 { 0x80 + (i % 128) as u8 } else { b'A' + (i % 26) as u8 });
        }
        let result = ucs1_to_utf8(&input);
        assert!(matches!(result, Owned(_)));
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
    }

    #[test]
    fn ucs1_all_latin1_chars() {
        let all_latin1: Vec<u8> = (128..=255).collect();
        let result = ucs1_to_utf8(&all_latin1);
        assert!(matches!(result, Owned(_)));
        assert!(std::str::from_utf8(result.as_bytes()).is_ok());
        
        let bump = bumpalo::Bump::new();
        let bump_result = ucs1_to_utf8_bump(&all_latin1, &bump);
        assert_eq!(result.as_bytes(), bump_result.as_bytes());
    }

    /* ── UCS2 (UTF-16) Tests ──────────────────────────────────────────── */
    
    #[test]
    fn ucs2_empty() {
        assert_eq!(ucs2_to_utf8(&[]), Vec::<u8>::new());
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs2_to_utf8_bump(&[], &bump), "");
    }

    #[test]
    fn ucs2_ascii() {
        let ascii: Vec<u16> = "Hello".chars().map(|c| c as u16).collect();
        assert_eq!(ucs2_to_utf8(&ascii), "Hello".as_bytes());
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs2_to_utf8_bump(&ascii, &bump), "Hello");
    }

    #[test]
    fn ucs2_basic() {
        let s = "漢字";
        let v: Vec<u16> = s.encode_utf16().collect();
        assert_eq!(ucs2_to_utf8(&v), s.as_bytes());
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs2_to_utf8_bump(&v, &bump), s);
    }

    #[test]
    fn ucs2_emoji() {
        let s = "🦀";
        let v: Vec<u16> = s.encode_utf16().collect();
        assert_eq!(ucs2_to_utf8(&v), s.as_bytes());
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs2_to_utf8_bump(&v, &bump), s);
    }

    #[test]
    fn ucs2_surrogate_pairs() {
        let emoji_family = "👨‍👩‍👧‍👦";
        let utf16: Vec<u16> = emoji_family.encode_utf16().collect();
        assert_eq!(ucs2_to_utf8(&utf16), emoji_family.as_bytes());
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs2_to_utf8_bump(&utf16, &bump), emoji_family);
    }

    #[test]
    fn ucs2_mixed_bmp_supplementary() {
        let mixed = "A漢🦀Ω";
        let utf16: Vec<u16> = mixed.encode_utf16().collect();
        assert_eq!(ucs2_to_utf8(&utf16), mixed.as_bytes());
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs2_to_utf8_bump(&utf16, &bump), mixed);
    }

    #[test]
    fn ucs2_large_ascii() {
        let large_ascii = "Z".repeat(1000);
        let utf16: Vec<u16> = large_ascii.encode_utf16().collect();
        assert_eq!(ucs2_to_utf8(&utf16), large_ascii.as_bytes());
    }

    #[test]
    fn ucs2_three_byte_utf8() {
        let korean = "안녕하세요";
        let utf16: Vec<u16> = korean.encode_utf16().collect();
        assert_eq!(ucs2_to_utf8(&utf16), korean.as_bytes());
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs2_to_utf8_bump(&utf16, &bump), korean);
    }

    /* ── UCS4 (UTF-32) Tests ──────────────────────────────────────────── */
    
    #[test]
    fn ucs4_empty() {
        assert_eq!(ucs4_to_utf8(&[]), Vec::<u8>::new());
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs4_to_utf8_bump(&[], &bump), "");
    }

    #[test]
    fn ucs4_ascii() {
        let ascii: Vec<u32> = "Hello".chars().map(|c| c as u32).collect();
        assert_eq!(ucs4_to_utf8(&ascii), "Hello".as_bytes());
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs4_to_utf8_bump(&ascii, &bump), "Hello");
    }

    #[test]
    fn ucs4_basic() {
        let cps = [0x41u32, 0x03A9u32]; // 'A', 'Ω'
        assert_eq!(ucs4_to_utf8(&cps), "AΩ".as_bytes());
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs4_to_utf8_bump(&cps, &bump), "AΩ");
    }

    #[test]
    fn ucs4_supp() {
        let cps = [0x1F984u32]; // 🦄
        assert_eq!(ucs4_to_utf8(&cps), "🦄".as_bytes());
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs4_to_utf8_bump(&cps, &bump), "🦄");
    }

    #[test]
    fn ucs4_full_range() {
        let codepoints = vec![
            0x00000041u32, // 'A' (1 byte)
            0x000000E9u32, // 'é' (2 bytes)
            0x00004E2Du32, // '中' (3 bytes)
            0x0001F984u32, // '🦄' (4 bytes)
        ];
        let expected = "Aé中🦄";
        assert_eq!(ucs4_to_utf8(&codepoints), expected.as_bytes());
        
        let bump = bumpalo::Bump::new();
        assert_eq!(ucs4_to_utf8_bump(&codepoints, &bump), expected);
    }

    #[test]
    fn ucs4_large_ascii() {
        let large_ascii: Vec<u32> = "X".repeat(1000).chars().map(|c| c as u32).collect();
        let expected = "X".repeat(1000);
        assert_eq!(ucs4_to_utf8(&large_ascii), expected.as_bytes());
    }

    #[test]
    fn ucs4_boundary_codepoints() {
        let boundary_points = vec![
            0x0000007Fu32, // Last 1-byte
            0x00000080u32, // First 2-byte
            0x000007FFu32, // Last 2-byte
            0x00000800u32, // First 3-byte
            0x0000FFFFu32, // Last 3-byte
            0x00010000u32, // First 4-byte
            0x0010FFFFu32, // Last valid Unicode
        ];
        let result = ucs4_to_utf8(&boundary_points);
        assert!(std::str::from_utf8(&result).is_ok());
        
        let bump = bumpalo::Bump::new();
        let bump_result = ucs4_to_utf8_bump(&boundary_points, &bump);
        assert_eq!(result, bump_result.as_bytes());
    }

    /* ── Round-trip and Property Tests ────────────────────────────────── */

    #[test]
    fn roundtrip_ucs1_utf8() {
        for i in 0..=255u8 {
            let input = [i];
            let utf8_result = ucs1_to_utf8(&input);
            let back_to_utf16: Vec<u16> = utf8_result.chars().map(|c| c as u16).collect();
            let utf8_from_utf16 = ucs2_to_utf8(&back_to_utf16);
            assert_eq!(utf8_result.as_bytes(), &utf8_from_utf16);
        }
    }

    #[test]
    fn simd_vs_scalar_consistency() {
        // Test that SIMD and scalar paths produce identical results
        let test_cases = vec![
            vec![0x41, 0x42, 0x43], // Pure ASCII
            vec![0x80, 0x81, 0x82], // Pure extended
            vec![0x41, 0x80, 0x42, 0x81], // Mixed
            (0..255).collect::<Vec<u8>>(), // Full range
        ];
        
        for case in test_cases {
            let result1 = ucs1_to_utf8(&case);
            
            let bump = bumpalo::Bump::new();
            let result2 = ucs1_to_utf8_bump(&case, &bump);
            
            assert_eq!(result1.as_bytes(), result2.as_bytes());
        }
    }

    #[test]
    fn output_length_bounds() {
        // UCS1: output <= input.len() * 2
        let latin1_input: Vec<u8> = (128..=255).collect();
        let utf8_output = ucs1_to_utf8(&latin1_input);
        assert!(utf8_output.len() <= latin1_input.len() * 2);
        
        // UCS2: output <= input.len() * 3 for BMP, * 4 for surrogates
        let bmp_input: Vec<u16> = vec![0x4E2D, 0x6587]; // 中文
        let utf8_output = ucs2_to_utf8(&bmp_input);
        assert!(utf8_output.len() <= bmp_input.len() * 3);
        
        // UCS4: output <= input.len() * 4
        let unicode_input: Vec<u32> = vec![0x1F984, 0x1F680]; // 🦄🚀
        let utf8_output = ucs4_to_utf8(&unicode_input);
        assert!(utf8_output.len() <= unicode_input.len() * 4);
    }
}
