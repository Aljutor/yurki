//! SIMD-accelerated transcoding for Python's UCS-1/2/4 string representations.
//!
//! This module provides branch-free codecs for converting between Python's internal
//! fixed-width string formats (UCS-1, UCS-2, UCS-4) and UTF-8. It uses the
//! portable SIMD API (`core::simd`) to compile for AVX2/AVX-512 on x86-64,
//! NEON on Apple M-series, and WASM-SIMD.

#![allow(dead_code)]

use core::simd::cmp::{SimdPartialEq, SimdPartialOrd};
use core::simd::prelude::SimdUint;
use core::simd::{LaneCount, Simd, SupportedLaneCount};

pub mod ucs1;
pub mod ucs2;
pub mod ucs4;

pub use ucs1::{ucs1_to_utf8, ucs1_to_utf8_bump, utf8_to_ucs1_simd};
pub use ucs2::{ucs2_to_utf8, ucs2_to_utf8_bump, utf8_to_ucs2_simd};
pub use ucs4::{ucs4_to_utf8, ucs4_to_utf8_bump, utf8_to_ucs4_simd};

// ========================================================================== //
//                        SIMD Lane-Width Selection                         //
// ========================================================================== //

// ----------------------------------- u8 ----------------------------------- //
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512vbmi2",
    target_feature = "avx512bw"
))]
pub(crate) type U8s = Simd<u8, 64>;
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512vbmi2",
    target_feature = "avx512bw"
))]
pub(crate) const LANES_U8: usize = 64;

#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx512vbmi2", target_feature = "avx512bw")),
    target_feature = "avx2"
))]
pub(crate) type U8s = Simd<u8, 32>;
#[cfg(all(
    target_arch = "x86_64",
    not(all(target_feature = "avx512vbmi2", target_feature = "avx512bw")),
    target_feature = "avx2"
))]
pub(crate) const LANES_U8: usize = 32;

#[cfg(all(target_arch = "aarch64", target_feature = "sve2"))]
pub(crate) type U8s = Simd<u8, 32>;
#[cfg(all(target_arch = "aarch64", target_feature = "sve2"))]
pub(crate) const LANES_U8: usize = 32;

// Default: 128-bit vectors (NEON, WASM-SIMD, or portable fallback).
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi2",
        target_feature = "avx512bw"
    ),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "sve2")
)))]
pub(crate) type U8s = Simd<u8, 16>;
#[cfg(not(any(
    all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi2",
        target_feature = "avx512bw"
    ),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "sve2")
)))]
pub(crate) const LANES_U8: usize = 16;

// ----------------------------------- u16 ---------------------------------- //
#[cfg(target_feature = "avx512bw")]
pub(crate) type U16s = Simd<u16, 32>;
#[cfg(target_feature = "avx512bw")]
pub(crate) const LANES_U16: usize = 32;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
pub(crate) type U16s = Simd<u16, 16>;
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
pub(crate) const LANES_U16: usize = 16;

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512bw")))]
pub(crate) type U16s = Simd<u16, 8>;
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512bw")))]
pub(crate) const LANES_U16: usize = 8;

// ----------------------------------- u32 ---------------------------------- //
#[cfg(target_feature = "avx512bw")]
pub(crate) type U32s = Simd<u32, 16>;
#[cfg(target_feature = "avx512bw")]
pub(crate) const LANES_U32: usize = 16;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
pub(crate) type U32s = Simd<u32, 8>;
#[cfg(all(target_feature = "avx2", not(target_feature = "avx512bw")))]
pub(crate) const LANES_U32: usize = 8;

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512bw")))]
pub(crate) type U32s = Simd<u32, 4>;
#[cfg(not(any(target_feature = "avx2", target_feature = "avx512bw")))]
pub(crate) const LANES_U32: usize = 4;

// ========================================================================== //
//                         Performance Thresholds                             //
// ========================================================================== //

/// Minimum input size (in bytes) to prefer SIMD for UTF-8 analysis and decoding.
pub(crate) const SIMD_THRESHOLD_BYTES: usize = 64;
/// Minimum input size (in code units) to prefer SIMD for UCS-1 -> UTF-8.
pub(crate) const SIMD_THRESHOLD_UCS1: usize = 96;
/// Minimum input size (in code units) to prefer SIMD for UCS-2 -> UTF-8.
pub(crate) const SIMD_THRESHOLD_UCS2: usize = 48;
/// Minimum input size (in code units) to prefer SIMD for UCS-4 -> UTF-8.
pub(crate) const SIMD_THRESHOLD_UCS4: usize = 32;

// ========================================================================== //
//                           SIMD Helper Functions                            //
// ========================================================================== //

/// Splits a 16-byte vector into two 8-byte vectors.
#[inline(always)]
pub(crate) fn split_u8x16(v: Simd<u8, 16>) -> (Simd<u8, 8>, Simd<u8, 8>) {
    let array = v.to_array();
    let low = Simd::from_array([
        array[0], array[1], array[2], array[3], array[4], array[5], array[6], array[7],
    ]);
    let high = Simd::from_array([
        array[8], array[9], array[10], array[11], array[12], array[13], array[14], array[15],
    ]);
    (low, high)
}

#[inline(always)]
pub(crate) fn split_u8x32(v: Simd<u8, 32>) -> (Simd<u8, 16>, Simd<u8, 16>) {
    let array = v.to_array();
    let low = Simd::from_slice(&array[0..16]);
    let high = Simd::from_slice(&array[16..32]);
    (low, high)
}

/// Extracts the bytes from a SIMD vector into an array.
#[inline(always)]
pub(crate) fn simd_to_bytes<const N: usize>(v: Simd<u8, N>) -> [u8; N]
where
    LaneCount<N>: SupportedLaneCount,
{
    v.to_array()
}

/// Extracts the low byte of each `u16` lane, assuming ASCII content.
#[inline(always)]
pub(crate) fn simd_u16_to_ascii_bytes<const N: usize>(v: Simd<u16, N>) -> [u8; N]
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut result = [0u8; N];
    let array = v.to_array();
    for i in 0..N {
        result[i] = array[i] as u8; // Extract low byte only
    }
    result
}

/// Extracts the low byte of each `u32` lane, assuming ASCII content.
#[inline(always)]
pub(crate) fn simd_u32_to_ascii_bytes<const N: usize>(v: Simd<u32, N>) -> [u8; N]
where
    LaneCount<N>: SupportedLaneCount,
{
    let mut result = [0u8; N];
    let array = v.to_array();
    for i in 0..N {
        result[i] = array[i] as u8; // Extract low byte only
    }
    result
}

// ========================================================================== //
//                          Shared Helper Routines                            //
// ========================================================================== //

#[inline(always)]
pub(crate) fn push_utf8_4_bump(cp: u32, out: &mut bumpalo::collections::Vec<u8>) {
    out.extend_from_slice(&[
        (0xF0 | (cp >> 18)) as u8,
        (0x80 | ((cp >> 12) & 0x3F)) as u8,
        (0x80 | ((cp >> 6) & 0x3F)) as u8,
        (0x80 | (cp & 0x3F)) as u8,
    ]);
}

#[inline(always)]
pub(crate) fn push_utf8_4(cp: u32, out: &mut Vec<u8>) {
    out.extend_from_slice(&[
        (0xF0 | (cp >> 18)) as u8,
        (0x80 | ((cp >> 12) & 0x3F)) as u8,
        (0x80 | ((cp >> 6) & 0x3F)) as u8,
        (0x80 | (cp & 0x3F)) as u8,
    ]);
}

// ========================================================================== //
//                              UTF-8 Analysis                                //
// ========================================================================== //

/// Scalar routine to count characters and find the maximum codepoint.
#[inline]
fn analyze_utf8_scalar(input: &[u8]) -> (usize, u32) {
    let mut char_count = 0;
    let mut max_codepoint = 0u32;
    let mut i = 0;

    while i < input.len() {
        let byte = input[i];
        if byte < 0x80 {
            char_count += 1;
            max_codepoint = max_codepoint.max(byte as u32);
            i += 1;
        } else {
            // Decode UTF-8 character
            if let Ok(s) = core::str::from_utf8(&input[i..]) {
                if let Some(ch) = s.chars().next() {
                    char_count += 1;
                    max_codepoint = max_codepoint.max(ch as u32);
                    i += ch.len_utf8();
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
    }

    (char_count, max_codepoint)
}

/// Counts UTF-8 characters and finds the maximum codepoint using SIMD.
///
/// For short inputs, this function delegates to a scalar routine to avoid
/// SIMD overhead. For longer inputs, it processes the data in chunks,
/// using a fast path for pure ASCII blocks.
pub fn analyze_utf8_simd(input: &[u8]) -> (usize, u32) {
    if input.len() < SIMD_THRESHOLD_BYTES {
        return analyze_utf8_scalar(input);
    }

    let mut char_count = 0usize;
    let mut max_codepoint = 0u32;
    let mut i = 0;

    // SIMD loop
    while i + LANES_U8 <= input.len() {
        let chunk = U8s::from_slice(&input[i..i + LANES_U8]);

        // Fast path for pure ASCII blocks.
        let ascii_mask = chunk.simd_lt(U8s::splat(0x80));
        if ascii_mask.all() {
            char_count += LANES_U8;
            max_codepoint = max_codepoint.max(chunk.reduce_max() as u32);
            i += LANES_U8;
            continue;
        }

        // In mixed-content blocks, count characters by identifying all non-continuation
        // bytes. A UTF-8 continuation byte has the form `10xxxxxx`, so any byte
        // where `(byte & 0xC0) != 0x80` marks the start of a new character.
        let continuation_mask = chunk & U8s::splat(0xC0);
        let is_start_byte = continuation_mask.simd_ne(U8s::splat(0x80));
        char_count += is_start_byte.to_bitmask().count_ones() as usize;

        // To find the max codepoint, we take the max ASCII value from the chunk
        // and then perform a scalar decode only on the multi-byte sequences.
        let max_ascii_in_chunk = chunk.reduce_max();
        max_codepoint = max_codepoint.max(max_ascii_in_chunk as u32);

        let bitmask = is_start_byte.to_bitmask();
        for k in 0..LANES_U8 {
            if (bitmask >> k) & 1 != 0 {
                let byte = input[i + k];
                if byte >= 0xC0 {
                    // Start of a multi-byte sequence.
                    let char_start = i + k;
                    let mut char_len = 1;
                    while char_start + char_len < input.len()
                        && (input[char_start + char_len] & 0xC0) == 0x80
                    {
                        char_len += 1;
                    }
                    if let Ok(s) = core::str::from_utf8(&input[char_start..char_start + char_len]) {
                        if let Some(ch) = s.chars().next() {
                            max_codepoint = max_codepoint.max(ch as u32);
                        }
                    }
                }
            }
        }
        i += LANES_U8;
    }

    // Handle the remainder with the scalar routine.
    let (tail_char_count, tail_max_codepoint) = analyze_utf8_scalar(&input[i..]);
    char_count += tail_char_count;
    max_codepoint = max_codepoint.max(tail_max_codepoint);

    (char_count, max_codepoint)
}

/// Converts a Python string object to a UTF-8 string slice in a `bumpalo` arena.
///
/// This function inspects the internal representation of a `PyObject` and dispatches
/// to the appropriate UCS-1, UCS-2, or UCS-4 to UTF-8 conversion routine.
///
/// # Safety
///
/// The caller must ensure the `PyObject` pointer is valid, non-null, and points
/// to a Python unicode object. The GIL must also be held.
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
