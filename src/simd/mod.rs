//! SIMD-accelerated transcoding for Python Stable-ABI code-units
//!
//! **Pure Rust, zero-FFI:** Wide-SIMD, branch-free codecs inspired by
//! Daniel Lemire et al.'s **simdutf**, built entirely on the evolving
//! *portable-SIMD* API so a single crate compiles on x86-64 (AVX2/AVX-512),
//! Apple M-series (NEON) and WASM-SIMD.

#![allow(dead_code)]

use core::simd::Simd;
use core::simd::cmp::SimdPartialOrd;
use core::simd::prelude::SimdUint;

pub mod ucs1;
pub mod ucs2;
pub mod ucs4;

pub use ucs1::{ucs1_to_utf8, ucs1_to_utf8_bump, utf8_to_ucs1_simd};
pub use ucs2::{ucs2_to_utf8, ucs2_to_utf8_bump, utf8_to_ucs2_simd};
pub use ucs4::{ucs4_to_utf8, ucs4_to_utf8_bump, utf8_to_ucs4_simd};

/* ==================================================================== */
/*                      SIMD lane-width selection                       */
/* ==================================================================== */

/* ── u8 ───────────────────────────────────────────────────────────────*/
#[cfg(all(target_arch = "x86_64", target_feature = "avx512vbmi2", target_feature = "avx512bw"))]
pub(crate) type U8s = Simd<u8, 64>;
#[cfg(all(target_arch = "x86_64", target_feature = "avx512vbmi2", target_feature = "avx512bw"))]
pub(crate) const LANES_U8: usize = 64;

#[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx512vbmi2", target_feature = "avx512bw")), target_feature = "avx2"))]
pub(crate) type U8s = Simd<u8, 32>;
#[cfg(all(target_arch = "x86_64", not(all(target_feature = "avx512vbmi2", target_feature = "avx512bw")), target_feature = "avx2"))]
pub(crate) const LANES_U8: usize = 32;

#[cfg(all(target_arch = "aarch64", target_feature = "sve2"))]
pub(crate) type U8s = Simd<u8, 32>; // current Apple M4 (256‑bit SVE2)
#[cfg(all(target_arch = "aarch64", target_feature = "sve2"))]
pub(crate) const LANES_U8: usize = 32;

// default: NEON 128‑bit or WASM 128‑bit or plain portable SIMD 128‑bit
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx512vbmi2", target_feature = "avx512bw"),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "sve2"))
))]
pub(crate) type U8s = Simd<u8, 16>;
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx512vbmi2", target_feature = "avx512bw"),
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "sve2"))
))]
pub(crate) const LANES_U8: usize = 16;

/* ── u16 ──────────────────────────────────────────────────────────────*/
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

/* ── u32 ──────────────────────────────────────────────────────────────*/
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

/* ===================================================================== */
/*                      Performance Thresholds                           */
/* ===================================================================== */

// Optimal thresholds based on benchmarks - below these use scalar, above use SIMD
pub(crate) const SIMD_THRESHOLD_BYTES: usize = 64; // UTF‑8 analyser & decode
pub(crate) const SIMD_THRESHOLD_UCS1:  usize = 96; // Latin‑1 → UTF‑8
pub(crate) const SIMD_THRESHOLD_UCS2:  usize = 48; // UTF‑16 → UTF‑8
pub(crate) const SIMD_THRESHOLD_UCS4:  usize = 32; // UTF‑32 → UTF‑8

/* ===================================================================== */
/*                       Shared Helper routines                          */
/* ===================================================================== */

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

/* ===================================================================== */
/*                       UTF-8 Analysis                                  */
/* ===================================================================== */

/// Scalar UTF-8 analysis (optimized for short strings)
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

/// SIMD UTF-8 analysis: count characters + find max codepoint in one pass
/// Uses scalar for short strings, SIMD for longer ones.
pub fn analyze_utf8_simd(input: &[u8]) -> (usize, u32) {
    // Use scalar for short strings to avoid SIMD overhead
    if input.len() < SIMD_THRESHOLD_BYTES {
        return analyze_utf8_scalar(input);
    }

    let mut char_count = 0usize;
    let mut max_codepoint = 0u32;
    let mut i = 0;

    // SIMD ASCII detection and counting
    while i + LANES_U8 <= input.len() {
        let chunk = &input[i..i + LANES_U8];
        let v = U8s::from_slice(chunk);
        let ascii_mask = v.simd_lt(U8s::splat(0x80));

        if ascii_mask.all() {
            // Pure ASCII chunk
            char_count += LANES_U8;
            max_codepoint = max_codepoint.max(v.reduce_max() as u32);
            i += LANES_U8;
        } else {
            // Mixed content: handle byte by byte, but don't exit the SIMD loop
            let mut j = 0;
            while j < LANES_U8 {
                let byte = chunk[j];
                if byte < 0x80 {
                    char_count += 1;
                    max_codepoint = max_codepoint.max(byte as u32);
                    j += 1;
                } else {
                    // Decode one multi-byte character
                    let char_start = i + j;
                    let mut char_len = 1;
                    // Find the end of the character
                    while char_start + char_len < input.len() && (input[char_start + char_len] & 0xC0) == 0x80 {
                        char_len += 1;
                    }
                    
                    if let Ok(s) = core::str::from_utf8(&input[char_start..char_start + char_len]) {
                        if let Some(ch) = s.chars().next() {
                            char_count += 1;
                            max_codepoint = max_codepoint.max(ch as u32);
                        }
                    }
                    j += char_len;
                }
            }
            i += LANES_U8;
        }
    }

    // Handle remaining bytes with scalar processing
    while i < input.len() {
        let byte = input[i];
        if byte < 0x80 {
            char_count += 1;
            max_codepoint = max_codepoint.max(byte as u32);
            i += 1;
        } else {
            // Multi-byte UTF-8 - decode properly
            let char_start = i;
            while i < input.len() && (input[i] & 0xC0 == 0x80 || i == char_start) {
                i += 1;
            }
            if let Ok(s) = core::str::from_utf8(&input[char_start..i]) {
                if let Some(ch) = s.chars().next() {
                    char_count += 1;
                    max_codepoint = max_codepoint.max(ch as u32);
                }
            }
        }
    }

    (char_count, max_codepoint)
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
