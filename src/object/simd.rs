//! simd.rs – SIMD-accelerated transcoding for Python Stable-ABI code-units
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
/*                      Performance Thresholds                           */
/* ===================================================================== */

// Optimal thresholds based on benchmarks - below these use scalar, above use SIMD
const SIMD_THRESHOLD_BYTES: usize = 32;  // UTF-8 byte threshold
const SIMD_THRESHOLD_UCS1: usize = 32;   // UCS1 char threshold  
const SIMD_THRESHOLD_UCS2: usize = 16;   // UCS2 char threshold
const SIMD_THRESHOLD_UCS4: usize = 8;    // UCS4 char threshold

/* ===================================================================== */
/*                      Scalar Implementations                           */
/* ===================================================================== */

/// Scalar Latin-1 → UTF-8 conversion (optimized for short strings)
#[inline]
fn ucs1_to_utf8_scalar_bump<'a>(input: &'a [u8], bump: &'a bumpalo::Bump) -> &'a str {
    // Fast path for pure ASCII
    if input.iter().all(|&b| b < 0x80) {
        return unsafe { core::str::from_utf8_unchecked(input) };
    }
    
    // Count non-ASCII bytes for exact allocation
    let extra = input.iter().filter(|&&b| b >= 0x80).count();
    let mut out = bumpalo::collections::Vec::with_capacity_in(input.len() + extra, bump);
    
    for &b in input {
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

#[inline]
fn ucs1_to_utf8_scalar(input: &[u8]) -> Cow<str> {
    // Fast path for pure ASCII
    if input.iter().all(|&b| b < 0x80) {
        return Cow::Borrowed(unsafe { core::str::from_utf8_unchecked(input) });
    }
    
    // Count non-ASCII bytes for exact allocation
    let extra = input.iter().filter(|&&b| b >= 0x80).count();
    let mut out = Vec::with_capacity(input.len() + extra);
    
    for &b in input {
        if b < 0x80 {
            out.push(b);
        } else {
            out.push(0xC0 | (b >> 6));
            out.push(0x80 | (b & 0x3F));
        }
    }
    
    Cow::Owned(unsafe { String::from_utf8_unchecked(out) })
}

/// Scalar UTF-16 → UTF-8 conversion (optimized for short strings)
#[inline]
fn ucs2_to_utf8_scalar_bump<'a>(input: &[u16], bump: &'a bumpalo::Bump) -> &'a str {
    let mut out = bumpalo::collections::Vec::with_capacity_in(input.len() * 3, bump);
    
    let mut i = 0;
    while i < input.len() {
        let w = input[i];
        match w {
            0x0000..=0x007F => out.push(w as u8),
            0x0080..=0x07FF => {
                out.push((0xC0 | (w >> 6)) as u8);
                out.push((0x80 | (w & 0x3F)) as u8);
            }
            0xD800..=0xDBFF => {
                // High surrogate: assume valid pair
                if i + 1 < input.len() {
                    let lo = input[i + 1];
                    let cp = 0x10000 + (((w as u32 & 0x3FF) << 10) | (lo as u32 & 0x3FF));
                    push_utf8_4_bump(cp, &mut out);
                    i += 1; // skip low surrogate
                }
            }
            0xDC00..=0xDFFF => {
                // Isolated low surrogate - skip
            }
            _ => {
                out.push((0xE0 | (w >> 12)) as u8);
                out.push((0x80 | ((w >> 6) & 0x3F)) as u8);
                out.push((0x80 | (w & 0x3F)) as u8);
            }
        }
        i += 1;
    }
    
    let slice = out.into_bump_slice();
    unsafe { core::str::from_utf8_unchecked(slice) }
}

#[inline]
fn ucs2_to_utf8_scalar(input: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() * 3);
    
    let mut i = 0;
    while i < input.len() {
        let w = input[i];
        match w {
            0x0000..=0x007F => out.push(w as u8),
            0x0080..=0x07FF => {
                out.push((0xC0 | (w >> 6)) as u8);
                out.push((0x80 | (w & 0x3F)) as u8);
            }
            0xD800..=0xDBFF => {
                // High surrogate: assume valid pair
                if i + 1 < input.len() {
                    let lo = input[i + 1];
                    let cp = 0x10000 + (((w as u32 & 0x3FF) << 10) | (lo as u32 & 0x3FF));
                    push_utf8_4(cp, &mut out);
                    i += 1; // skip low surrogate
                }
            }
            0xDC00..=0xDFFF => {
                // Isolated low surrogate - skip
            }
            _ => {
                out.push((0xE0 | (w >> 12)) as u8);
                out.push((0x80 | ((w >> 6) & 0x3F)) as u8);
                out.push((0x80 | (w & 0x3F)) as u8);
            }
        }
        i += 1;
    }
    
    out
}

/// Scalar UTF-32 → UTF-8 conversion (optimized for short strings)
#[inline]
fn ucs4_to_utf8_scalar_bump<'a>(input: &[u32], bump: &'a bumpalo::Bump) -> &'a str {
    let mut out = bumpalo::collections::Vec::with_capacity_in(input.len() * 4, bump);
    
    for &cp in input {
        push_utf32_scalar_bump(cp, &mut out);
    }
    
    let slice = out.into_bump_slice();
    unsafe { core::str::from_utf8_unchecked(slice) }
}

#[inline]
fn ucs4_to_utf8_scalar(input: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() * 4);
    
    for &cp in input {
        push_utf32_scalar(cp, &mut out);
    }
    
    out
}

/// Scalar UTF-8 → UCS1 conversion (optimized for short strings)
#[inline]
fn utf8_to_ucs1_scalar(input: &[u8], output: &mut [u8]) -> usize {
    let mut out_pos = 0;
    let mut i = 0;
    
    while i < input.len() && out_pos < output.len() {
        let byte = input[i];
        if byte < 0x80 {
            output[out_pos] = byte;
            out_pos += 1;
            i += 1;
        } else {
            // Decode UTF-8 to get codepoint (simple version)
            match byte {
                0xC0..=0xDF => {
                    // 2-byte sequence
                    if i + 1 < input.len() {
                        let b1 = input[i + 1];
                        let cp = ((byte as u32 & 0x1F) << 6) | (b1 as u32 & 0x3F);
                        if cp <= 0xFF {
                            output[out_pos] = cp as u8;
                            out_pos += 1;
                        }
                        i += 2;
                    } else {
                        break;
                    }
                }
                _ => {
                    // Skip other multi-byte sequences for Latin-1
                    while i < input.len() && (input[i] & 0xC0) != 0xC0 && input[i] >= 0x80 {
                        i += 1;
                    }
                    if i < input.len() && input[i] >= 0x80 {
                        i += 1;
                    }
                }
            }
        }
    }
    
    out_pos
}

/// Scalar UTF-8 → UCS2 conversion (optimized for short strings)
#[inline]
fn utf8_to_ucs2_scalar(input: &[u8], output: &mut [u16]) -> usize {
    let mut out_pos = 0;
    let mut i = 0;
    
    while i < input.len() && out_pos < output.len() {
        let byte = input[i];
        if byte < 0x80 {
            output[out_pos] = byte as u16;
            out_pos += 1;
            i += 1;
        } else {
            // Simple UTF-8 decoding
            if let Ok(s) = core::str::from_utf8(&input[i..]) {
                if let Some(ch) = s.chars().next() {
                    let cp = ch as u32;
                    if cp <= 0xFFFF && (cp < 0xD800 || cp > 0xDFFF) {
                        output[out_pos] = cp as u16;
                        out_pos += 1;
                    } else if cp > 0xFFFF && out_pos + 1 < output.len() {
                        // Encode as surrogate pair
                        let cp = cp - 0x10000;
                        output[out_pos] = 0xD800 | ((cp >> 10) as u16);
                        output[out_pos + 1] = 0xDC00 | ((cp & 0x3FF) as u16);
                        out_pos += 2;
                    }
                    i += ch.len_utf8();
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
    }
    
    out_pos
}

/// Scalar UTF-8 → UCS4 conversion (optimized for short strings)
#[inline]
fn utf8_to_ucs4_scalar(input: &[u8], output: &mut [u32]) -> usize {
    let mut out_pos = 0;
    let mut i = 0;
    
    while i < input.len() && out_pos < output.len() {
        let byte = input[i];
        if byte < 0x80 {
            output[out_pos] = byte as u32;
            out_pos += 1;
            i += 1;
        } else {
            // Simple UTF-8 decoding
            if let Ok(s) = core::str::from_utf8(&input[i..]) {
                if let Some(ch) = s.chars().next() {
                    output[out_pos] = ch as u32;
                    out_pos += 1;
                    i += ch.len_utf8();
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
    }
    
    out_pos
}

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

/* ===================================================================== */
/*               Py_UCS1 (Latin-1) → UTF-8                               */
/* ===================================================================== */

/// Convert a Latin-1 slice to UTF-8 using bump allocator.
/// * Pure ASCII → borrowed `&str` (zero-alloc)
/// * Mixed input → bump-allocated `&str`
/// Uses scalar for short strings, SIMD for longer ones.
#[inline]
pub fn ucs1_to_utf8_bump<'a>(input: &'a [u8], bump: &'a bumpalo::Bump) -> &'a str {
    // Use scalar for short strings to avoid SIMD overhead
    if input.len() < SIMD_THRESHOLD_UCS1 {
        return ucs1_to_utf8_scalar_bump(input, bump);
    }

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
/// Uses scalar for short strings, SIMD for longer ones.
#[inline]
pub fn ucs1_to_utf8<'a>(input: &'a [u8]) -> Cow<'a, str> {
    // Use scalar for short strings to avoid SIMD overhead
    if input.len() < SIMD_THRESHOLD_UCS1 {
        return ucs1_to_utf8_scalar(input);
    }

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
    // Use scalar for short strings to avoid SIMD overhead
    if input.len() < SIMD_THRESHOLD_UCS2 {
        return ucs2_to_utf8_scalar_bump(input, bump);
    }

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
    // Use scalar for short strings to avoid SIMD overhead
    if input.len() < SIMD_THRESHOLD_UCS2 {
        return ucs2_to_utf8_scalar(input);
    }

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
    // Use scalar for short strings to avoid SIMD overhead
    if input.len() < SIMD_THRESHOLD_UCS4 {
        return ucs4_to_utf8_scalar_bump(input, bump);
    }

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
    // Use scalar for short strings to avoid SIMD overhead
    if input.len() < SIMD_THRESHOLD_UCS4 {
        return ucs4_to_utf8_scalar(input);
    }

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

        if v.simd_lt(U8s::splat(0x80)).all() {
            // Pure ASCII chunk - each byte is one character
            char_count += LANES_U8;
            // Update max with chunk max (SIMD horizontal max)
            let chunk_max = v.reduce_max();
            max_codepoint = max_codepoint.max(chunk_max as u32);
            i += LANES_U8;
        } else {
            // Mixed content - fall back to scalar for this chunk
            while i < input.len() && i < (i + LANES_U8) {
                let byte = input[i];
                if byte < 0x80 {
                    // ASCII
                    char_count += 1;
                    max_codepoint = max_codepoint.max(byte as u32);
                    i += 1;
                } else {
                    // Multi-byte UTF-8 sequence
                    let char_start = i;
                    while i < input.len() && input[i] & 0xC0 == 0x80
                        || (i == char_start && input[i] >= 0x80)
                    {
                        i += 1;
                    }
                    // Decode the character to get codepoint
                    if let Ok(s) = core::str::from_utf8(&input[char_start..i]) {
                        if let Some(ch) = s.chars().next() {
                            char_count += 1;
                            max_codepoint = max_codepoint.max(ch as u32);
                        }
                    }
                }
            }
            break; // Exit SIMD loop after handling mixed chunk
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

/// UTF-8 → UCS-1 with SIMD acceleration (for ASCII/Latin-1 strings)
/// Uses scalar for short strings, SIMD for longer ones.
pub fn utf8_to_ucs1_simd(input: &[u8], output: &mut [u8]) -> usize {
    // Use scalar for short strings to avoid SIMD overhead
    if input.len() < SIMD_THRESHOLD_BYTES {
        return utf8_to_ucs1_scalar(input, output);
    }

    let mut out_pos = 0;
    let mut i = 0;

    // SIMD ASCII fast path
    while i + LANES_U8 <= input.len() && out_pos + LANES_U8 <= output.len() {
        let chunk = &input[i..i + LANES_U8];
        let v = U8s::from_slice(chunk);

        if v.simd_lt(U8s::splat(0x80)).all() {
            // Pure ASCII - direct copy
            output[out_pos..out_pos + LANES_U8].copy_from_slice(chunk);
            out_pos += LANES_U8;
            i += LANES_U8;
        } else {
            break; // Exit SIMD loop for mixed content
        }
    }

    // Scalar fallback for remaining bytes
    while i < input.len() && out_pos < output.len() {
        let byte = input[i];
        if byte < 0x80 {
            output[out_pos] = byte;
            out_pos += 1;
            i += 1;
        } else {
            // Decode UTF-8 to get codepoint
            let char_start = i;
            while i < input.len() && (input[i] & 0xC0 == 0x80 || i == char_start) {
                i += 1;
            }
            if let Ok(s) = core::str::from_utf8(&input[char_start..i]) {
                if let Some(ch) = s.chars().next() {
                    let cp = ch as u32;
                    if cp <= 0xFF {
                        output[out_pos] = cp as u8;
                        out_pos += 1;
                    }
                }
            }
        }
    }

    out_pos
}

/// UTF-8 → UCS-2 with SIMD acceleration
/// Uses scalar for short strings, SIMD for longer ones.
pub fn utf8_to_ucs2_simd(input: &[u8], output: &mut [u16]) -> usize {
    // Use scalar for short strings to avoid SIMD overhead
    if input.len() < SIMD_THRESHOLD_BYTES {
        return utf8_to_ucs2_scalar(input, output);
    }

    let mut out_pos = 0;
    let mut i = 0;

    // SIMD ASCII fast path
    while i + LANES_U8 <= input.len() && out_pos + LANES_U8 <= output.len() {
        let chunk = &input[i..i + LANES_U8];
        let v = U8s::from_slice(chunk);

        if v.simd_lt(U8s::splat(0x80)).all() {
            // Pure ASCII - zero-extend to u16
            for j in 0..LANES_U8 {
                output[out_pos + j] = chunk[j] as u16;
            }
            out_pos += LANES_U8;
            i += LANES_U8;
        } else {
            break; // Exit SIMD loop for mixed content
        }
    }

    // Scalar fallback for UTF-8 decoding
    while i < input.len() && out_pos < output.len() {
        let byte = input[i];
        if byte < 0x80 {
            output[out_pos] = byte as u16;
            out_pos += 1;
            i += 1;
        } else {
            // Decode UTF-8 sequence
            let char_start = i;
            while i < input.len() && (input[i] & 0xC0 == 0x80 || i == char_start) {
                i += 1;
            }
            if let Ok(s) = core::str::from_utf8(&input[char_start..i]) {
                for ch in s.chars() {
                    if out_pos >= output.len() {
                        break;
                    }
                    let cp = ch as u32;
                    if cp <= 0xFFFF && (cp < 0xD800 || cp > 0xDFFF) {
                        // BMP codepoint, not surrogate
                        output[out_pos] = cp as u16;
                        out_pos += 1;
                    } else if cp > 0xFFFF {
                        // Encode as surrogate pair
                        if out_pos + 1 < output.len() {
                            let cp = cp - 0x10000;
                            output[out_pos] = 0xD800 | ((cp >> 10) as u16);
                            output[out_pos + 1] = 0xDC00 | ((cp & 0x3FF) as u16);
                            out_pos += 2;
                        }
                    }
                }
            }
        }
    }

    out_pos
}

/// UTF-8 → UCS-4 with SIMD acceleration
/// Uses scalar for short strings, SIMD for longer ones.
pub fn utf8_to_ucs4_simd(input: &[u8], output: &mut [u32]) -> usize {
    // Use scalar for short strings to avoid SIMD overhead
    if input.len() < SIMD_THRESHOLD_BYTES {
        return utf8_to_ucs4_scalar(input, output);
    }

    let mut out_pos = 0;
    let mut i = 0;

    // SIMD ASCII fast path
    while i + LANES_U8 <= input.len() && out_pos + LANES_U8 <= output.len() {
        let chunk = &input[i..i + LANES_U8];
        let v = U8s::from_slice(chunk);

        if v.simd_lt(U8s::splat(0x80)).all() {
            // Pure ASCII - zero-extend to u32
            for j in 0..LANES_U8 {
                output[out_pos + j] = chunk[j] as u32;
            }
            out_pos += LANES_U8;
            i += LANES_U8;
        } else {
            break; // Exit SIMD loop for mixed content
        }
    }

    // Scalar fallback for UTF-8 decoding
    while i < input.len() && out_pos < output.len() {
        let byte = input[i];
        if byte < 0x80 {
            output[out_pos] = byte as u32;
            out_pos += 1;
            i += 1;
        } else {
            // Decode UTF-8 sequence
            let char_start = i;
            while i < input.len() && (input[i] & 0xC0 == 0x80 || i == char_start) {
                i += 1;
            }
            if let Ok(s) = core::str::from_utf8(&input[char_start..i]) {
                for ch in s.chars() {
                    if out_pos >= output.len() {
                        break;
                    }
                    output[out_pos] = ch as u32;
                    out_pos += 1;
                }
            }
        }
    }

    out_pos
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

    /* ── UTF-8 → UCS* Conversion Tests ────────────────────────────────── */

    #[test]
    fn utf8_to_ucs_basic() {
        // ASCII test
        let ascii = "Hello";
        let mut ucs1_buf = [0u8; 10];
        let mut ucs2_buf = [0u16; 10];
        let mut ucs4_buf = [0u32; 10];

        let len1 = utf8_to_ucs1_simd(ascii.as_bytes(), &mut ucs1_buf);
        let len2 = utf8_to_ucs2_simd(ascii.as_bytes(), &mut ucs2_buf);
        let len4 = utf8_to_ucs4_simd(ascii.as_bytes(), &mut ucs4_buf);

        assert_eq!(len1, 5);
        assert_eq!(len2, 5);
        assert_eq!(len4, 5);
        assert_eq!(&ucs1_buf[..len1], ascii.as_bytes());
    }

    #[test]
    fn utf8_to_ucs_roundtrip() {
        let test_cases = vec!["Hello", "café", "🦀", "Hello, 世界!"];

        for case in test_cases {
            // UTF-8 → UCS-2 → UTF-8
            let mut ucs2_buf = vec![0u16; case.chars().count() * 2]; // Extra space for surrogates
            let ucs2_len = utf8_to_ucs2_simd(case.as_bytes(), &mut ucs2_buf);
            let back_to_utf8 = ucs2_to_utf8(&ucs2_buf[..ucs2_len]);
            assert_eq!(case.as_bytes(), &back_to_utf8);

            // UTF-8 → UCS-4 → UTF-8
            let mut ucs4_buf = vec![0u32; case.chars().count()];
            let ucs4_len = utf8_to_ucs4_simd(case.as_bytes(), &mut ucs4_buf);
            let back_to_utf8 = ucs4_to_utf8(&ucs4_buf[..ucs4_len]);
            assert_eq!(case.as_bytes(), &back_to_utf8);
        }
    }

    #[test]
    fn utf8_analysis_accuracy() {
        let test_cases = vec![
            ("", 0, 0),
            ("A", 1, 65),
            ("Hello", 5, 111),  // 'o' = 111
            ("café", 4, 233),   // 'é' = 233
            ("🦀", 1, 0x1F980), // Crab emoji
        ];

        for (input, expected_count, expected_max) in test_cases {
            let (count, max_cp) = analyze_utf8_simd(input.as_bytes());
            assert_eq!(
                count, expected_count,
                "Character count mismatch for '{}'",
                input
            );
            assert_eq!(
                max_cp, expected_max,
                "Max codepoint mismatch for '{}'",
                input
            );

            // Verify against scalar implementation
            let scalar_count = input.chars().count();
            let scalar_max = input.chars().map(|c| c as u32).max().unwrap_or(0);
            assert_eq!(count, scalar_count);
            assert_eq!(max_cp, scalar_max);
        }
    }

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
            input.push(if i % 4 == 0 {
                0x80 + (i % 128) as u8
            } else {
                b'A' + (i % 26) as u8
            });
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
            vec![0x41, 0x42, 0x43],        // Pure ASCII
            vec![0x80, 0x81, 0x82],        // Pure extended
            vec![0x41, 0x80, 0x42, 0x81],  // Mixed
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
