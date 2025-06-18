//! py_unicode_simd.rs – SIMD‑accelerated transcoding for Python Stable‑ABI code‑units
//!
//! **Pure Rust, zero‑FFI:** Implements the high‑performance strategy popularised by
//! Daniel Lemire et al.’s **simdutf** project <https://github.com/simdutf/simdutf>.
//! We mirror simdutf’s design – an ASCII vectorised fast‑path followed by a minimal
//! scalar fallback – but re‑create the kernels with the portable‑SIMD API so **no C++
//! tool‑chain is required** and a single crate builds on x86‑64 (SSE/AVX), Apple M‑series
//! (NEON), WASM‑SIMD and more.
//!
//! Supported Python Stable‑ABI inputs:
//! • **Py_UCS1** – 8‑bit Latin‑1 code units (0x00–0xFF)
//! • **Py_UCS2** – 16‑bit UTF‑16 code units (with surrogate pairs)
//! • **Py_UCS4** – 32‑bit Unicode code points
//!
//! The functions assume each slice is **already valid Unicode** (as guaranteed by
//! CPython) and therefore skip expensive validation on the hot path.
#![allow(dead_code)]

use core::simd::{Simd};
use core::simd::cmp::SimdPartialOrd;
use core::simd::num::SimdUint; // `.cast()` lives here

/* ---------- Py_UCS1 → UTF‑8 ---------- */

/// Convert a `Py_UCS1` slice (Latin‑1 range 0x00‑0xFF) into UTF‑8.
#[inline]
pub fn ucs1_to_utf8(input: &[u8]) -> Vec<u8> {
    type Block = Simd<u8, 16>;
    const LANES: usize = 16;

    let mut out = Vec::with_capacity(input.len() * 2);
    let mut i = 0;
    while i + LANES <= input.len() {
        let v = Block::from_slice(&input[i..i + LANES]);
        // ASCII fast‑path – every lane ≤ 0x7F
        if v.simd_le(Block::splat(0x7F)).all() {
            out.extend_from_slice(v.as_array());
            i += LANES;
            continue;
        }
        // Mixed block
        for &b in &input[i..i + LANES] {
            if b < 0x80 {
                out.push(b);
            } else {
                out.push(0xC0 | (b >> 6));
                out.push(0x80 | (b & 0x3F));
            }
        }
        i += LANES;
    }
    // Tail
    for &b in &input[i..] {
        if b < 0x80 {
            out.push(b);
        } else {
            out.push(0xC0 | (b >> 6));
            out.push(0x80 | (b & 0x3F));
        }
    }
    out
}

/* ---------- Py_UCS2 → UTF‑8 ---------- */

/// Convert a `Py_UCS2` slice (UTF‑16) into UTF‑8.
#[inline]
pub fn ucs2_to_utf8(input: &[u16]) -> Vec<u8> {
    type Block = Simd<u16, 8>;
    const LANES: usize = 8;

    let mut out = Vec::with_capacity(input.len() * 3);
    let mut i = 0;
    while i + LANES <= input.len() {
        let v = Block::from_slice(&input[i..i + LANES]);
        if v.simd_le(Block::splat(0x007F)).all() {
            let bytes: Simd<u8, 8> = v.cast();
            out.extend_from_slice(bytes.as_array());
            i += LANES;
            continue;
        }
        let mut j = 0;
        while j < LANES {
            let w = input[i + j];
            match w {
                0x0000..=0x007F => out.push(w as u8),
                0x0080..=0x07FF => {
                    out.push((0xC0 | (w >> 6)) as u8);
                    out.push((0x80 | (w & 0x3F)) as u8);
                }
                0xD800..=0xDBFF => {
                    // High surrogate
                    if i + j + 1 >= input.len() {
                        debug_assert!(false, "unterminated surrogate pair at end of slice");
                        break;
                    }
                    let hi = w;
                    let lo = input[i + j + 1];
                    let cp = 0x10000 + (((hi as u32 & 0x3FF) << 10) | (lo as u32 & 0x3FF));
                    push_utf8_4(cp, &mut out);
                    j += 1; // skip low surrogate
                }
                0xDC00..=0xDFFF => unreachable!("unpaired low surrogate"),
                _ => {
                    out.push((0xE0 | (w >> 12)) as u8);
                    out.push((0x80 | ((w >> 6) & 0x3F)) as u8);
                    out.push((0x80 | (w & 0x3F)) as u8);
                }
            }
            j += 1;
        }
        i += LANES;
    }
    // Tail scalar path
    while i < input.len() {
        let w = input[i];
        match w {
            0x0000..=0x007F => out.push(w as u8),
            0x0080..=0x07FF => {
                out.push((0xC0 | (w >> 6)) as u8);
                out.push((0x80 | (w & 0x3F)) as u8);
            }
            0xD800..=0xDBFF => {
                let hi = w;
                let lo = input[i + 1];
                let cp = 0x10000 + (((hi as u32 & 0x3FF) << 10) | (lo as u32 & 0x3FF));
                push_utf8_4(cp, &mut out);
                i += 1;
            }
            0xDC00..=0xDFFF => unreachable!(),
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

/* ---------- Py_UCS4 → UTF‑8 ---------- */

/// Convert a `Py_UCS4` slice (32‑bit code points) into UTF‑8.
#[inline]
pub fn ucs4_to_utf8(input: &[u32]) -> Vec<u8> {
    type Block = Simd<u32, 8>;
    const LANES: usize = 8;

    let mut out = Vec::with_capacity(input.len() * 4);
    let mut i = 0;
    while i + LANES <= input.len() {
        let v = Block::from_slice(&input[i..i + LANES]);
        if v.simd_le(Block::splat(0x7F)).all() {
            let bytes: Simd<u8, 8> = v.cast();
            out.extend_from_slice(bytes.as_array());
            i += LANES;
            continue;
        }
        for &cp in &input[i..i + LANES] {
            push_utf32_scalar(cp, &mut out);
        }
        i += LANES;
    }
    for &cp in &input[i..] {
        push_utf32_scalar(cp, &mut out);
    }
    out
}

/* ---------- helpers ---------- */

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

/* ---------- unit tests ---------- */

#[cfg(test)]
mod tests {
    use super::*;

    /* --- Py_UCS1 --- */
    #[test]
    fn ucs1_ascii() {
        assert_eq!(ucs1_to_utf8(b"Test"), b"Test");
    }

    #[test]
    fn ucs1_extended() {
        let bytes = [0xA1u8, 0xB5u8];
        assert_eq!(ucs1_to_utf8(&bytes), "¡µ".as_bytes());
    }

    #[test]
    fn ucs1_control() {
        let bytes = [0x00u8, 0x1Fu8];
        assert_eq!(ucs1_to_utf8(&bytes), &bytes);
    }

    /* --- Py_UCS2 --- */
    #[test]
    fn ucs2_cjk() {
        let s = "漢字";
        let utf16: Vec<u16> = s.encode_utf16().collect();
        assert_eq!(ucs2_to_utf8(&utf16), s.as_bytes());
    }

    #[test]
    fn ucs2_emoji() {
        let s = "🦀"; // Ferris crab
        let utf16: Vec<u16> = s.encode_utf16().collect();
        assert_eq!(ucs2_to_utf8(&utf16), s.as_bytes());
    }
}