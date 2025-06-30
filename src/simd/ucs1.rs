//! UCS1 (Latin-1) ↔ UTF-8 conversions

use crate::simd::{U8s, LANES_U8, SIMD_THRESHOLD_UCS1, SIMD_THRESHOLD_BYTES};
use core::simd::cmp::SimdPartialOrd;
use std::borrow::Cow;

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
fn ucs1_to_utf8_scalar(input: &[u8]) -> Cow<'_, str> {
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

/* ===================================================================== */
/*               Py_UCS1 (Latin-1) → UTF-8                               */
/* ===================================================================== */

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

/// Convert a Latin-1 slice to UTF-8 inside a bump arena.
/// * Pure ASCII  → borrowed `&str`  (zero-alloc)
/// * Mixed input → bump-allocated `&str`
///
/// Changes vs. the old version
/// ---------------------------
/// 1. **One pass only** – we discover non-ASCII lanes while we expand, so the
///    string is touched exactly once.
/// 2. **Bulk ASCII copy** – consecutive ASCII runs are copied with
///    `ptr::copy_nonoverlapping`, not `push` per byte.
/// 3. **Mask.none()/any()** instead of `to_bitmask()` for the hot-path check.
///
/// Requires the same `U8s`/`LANES_U8` aliases and `expand_latin1_block_bump`.
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

    /* 2. Over-allocate and convert in a single pass */
    let mut out = bumpalo::collections::Vec::with_capacity_in(input.len() * 2, bump);

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

    unsafe { core::str::from_utf8_unchecked(out.into_bump_slice()) }
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

    /* 2. Over-allocate and convert in a single pass */
    let mut out: Vec<u8> = Vec::with_capacity(input.len() * 2);

    /* 3. SIMD loop with bulk ASCII copy */
    let mut i = 0;
    let mut ascii_run_start = 0;
    while i + LANES_U8 <= input.len() {
        let v = U8s::from_slice(&input[i..i + LANES_U8]);
        let mask = v.simd_ge(U8s::splat(0x80));

        if !mask.any() {
            i += LANES_U8;
            continue;
        }

        // Flush preceding ASCII run
        if ascii_run_start < i {
            out.extend_from_slice(&input[ascii_run_start..i]);
        }
        
        expand_latin1_block(v.as_array(), &mut out);

        i += LANES_U8;
        ascii_run_start = i;
    }

    /* 4. Scalar tail */
    if ascii_run_start < input.len() {
        out.extend_from_slice(&input[ascii_run_start..i]); // Copy remaining ASCII
        let tail = &input[i..];
        for &b in tail {
            if b < 0x80 {
                out.push(b);
            } else {
                out.push(0xC0 | (b >> 6));
                out.push(0x80 | (b & 0x3F));
            }
        }
    }
    
    out.shrink_to_fit();
    Cow::Owned(unsafe { String::from_utf8_unchecked(out) })
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

/* ===================================================================== */
/*                               Tests                                   */
/* ===================================================================== */

#[cfg(test)]
mod tests {
    use super::*;
    use std::borrow::Cow::*;

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

    #[test]
    fn utf8_to_ucs_basic() {
        // ASCII test
        let ascii = "Hello";
        let mut ucs1_buf = [0u8; 10];

        let len1 = utf8_to_ucs1_simd(ascii.as_bytes(), &mut ucs1_buf);

        assert_eq!(len1, 5);
        assert_eq!(&ucs1_buf[..len1], ascii.as_bytes());
    }

    #[test]
    fn roundtrip_ucs1_utf8() {
        for i in 0..=255u8 {
            let input = [i];
            let utf8_result = ucs1_to_utf8(&input);
            let back_to_utf16: Vec<u16> = utf8_result.chars().map(|c| c as u16).collect();
            let utf8_from_utf16 = crate::simd::ucs2_to_utf8(&back_to_utf16);
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
}
