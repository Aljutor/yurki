//! UCS2 (UTF-16) ↔ UTF-8 conversions

use crate::simd::{U16s, U8s, LANES_U16, LANES_U8, SIMD_THRESHOLD_UCS2, SIMD_THRESHOLD_BYTES, push_utf8_4_bump, push_utf8_4};
use core::simd::cmp::SimdPartialOrd;

/* ===================================================================== */
/*                      Scalar Implementations                           */
/* ===================================================================== */

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

/* ===================================================================== */
/*               Py_UCS2 (UTF-16) → UTF-8                                */
/* ===================================================================== */

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
                // High surrogate: assume valid pair
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
                // High surrogate: assume valid pair
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

#[inline]
pub fn ucs2_to_utf8_bump<'a>(input: &[u16], bump: &'a bumpalo::Bump) -> &'a str {
    if input.len() < SIMD_THRESHOLD_UCS2 {
        return ucs2_to_utf8_scalar_bump(input, bump);
    }

    let mut out = bumpalo::collections::Vec::with_capacity_in(input.len() * 3, bump);
    let mut i = 0;
    let mut ascii_run_start = 0;

    while i + LANES_U16 <= input.len() {
        let v = U16s::from_slice(&input[i..i + LANES_U16]);
        if v.simd_le(U16s::splat(0x7F)).all() {
            i += LANES_U16;
            continue;
        }

        // Found non-ASCII, flush previous ASCII run
        if ascii_run_start < i {
            for &w in &input[ascii_run_start..i] {
                out.push(w as u8);
            }
        }

        // Process the mixed block
        expand_ucs2_block_bump(&input[i..i + LANES_U16], &mut out);
        i += LANES_U16;
        ascii_run_start = i;
    }

    // Flush any remaining ASCII run from the main loop
    if ascii_run_start < i {
        for &w in &input[ascii_run_start..i] {
            out.push(w as u8);
        }
    }

    // Handle the final tail
    if i < input.len() {
        expand_ucs2_block_bump(&input[i..], &mut out);
    }

    let slice = out.into_bump_slice();
    unsafe { core::str::from_utf8_unchecked(slice) }
}

#[inline]
pub fn ucs2_to_utf8(input: &[u16]) -> Vec<u8> {
    if input.len() < SIMD_THRESHOLD_UCS2 {
        return ucs2_to_utf8_scalar(input);
    }

    let mut out: Vec<u8> = Vec::with_capacity(input.len() * 3);
    let mut i = 0;
    let mut ascii_run_start = 0;

    while i + LANES_U16 <= input.len() {
        let v = U16s::from_slice(&input[i..i + LANES_U16]);
        if v.simd_le(U16s::splat(0x7F)).all() {
            i += LANES_U16;
            continue;
        }

        // Found non-ASCII, flush previous ASCII run
        if ascii_run_start < i {
            out.reserve(i - ascii_run_start);
            for &w in &input[ascii_run_start..i] {
                out.push(w as u8);
            }
        }

        // Process the mixed block
        expand_ucs2_block(&input[i..i + LANES_U16], &mut out);
        i += LANES_U16;
        ascii_run_start = i;
    }

    // Flush any remaining ASCII run from the main loop
    if ascii_run_start < i {
        out.reserve(i - ascii_run_start);
        for &w in &input[ascii_run_start..i] {
            out.push(w as u8);
        }
    }

    // Handle the final tail
    if i < input.len() {
        expand_ucs2_block(&input[i..], &mut out);
    }

    out
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

/* ===================================================================== */
/*                               Tests                                   */
/* ===================================================================== */

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn utf8_to_ucs2_basic() {
        let ascii = "Hello";
        let mut ucs2_buf = [0u16; 10];

        let len2 = utf8_to_ucs2_simd(ascii.as_bytes(), &mut ucs2_buf);

        assert_eq!(len2, 5);
    }

    #[test]
    fn roundtrip_utf8_ucs2() {
        let test_cases = vec!["Hello", "café", "🦀", "Hello, 世界!"];

        for case in test_cases {
            // UTF-8 → UCS-2 → UTF-8
            let mut ucs2_buf = vec![0u16; case.chars().count() * 2]; // Extra space for surrogates
            let ucs2_len = utf8_to_ucs2_simd(case.as_bytes(), &mut ucs2_buf);
            let back_to_utf8 = ucs2_to_utf8(&ucs2_buf[..ucs2_len]);
            assert_eq!(case.as_bytes(), &back_to_utf8);
        }
    }
}
