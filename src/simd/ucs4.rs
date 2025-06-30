//! UCS4 (UTF-32) ↔ UTF-8 conversions

use crate::simd::{U32s, U8s, LANES_U32, LANES_U8, SIMD_THRESHOLD_UCS4, SIMD_THRESHOLD_BYTES, push_utf8_4_bump, push_utf8_4};
use core::simd::cmp::SimdPartialOrd;

/* ===================================================================== */
/*                      Scalar Implementations                           */
/* ===================================================================== */

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
    if input.len() < SIMD_THRESHOLD_UCS4 {
        return ucs4_to_utf8_scalar_bump(input, bump);
    }

    let mut out = bumpalo::collections::Vec::with_capacity_in(input.len() * 4, bump);
    let mut i = 0;
    let mut ascii_run_start = 0;

    while i + LANES_U32 <= input.len() {
        let v = U32s::from_slice(&input[i..i + LANES_U32]);
        if v.simd_le(U32s::splat(0x7F)).all() {
            i += LANES_U32;
            continue;
        }

        // Found non-ASCII, flush previous ASCII run
        if ascii_run_start < i {
            for &cp in &input[ascii_run_start..i] {
                out.push(cp as u8);
            }
        }

        // Process the mixed block
        for &cp in &input[i..i + LANES_U32] {
            push_utf32_scalar_bump(cp, &mut out);
        }
        i += LANES_U32;
        ascii_run_start = i;
    }

    // Flush any remaining ASCII run from the main loop
    if ascii_run_start < i {
        for &cp in &input[ascii_run_start..i] {
            out.push(cp as u8);
        }
    }

    // Handle the final tail
    if i < input.len() {
        for &cp in &input[i..] {
            push_utf32_scalar_bump(cp, &mut out);
        }
    }

    let slice = out.into_bump_slice();
    unsafe { core::str::from_utf8_unchecked(slice) }
}

#[inline]
pub fn ucs4_to_utf8(input: &[u32]) -> Vec<u8> {
    if input.len() < SIMD_THRESHOLD_UCS4 {
        return ucs4_to_utf8_scalar(input);
    }

    let mut out: Vec<u8> = Vec::with_capacity(input.len() * 4);
    let mut i = 0;
    let mut ascii_run_start = 0;

    while i + LANES_U32 <= input.len() {
        let v = U32s::from_slice(&input[i..i + LANES_U32]);
        if v.simd_le(U32s::splat(0x7F)).all() {
            i += LANES_U32;
            continue;
        }

        // Found non-ASCII, flush previous ASCII run
        if ascii_run_start < i {
            out.reserve(i - ascii_run_start);
            for &cp in &input[ascii_run_start..i] {
                out.push(cp as u8);
            }
        }

        // Process the mixed block
        for &cp in &input[i..i + LANES_U32] {
            push_utf32_scalar(cp, &mut out);
        }
        i += LANES_U32;
        ascii_run_start = i;
    }

    // Flush any remaining ASCII run from the main loop
    if ascii_run_start < i {
        out.reserve(i - ascii_run_start);
        for &cp in &input[ascii_run_start..i] {
            out.push(cp as u8);
        }
    }

    // Handle the final tail
    if i < input.len() {
        for &cp in &input[i..] {
            push_utf32_scalar(cp, &mut out);
        }
    }

    out
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

/* ===================================================================== */
/*                               Tests                                   */
/* ===================================================================== */

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn utf8_to_ucs4_basic() {
        let ascii = "Hello";
        let mut ucs4_buf = [0u32; 10];

        let len4 = utf8_to_ucs4_simd(ascii.as_bytes(), &mut ucs4_buf);

        assert_eq!(len4, 5);
        assert_eq!(&ucs4_buf[..len4], &[72, 101, 108, 108, 111]);
    }

    #[test]
    fn roundtrip_utf8_ucs4() {
        let test_cases = vec!["Hello", "café", "🦀", "Hello, 世界!"];

        for case in test_cases {
            // UTF-8 → UCS-4 → UTF-8
            let mut ucs4_buf = vec![0u32; case.chars().count()];
            let ucs4_len = utf8_to_ucs4_simd(case.as_bytes(), &mut ucs4_buf);
            let back_to_utf8 = ucs4_to_utf8(&ucs4_buf[..ucs4_len]);
            assert_eq!(case.as_bytes(), &back_to_utf8);
        }
    }

    #[test]
    fn output_length_bounds() {
        // UCS4: output <= input.len() * 4
        let unicode_input: Vec<u32> = vec![0x1F984, 0x1F680]; // 🦄🚀
        let utf8_output = ucs4_to_utf8(&unicode_input);
        assert!(utf8_output.len() <= unicode_input.len() * 4);
    }
}
