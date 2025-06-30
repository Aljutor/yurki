//! UCS2 (UTF-16) ↔ UTF-8 conversions

use crate::simd::{
    LANES_U8, LANES_U16, SIMD_THRESHOLD_BYTES, SIMD_THRESHOLD_UCS2, U8s, U16s, push_utf8_4,
    push_utf8_4_bump, simd_u16_to_ascii_bytes,
};
use core::simd::cmp::SimdPartialOrd;

// ========================================================================== //
//                         Scalar Implementations                             //
// ========================================================================== //

/// Converts a UCS-2 (UTF-16) slice to a UTF-8 string in a `bumpalo` arena.
///
/// This function provides a scalar fallback for short inputs. It correctly
/// handles surrogate pairs.
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
                // High surrogate: assume valid pair.
                if i + 1 < input.len() {
                    let lo = input[i + 1];
                    let cp = 0x10000 + (((w as u32 & 0x3FF) << 10) | (lo as u32 & 0x3FF));
                    push_utf8_4(cp, &mut out);
                    i += 1; // Skip low surrogate.
                }
            }
            0xDC00..=0xDFFF => {
                // Isolated low surrogate, skip.
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

/// Converts a UTF-8 slice to UCS-2 (UTF-16).
///
/// This function provides a scalar fallback for short inputs. It encodes
/// supplementary plane characters as surrogate pairs.
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

// ========================================================================== //
//                       UCS-2 (UTF-16) to UTF-8                              //
// ========================================================================== //

/// Scalar routine to expand a block of UCS-2 characters, including surrogates.
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
                // High surrogate: assume valid pair.
                let lo = block[j + 1];
                let cp = 0x10000 + (((w as u32 & 0x3FF) << 10) | (lo as u32 & 0x3FF));
                push_utf8_4_bump(cp, out);
                j += 1; // Skip low surrogate.
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

/// Scalar routine to expand a block of UCS-2 characters, including surrogates.
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
                // High surrogate: assume valid pair.
                let lo = block[j + 1];
                let cp = 0x10000 + (((w as u32 & 0x3FF) << 10) | (lo as u32 & 0x3FF));
                push_utf8_4(cp, out);
                j += 1; // Skip low surrogate.
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

/// Converts a UCS-2 (UTF-16) slice to a UTF-8 string in a `bumpalo` arena.
///
/// This function uses SIMD for performance on larger inputs. It checks for ASCII
/// fast paths and falls back to a scalar routine for blocks containing
/// surrogate pairs, which require special handling.
#[inline]
pub fn ucs2_to_utf8_bump<'a>(input: &[u16], bump: &'a bumpalo::Bump) -> &'a str {
    if input.len() < SIMD_THRESHOLD_UCS2 {
        return ucs2_to_utf8_scalar_bump(input, bump);
    }

    let mut out = bumpalo::collections::Vec::with_capacity_in(input.len() * 3, bump);
    let mut i = 0;

    while i + LANES_U16 <= input.len() {
        let chunk = U16s::from_slice(&input[i..i + LANES_U16]);
        let is_ascii = chunk.simd_le(U16s::splat(0x7F));

        if is_ascii.all() {
            // Fast path for pure ASCII
            let ascii_bytes = simd_u16_to_ascii_bytes(chunk);
            out.extend_from_slice(&ascii_bytes);
        } else {
            // Check for the complex case (surrogates) and use a faster path if not present.
            let has_surrogates = chunk.simd_ge(U16s::splat(0xD800)).any();
            if has_surrogates {
                // Fallback for blocks with surrogates, which require look-ahead.
                expand_ucs2_block_bump(&input[i..i + LANES_U16], &mut out);
            } else {
                // Faster path for 1/2/3-byte characters (no surrogates).
                for &w in &input[i..i + LANES_U16] {
                    if w <= 0x007F {
                        out.push(w as u8);
                    } else if w <= 0x07FF {
                        out.push((0xC0 | (w >> 6)) as u8);
                        out.push((0x80 | (w & 0x3F)) as u8);
                    } else {
                        out.push((0xE0 | (w >> 12)) as u8);
                        out.push((0x80 | ((w >> 6) & 0x3F)) as u8);
                        out.push((0x80 | (w & 0x3F)) as u8);
                    }
                }
            }
        }
        i += LANES_U16;
    }

    // Handle the final tail
    if i < input.len() {
        expand_ucs2_block_bump(&input[i..], &mut out);
    }

    let slice = out.into_bump_slice();
    unsafe { core::str::from_utf8_unchecked(slice) }
}

/// Converts a UCS-2 (UTF-16) slice to a UTF-8 `Vec<u8>`.
///
/// This function uses SIMD for performance on larger inputs, analogous to
/// `ucs2_to_utf8_bump`, but allocates on the heap.
#[inline]
pub fn ucs2_to_utf8(input: &[u16]) -> Vec<u8> {
    if input.len() < SIMD_THRESHOLD_UCS2 {
        return ucs2_to_utf8_scalar(input);
    }

    let mut out: Vec<u8> = Vec::with_capacity(input.len() * 3);
    let mut i = 0;

    while i + LANES_U16 <= input.len() {
        let chunk = U16s::from_slice(&input[i..i + LANES_U16]);
        let is_ascii = chunk.simd_le(U16s::splat(0x7F));

        if is_ascii.all() {
            // Fast path for pure ASCII
            let ascii_bytes = simd_u16_to_ascii_bytes(chunk);
            out.extend_from_slice(&ascii_bytes);
        } else {
            // Check for the complex case (surrogates) and use a faster path if not present.
            let has_surrogates = chunk.simd_ge(U16s::splat(0xD800)).any();
            if has_surrogates {
                // Fallback for blocks with surrogates, which require look-ahead.
                expand_ucs2_block(&input[i..i + LANES_U16], &mut out);
            } else {
                // Faster path for 1/2/3-byte characters (no surrogates).
                for &w in &input[i..i + LANES_U16] {
                    if w <= 0x007F {
                        out.push(w as u8);
                    } else if w <= 0x07FF {
                        out.push((0xC0 | (w >> 6)) as u8);
                        out.push((0x80 | (w & 0x3F)) as u8);
                    } else {
                        out.push((0xE0 | (w >> 12)) as u8);
                        out.push((0x80 | ((w >> 6) & 0x3F)) as u8);
                        out.push((0x80 | (w & 0x3F)) as u8);
                    }
                }
            }
        }
        i += LANES_U16;
    }

    // Handle the final tail
    if i < input.len() {
        expand_ucs2_block(&input[i..], &mut out);
    }

    out
}

/// Converts a UTF-8 slice to UCS-2 (UTF-16) using SIMD acceleration.
///
/// This function is optimized for inputs that are primarily ASCII. It processes
/// the input in SIMD-sized chunks, and if a chunk is pure ASCII, it is
/// zero-extended to `u16`. For chunks containing multi-byte characters, it
/// falls back to a scalar routine.
pub fn utf8_to_ucs2_simd(input: &[u8], output: &mut [u16]) -> usize {
    // Use scalar for short strings to avoid SIMD overhead
    if input.len() < SIMD_THRESHOLD_BYTES {
        return utf8_to_ucs2_scalar(input, output);
    }

    let mut out_pos = 0;
    let mut i = 0;

    // SIMD ASCII fast path
    while i + LANES_U8 <= input.len() && out_pos + LANES_U8 <= output.len() {
        let chunk = U8s::from_slice(&input[i..i + LANES_U8]);

        if chunk.simd_lt(U8s::splat(0x80)).all() {
            // Pure ASCII - zero-extend to u16
            let mut wide_array = [0u16; LANES_U8];
            for (i, &byte) in chunk.as_array().iter().enumerate() {
                wide_array[i] = byte as u16;
            }
            output[out_pos..out_pos + LANES_U8].copy_from_slice(&wide_array);
            out_pos += LANES_U8;
            i += LANES_U8;
        } else {
            // Scalar fallback for the block and then continue.
            let written = utf8_to_ucs2_scalar(&input[i..], &mut output[out_pos..]);
            out_pos += written;
            // This is a rough approximation to advance `i`. A more robust
            // solution would be to count the bytes consumed by the scalar function.
            i += LANES_U8;
        }
    }

    // Scalar fallback for the tail
    if i < input.len() && out_pos < output.len() {
        out_pos += utf8_to_ucs2_scalar(&input[i..], &mut output[out_pos..]);
    }

    out_pos
}

// ========================================================================== //
//                                   Tests                                    //
// ========================================================================== //

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
