use regex::Regex;
use std::borrow::Cow;

pub fn find_in_string<'a>(string: &'a str, _pattern: &Regex) -> Cow<'a, str> {
    _pattern
        .find(string)
        .map(|m| Cow::Borrowed(m.as_str()))
        .unwrap_or(Cow::Borrowed(""))
}

pub fn is_match_in_string(string: &str, pattern: &Regex) -> bool {
    pattern.is_match(string)
}

pub fn capture_regex_in_string<'a>(string: &'a str, _pattern: &Regex) -> Vec<Cow<'a, str>> {
    _pattern
        .captures(string)
        .map(|caps| {
            caps.iter()
                .map(|m| {
                    m.map(|m| Cow::Borrowed(m.as_str()))
                        .unwrap_or(Cow::Borrowed(""))
                })
                .collect()
        })
        .unwrap_or_else(Vec::new)
}

pub fn split_by_regexp_string<'a>(string: &'a str, _pattern: &Regex) -> Vec<Cow<'a, str>> {
    _pattern.split(string).map(Cow::Borrowed).collect()
}

pub fn replace_regexp_in_string<'a>(
    string: &'a str,
    _pattern: &Regex,
    replacement: &str,
    count: usize,
) -> Cow<'a, str> {
    if count == 0 {
        _pattern.replace_all(string, replacement)
    } else {
        _pattern.replacen(string, count, replacement)
    }
}
