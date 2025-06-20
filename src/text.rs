use regex::Regex;
use std::borrow::Cow;

pub fn find_in_string(string: &str, pattern: &Regex) -> String {
    pattern
        .find(string)
        .map(|m| m.as_str())
        .unwrap_or("")
        .to_string()
}

pub fn is_match_in_string(string: &str, pattern: &Regex) -> bool {
    pattern.is_match(string)
}

pub fn capture_regex_in_string(string: &str, pattern: &Regex) -> Vec<String> {
    match pattern.captures(string) {
        Some(caps) => caps
            .iter()
            .map(|m| m.map_or(String::new(), |m| m.as_str().to_string()))
            .collect(),
        None => Vec::new(),
    }
}

pub fn split_by_regexp_string(string: &str, pattern: &Regex) -> Vec<String> {
    pattern.split(string).map(str::to_string).collect()
}

pub fn replace_regexp_in_string(
    string: &str,
    pattern: &Regex,
    replacement: &str,
    count: usize,
) -> String {
    let result: Cow<str> = if count == 0 {
        pattern.replace_all(string, replacement)
    } else {
        pattern.replacen(string, count, replacement)
    };

    result.into_owned()
}
