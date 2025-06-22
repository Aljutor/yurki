use regex::Regex;

pub fn find_in_string(string: &str, pattern: &Regex) -> String {
    let mat = pattern.find(string);
    mat.map(|x| x.as_str()).unwrap_or("").to_string()
}

pub fn is_match_in_string(string: &str, pattern: &Regex) -> bool {
    pattern.is_match(string)
}

pub fn capture_regex_in_string(string: &str, pattern: &Regex) -> Vec<String> {
    pattern
        .captures(string)
        .map(|caps| {
            caps.iter()
                .map(|m| m.map(|m| m.as_str()).unwrap_or(""))
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_else(Vec::new)
}

pub fn split_by_regexp_string(string: &str, pattern: &Regex) -> Vec<String> {
    pattern.split(string).map(|s| s.to_string()).collect()
}

pub fn replace_regexp_in_string(
    string: &str,
    pattern: &Regex,
    replacement: &str,
    count: usize,
) -> String {
    if count == 0 {
        pattern.replace_all(string, replacement).to_string()
    } else {
        pattern.replacen(string, count, replacement).to_string()
    }
}
