use regex::Regex;

pub fn find_in_string(string: &str, pattern: &Regex) -> String {
    let mat = pattern.find(string);
    mat.map(|x| x.as_str()).unwrap_or("").to_string()
}

pub fn is_match_in_string(string: &str, pattern: &Regex) -> bool {
    pattern.is_match(string)
}
