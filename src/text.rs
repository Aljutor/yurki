use regex::Regex;

pub fn find_in_string(string: &str, pattern: &Regex) -> String {
    let mat = pattern.find(string);
    mat.map(|x| x.as_str()).unwrap_or("").to_string()
}

pub fn tokenize_word_bound(text: &str, ngram: (usize, usize)) -> Vec<String> {
    let mut ngrams = Vec::<String>::new();

    text.split_whitespace().for_each(|word| {
        let mut chars: Vec<char> = word.chars().collect();
        chars.push(' ');
        chars.insert(0, ' ');

        let chars = chars.as_slice();

        for n in ngram.0..ngram.1 + 1 {
            let mut offset = 0;
            let ngram = chars[offset..offset + n].iter().collect();
            ngrams.push(ngram);

            while offset + n < chars.len() {
                offset += 1;
                let ngram = chars[offset..offset + n].iter().collect();
                ngrams.push(ngram);
            }

            if offset == 0 {
                break;
            }
        }
    });

    ngrams
}
