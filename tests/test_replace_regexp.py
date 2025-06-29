import re

import pytest

import yurki


PATTERN = r"[Tt]est\s+\w{6,}"
REPLACEMENT = "MATCHED"
JOBS = [1, 4]


def regex_replace_python(data, pattern, replacement, count=1):
    compiled_pattern = re.compile(pattern)
    if count == 0:
        return [compiled_pattern.sub(replacement, s) for s in data]
    else:
        return [compiled_pattern.sub(replacement, s, count=count) for s in data]


def generate_test_data(size):
    """Generate test data and expected results together."""
    data = [f"some text with test string and more test content {i}" for i in range(size)]
    expected = [f"some text with MATCHED and more test content {i}" for i in range(size)]  # Only first occurrence
    return data, expected


class TestReplace:
    @pytest.mark.parametrize("jobs", JOBS)
    @pytest.mark.parametrize("inplace", [False, True])
    def test_replace_default(self, jobs, inplace):
        data, expected = generate_test_data(10)
        result = yurki.regexp.replace(data=data, pattern=PATTERN, replacement=REPLACEMENT, jobs=jobs, inplace=inplace)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_empty_list(self, jobs):
        assert yurki.regexp.replace(data=[], pattern=PATTERN, replacement=REPLACEMENT, jobs=jobs, inplace=False) == []

    @pytest.mark.parametrize("jobs", JOBS)
    def test_no_match(self, jobs):
        data = ["abc", "def", "ghi"]
        expected = ["abc", "def", "ghi"]
        result = yurki.regexp.replace(data=data, pattern=PATTERN, replacement=REPLACEMENT, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_replace_all_count_zero(self, jobs):
        data = ["test string with test and more test content"]
        expected = ["MATCHED with test and more MATCHED"]
        result = yurki.regexp.replace(
            data=data, pattern=PATTERN, replacement=REPLACEMENT, count=0, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_replace_count_one(self, jobs):
        data = ["test string with test and more test content"]
        expected = ["MATCHED with test and more test content"]
        result = yurki.regexp.replace(
            data=data, pattern=PATTERN, replacement=REPLACEMENT, count=1, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_replace_count_two(self, jobs):
        data = ["test string with test and more test content"]
        expected = ["MATCHED with test and more MATCHED"]
        result = yurki.regexp.replace(
            data=data, pattern=PATTERN, replacement=REPLACEMENT, count=2, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_replace_count_more_than_matches(self, jobs):
        data = ["test string with test content"]
        expected = ["MATCHED with MATCHED"]
        result = yurki.regexp.replace(
            data=data, pattern=PATTERN, replacement=REPLACEMENT, count=10, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_replace_empty_string(self, jobs):
        data = ["test string test"]
        expected = [" test"]
        result = yurki.regexp.replace(data=data, pattern=PATTERN, replacement="", count=1, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_replace_with_groups(self, jobs):
        data = ["name: John", "name: Jane"]
        pattern = r"name: (\w+)"
        replacement = r"Hello $1"  # Rust uses $1 instead of \1
        expected = ["Hello John", "Hello Jane"]
        result = yurki.regexp.replace(data=data, pattern=pattern, replacement=replacement, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_unicode(self, jobs):
        data = ["привет мир привет", "你好 世界 你好", "नमस्ते दुनिया नमस्ते"]
        pattern = r"привет|你好|नमस्ते"
        replacement = "HELLO"
        expected = ["HELLO мир привет", "HELLO 世界 你好", "HELLO दुनिया नमस्ते"]
        result = yurki.regexp.replace(
            data=data, pattern=pattern, replacement=replacement, count=1, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_unicode_replace_all(self, jobs):
        data = ["привет мир привет", "你好 世界 你好", "नमस्ते दुनिया नमस्ते"]
        pattern = r"привет|你好|नमस्ते"
        replacement = "HELLO"
        expected = ["HELLO мир HELLO", "HELLO 世界 HELLO", "HELLO दुनिया HELLO"]
        result = yurki.regexp.replace(
            data=data, pattern=pattern, replacement=replacement, count=0, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_case_insensitive(self, jobs):
        data = ["Test string with TEST and more test content"]
        pattern = r"test"
        replacement = "REPLACED"
        expected = ["REPLACED string with TEST and more test content"]
        result = yurki.regexp.replace(
            data=data, pattern=pattern, replacement=replacement, case=True, count=1, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_complex_regex_patterns(self, jobs):
        # Email pattern replacement
        data = ["Contact: user@example.com and admin@test.org for support"]
        pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        replacement = "[EMAIL]"
        expected = ["Contact: [EMAIL] and admin@test.org for support"]
        result = yurki.regexp.replace(
            data=data, pattern=pattern, replacement=replacement, count=1, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_complex_digit_letter_patterns(self, jobs):
        # Replace digits followed by letters (pattern includes the letter)
        data = ["abc123def 456 ghi 789xyz"]
        pattern = r"\d+[a-z]"
        replacement = "NUM"
        expected = ["abcNUMef 456 ghi NUMyz"]
        result = yurki.regexp.replace(
            data=data, pattern=pattern, replacement=replacement, count=0, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_complex_word_boundaries(self, jobs):
        # Word boundary pattern for whole words only
        data = ["testing test tested tester"]
        pattern = r"\btest\b"
        replacement = "EXAM"
        expected = ["testing EXAM tested tester"]
        result = yurki.regexp.replace(
            data=data, pattern=pattern, replacement=replacement, count=1, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_complex_alternation_groups(self, jobs):
        # Complex alternation with groups
        data = ["color: red, colour: blue, color: green"]
        pattern = r"colou?r: (\w+)"
        replacement = r"paint: $1"
        expected = ["paint: red, colour: blue, color: green"]
        result = yurki.regexp.replace(
            data=data, pattern=pattern, replacement=replacement, count=1, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_complex_quantifiers(self, jobs):
        # Non-greedy quantifiers (without backreferences)
        data = ["<tag>content</tag> and <div>more</div>"]
        pattern = r"<\w+?>.*?</\w+?>"
        replacement = "[ELEMENT]"
        expected = ["[ELEMENT] and <div>more</div>"]
        result = yurki.regexp.replace(
            data=data, pattern=pattern, replacement=replacement, count=1, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_complex_unicode_patterns(self, jobs):
        # Unicode character classes and complex patterns
        data = ["Price: $123.45 €67.89 ¥1000"]
        pattern = r"[\$€¥]\d+(?:\.\d+)?"
        replacement = "[PRICE]"
        expected = ["Price: [PRICE] €67.89 ¥1000"]
        result = yurki.regexp.replace(
            data=data, pattern=pattern, replacement=replacement, count=1, jobs=jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_complex_nested_groups(self, jobs):
        # Nested capturing groups
        data = ["Date: 2023-12-25 and Time: 14:30:45"]
        pattern = r"(\d{4})-(\d{2})-(\d{2})"
        replacement = r"$3/$2/$1"
        expected = ["Date: 25/12/2023 and Time: 14:30:45"]
        result = yurki.regexp.replace(
            data=data, pattern=pattern, replacement=replacement, count=1, jobs=jobs, inplace=False
        )
        assert result == expected


class TestBenchReplaceShort:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(1_000)

    @pytest.mark.benchmark(group="replace-short")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_replace_rust_short(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(
            yurki.internal.replace_regexp_in_string, data, PATTERN, REPLACEMENT, 1, False, jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.benchmark(group="replace-short")
    def test_replace_python_short(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_replace_python, data, PATTERN, REPLACEMENT, 1)
        assert result == expected


class TestBenchReplaceMedium:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(100_000)

    @pytest.mark.benchmark(group="replace-medium")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_replace_rust_medium(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(
            yurki.internal.replace_regexp_in_string, data, PATTERN, REPLACEMENT, 1, False, jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.benchmark(group="replace-medium")
    def test_replace_python_medium(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_replace_python, data, PATTERN, REPLACEMENT, 1)
        assert result == expected


class TestBenchReplaceLong:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(10_000_000)

    @pytest.mark.benchmark(group="replace-long")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_replace_rust_long(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(
            yurki.internal.replace_regexp_in_string, data, PATTERN, REPLACEMENT, 1, False, jobs, inplace=False
        )
        assert result == expected

    @pytest.mark.benchmark(group="replace-long")
    def test_replace_python_long(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_replace_python, data, PATTERN, REPLACEMENT, 1)
        assert result == expected
