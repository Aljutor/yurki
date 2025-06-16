import re

import pytest

import yurki


PATTERN = r"(hi_how_are_you)|(hello)|(привет\d+)"
JOBS = [1, 4]


def regex_is_match_python(data, pattern):
    compiled_pattern = re.compile(pattern)
    return [bool(compiled_pattern.search(s)) for s in data]


def generate_test_data(size):
    """Generate test data and expected results together."""
    data = [f"making_this_string_long_enough_to_test_hi_привет{i}" for i in range(size)]
    expected = [True for _ in range(size)]
    return data, expected


def generate_no_match_data(size):
    """Generate test data with no matches."""
    data = [f"no_match_here_{i}" for i in range(size)]
    expected = [False for _ in range(size)]
    return data, expected


class TestIsMatch:
    @pytest.mark.parametrize("jobs", JOBS)
    @pytest.mark.parametrize("inplace", [True, False])
    def test_is_match(self, jobs, inplace):
        data, expected = generate_test_data(10)
        result = yurki.regexp.is_match(data=data, pattern=PATTERN, jobs=jobs, inplace=inplace)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_empty_list(self, jobs):
        assert yurki.regexp.is_match(data=[], pattern=PATTERN, jobs=jobs, inplace=False) == []

    @pytest.mark.parametrize("jobs", JOBS)
    def test_no_match(self, jobs):
        data, expected = generate_no_match_data(3)
        result = yurki.regexp.is_match(data=data, pattern=PATTERN, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_unicode(self, jobs):
        data = ["привет_мир", "你好世界", "नमस्ते दुनिया"]
        pattern = r"(мир)|(世界)|(दुनिया)"
        expected = [True, True, True]
        result = yurki.regexp.is_match(data=data, pattern=pattern, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_mixed_matches(self, jobs):
        data = ["привет123", "no_match", "hello_world"]
        expected = [True, False, True]
        result = yurki.regexp.is_match(data=data, pattern=PATTERN, jobs=jobs, inplace=False)
        assert result == expected


class TestBenchIsMatchShort:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(1_000)

    @pytest.mark.benchmark(group="match-short")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_is_match_rust_short(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.is_match_regex_in_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected

    @pytest.mark.benchmark(group="match-short")
    def test_is_match_python_short(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_is_match_python, data, PATTERN)
        assert result == expected


class TestBenchIsMatchMedium:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(100_000)

    @pytest.mark.benchmark(group="match-medium")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_is_match_rust_medium(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.is_match_regex_in_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected

    @pytest.mark.benchmark(group="match-medium")
    def test_is_match_python_medium(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_is_match_python, data, PATTERN)
        assert result == expected


class TestBenchIsMatchLong:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(10_000_000)

    @pytest.mark.benchmark(group="match-long")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_is_match_rust_long(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.is_match_regex_in_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected

    @pytest.mark.benchmark(group="match-long")
    def test_is_match_python_long(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_is_match_python, data, PATTERN)
        assert result == expected
