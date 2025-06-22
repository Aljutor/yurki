import re

import pytest

import yurki


PATTERN = r"(hi_how_are_you)|(hello)|(привет\d+)"
JOBS = [1, 4]


def regex_find_python(data, pattern):
    compiled_pattern = re.compile(pattern)
    return [match.group(0) if (match := compiled_pattern.search(s)) else "" for s in data]


def generate_test_data(size):
    """Generate test data and expected results together."""
    data = [f"making_this_string_long_enough_to_test_hi_привет{i}" for i in range(size)]
    expected = [f"привет{i}" for i in range(size)]
    return data, expected


class TestFind:
    @pytest.mark.parametrize("jobs", JOBS)
    @pytest.mark.parametrize("inplace", [False, True])
    def test_find(self, jobs, inplace):
        data, expected = generate_test_data(10)
        result = yurki.regexp.find(data=data, pattern=PATTERN, jobs=jobs, inplace=inplace)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_empty_list(self, jobs):
        assert yurki.regexp.find(data=[], pattern=PATTERN, jobs=jobs, inplace=False) == []

    @pytest.mark.parametrize("jobs", JOBS)
    def test_no_match(self, jobs):
        data = ["a", "b", "c"]
        expected = ["", "", ""]
        result = yurki.regexp.find(data=data, pattern=PATTERN, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_unicode(self, jobs):
        data = ["привет_мир", "你好世界", "नमस्ते दुनिया"]
        pattern = r"(мир)|(世界)|(दुनिया)"
        expected = ["мир", "世界", "दुनिया"]
        result = yurki.regexp.find(data=data, pattern=pattern, jobs=jobs, inplace=False)
        assert result == expected


class TestBenchFindShort:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(1_000)

    @pytest.mark.benchmark(group="find-short")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_find_rust_short(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.find_regex_in_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected

    @pytest.mark.benchmark(group="find-short")
    def test_find_python_short(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_find_python, data, PATTERN)
        assert result == expected


class TestBenchFindMedium:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(100_000)

    @pytest.mark.benchmark(group="find-medium")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_find_rust_medium(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.find_regex_in_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected

    @pytest.mark.benchmark(group="find-medium")
    def test_find_python_medium(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_find_python, data, PATTERN)
        assert result == expected


class TestBenchFindLong:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(10_000_000)

    @pytest.mark.benchmark(group="find-long")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_find_rust_long(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.find_regex_in_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected

    @pytest.mark.benchmark(group="find-long")
    def test_find_python_long(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_find_python, data, PATTERN)
        assert result == expected
