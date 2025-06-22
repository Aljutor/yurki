import re

import pytest

import yurki


PATTERN = r"(hi_how_are_you)|(hello)|(привет\d+)"
JOBS = [1, 4]


def regex_capture_python(data, pattern):
    compiled_pattern = re.compile(pattern)
    return [[match.group(0)] + list(match.groups()) if (match := compiled_pattern.search(s)) else [] for s in data]


def generate_test_data(size):
    """Generate test data and expected results together."""
    data = [f"making_this_string_long_enough_to_test_hi_привет{i}" for i in range(size)]
    # Rust ordering: group(0), group(1), group(2), group(3)
    # For pattern (hi_how_are_you)|(hello)|(привет\d+), matching "привет{i}":
    # group(0) = "привет{i}" (full match), group(1) = None, group(2) = None, group(3) = "привет{i}"
    expected = [[f"привет{i}", None, None, f"привет{i}"] for i in range(size)]
    return data, expected


class TestCapture:
    @pytest.mark.parametrize("jobs", JOBS)
    @pytest.mark.parametrize("inplace", [False, True])
    def test_capture(self, jobs, inplace):
        data, expected = generate_test_data(10)
        result = yurki.regexp.capture(data=data, pattern=PATTERN, jobs=jobs, inplace=inplace)
        # Convert None to empty string for comparison as Rust returns empty strings
        expected_rust = [["" if x is None else x for x in groups] for groups in expected]
        assert result == expected_rust

    @pytest.mark.parametrize("jobs", JOBS)
    def test_empty_list(self, jobs):
        assert yurki.regexp.capture(data=[], pattern=PATTERN, jobs=jobs, inplace=False) == []

    @pytest.mark.parametrize("jobs", JOBS)
    def test_no_match(self, jobs):
        data = ["a", "b", "c"]
        expected = [[], [], []]
        result = yurki.regexp.capture(data=data, pattern=PATTERN, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_capture_groups(self, jobs):
        data = ["hello world", "hi_how_are_you there", "привет123 test"]
        pattern = r"(hello|hi_how_are_you|привет\d+)"
        expected = [["hello", "hello"], ["hi_how_are_you", "hi_how_are_you"], ["привет123", "привет123"]]
        result = yurki.regexp.capture(data=data, pattern=pattern, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_multiple_groups(self, jobs):
        data = ["name: John, age: 25", "name: Jane, age: 30"]
        pattern = r"name: (\w+), age: (\d+)"
        # Rust ordering: group(0), group(1), group(2)
        expected = [["name: John, age: 25", "John", "25"], ["name: Jane, age: 30", "Jane", "30"]]
        result = yurki.regexp.capture(data=data, pattern=pattern, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_unicode(self, jobs):
        data = ["привет мир", "你好 世界", "नमस्ते दुनिया"]
        pattern = r"(मир|世界|दुनिया)"
        expected = [[], ["世界", "世界"], ["दुनिया", "दुनिया"]]
        result = yurki.regexp.capture(data=data, pattern=pattern, jobs=jobs, inplace=False)
        assert result == expected


class TestBenchCaptureShort:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(1_000)

    @pytest.mark.benchmark(group="capture-short")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_capture_rust_short(self, jobs, benchmark, test_data):
        data, expected = test_data
        expected_rust = [["" if x is None else x for x in groups] for groups in expected]
        result = benchmark(yurki.internal.capture_regex_in_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected_rust

    @pytest.mark.benchmark(group="capture-short")
    def test_capture_python_short(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_capture_python, data, PATTERN)
        assert result == expected


class TestBenchCaptureMedium:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(100_000)

    @pytest.mark.benchmark(group="capture-medium")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_capture_rust_medium(self, jobs, benchmark, test_data):
        data, expected = test_data
        expected_rust = [["" if x is None else x for x in groups] for groups in expected]
        result = benchmark(yurki.internal.capture_regex_in_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected_rust

    @pytest.mark.benchmark(group="capture-medium")
    def test_capture_python_medium(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_capture_python, data, PATTERN)
        assert result == expected


class TestBenchCaptureLong:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(10_000_000)

    @pytest.mark.benchmark(group="capture-long")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_capture_rust_long(self, jobs, benchmark, test_data):
        data, expected = test_data
        expected_rust = [["" if x is None else x for x in groups] for groups in expected]
        result = benchmark(yurki.internal.capture_regex_in_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected_rust

    @pytest.mark.benchmark(group="capture-long")
    def test_capture_python_long(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_capture_python, data, PATTERN)
        assert result == expected
