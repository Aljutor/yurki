import re

import pytest

import yurki


PATTERN = r"[,;]"
JOBS = [1, 4]


def regex_split_python(data, pattern):
    compiled_pattern = re.compile(pattern)
    return [compiled_pattern.split(s) for s in data]


def generate_test_data(size):
    """Generate test data and expected results together."""
    data = [f"item1,item2;item3,item{i}" for i in range(size)]
    expected = [["item1", "item2", "item3", f"item{i}"] for i in range(size)]
    return data, expected


class TestSplit:
    @pytest.mark.parametrize("jobs", JOBS)
    @pytest.mark.parametrize("inplace", [False, True])
    def test_split(self, jobs, inplace):
        data, expected = generate_test_data(10)
        result = yurki.regexp.split(data=data, pattern=PATTERN, jobs=jobs, inplace=inplace)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_empty_list(self, jobs):
        assert yurki.regexp.split(data=[], pattern=PATTERN, jobs=jobs, inplace=False) == []

    @pytest.mark.parametrize("jobs", JOBS)
    def test_no_match(self, jobs):
        data = ["abc", "def", "ghi"]
        expected = [["abc"], ["def"], ["ghi"]]
        result = yurki.regexp.split(data=data, pattern=PATTERN, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_split_whitespace(self, jobs):
        data = ["hello world test", "foo  bar", "single"]
        pattern = r"\s+"
        expected = [["hello", "world", "test"], ["foo", "bar"], ["single"]]
        result = yurki.regexp.split(data=data, pattern=pattern, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_split_multiple_delimiters(self, jobs):
        data = ["a,b;c:d", "x|y&z", "single"]
        pattern = r"[,;:|&]"
        expected = [["a", "b", "c", "d"], ["x", "y", "z"], ["single"]]
        result = yurki.regexp.split(data=data, pattern=pattern, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_split_empty_parts(self, jobs):
        data = ["a,,b", "x;;y", ""]
        pattern = r"[,;]"
        expected = [["a", "", "b"], ["x", "", "y"], [""]]
        result = yurki.regexp.split(data=data, pattern=pattern, jobs=jobs, inplace=False)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_unicode(self, jobs):
        data = ["привет мир тест", "你好 世界 测试", "नमस्ते दुनिया परीक्षण"]
        pattern = r"\s+"
        expected = [["привет", "мир", "тест"], ["你好", "世界", "测试"], ["नमस्ते", "दुनिया", "परीक्षण"]]
        result = yurki.regexp.split(data=data, pattern=pattern, jobs=jobs, inplace=False)
        assert result == expected


class TestBenchSplitShort:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(1_000)

    @pytest.mark.benchmark(group="split-short")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_split_rust_short(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.split_by_regexp_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected

    @pytest.mark.benchmark(group="split-short")
    def test_split_python_short(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_split_python, data, PATTERN)
        assert result == expected


class TestBenchSplitMedium:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(100_000)

    @pytest.mark.benchmark(group="split-medium")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_split_rust_medium(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.split_by_regexp_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected

    @pytest.mark.benchmark(group="split-medium")
    def test_split_python_medium(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_split_python, data, PATTERN)
        assert result == expected


class TestBenchSplitLong:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(10_000_000)

    @pytest.mark.benchmark(group="split-long")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_split_rust_long(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.split_by_regexp_string, data, PATTERN, False, jobs, inplace=False)
        assert result == expected

    @pytest.mark.benchmark(group="split-long")
    def test_split_python_long(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(regex_split_python, data, PATTERN)
        assert result == expected
