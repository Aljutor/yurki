import re
import yurki
import pytest

PATTERN = r"(hi_how_are_you)|(hello)|(привет\d+)"


def make_data(size=100000, long=False):
    if long:
        data = [
            f"making_this_string_" f"long_enough_to_test_" f"hi_привет{i}"
            for i in range(size)
        ]
    else:
        data = [f"hi_привет{i}|" for i in range(size)]

    expected = [f"привет{i}" for i in range(size)]
    return data, expected


test_data = [
    (1, False),
    (1, True),
    (4, False),
    (4, True),
]


class TestFind:
    @pytest.mark.parametrize("jobs, inplace", test_data)
    def test_find_rust(self, jobs, inplace):
        data, expected = make_data(1000, long=False)
        result = yurki.find_in_string(data, PATTERN, jobs, inplace)
        assert result == expected


class TestBenchFind:
    @pytest.mark.parametrize("jobs, inplace", test_data)
    def test_find_rust(self, jobs, inplace, benchmark):
        data, expected = make_data(long=False)
        result = benchmark(yurki.find_in_string, data, PATTERN, jobs, inplace)
        assert result == expected

    def test_find_python(self, benchmark):
        pattern = re.compile(PATTERN)
        data, expected = make_data(long=False)

        def find(data):
            return [pattern.search(s).group(0) for s in data]

        result = benchmark(find, data)
        assert result == expected


class TestBenchFindLong:
    @pytest.mark.parametrize("jobs, inplace", test_data)
    def test_find_rust(self, jobs, inplace, benchmark):
        data, expected = make_data(long=True)
        result = benchmark(yurki.find_in_string, data, PATTERN, jobs, inplace)
        assert result == expected

    def test_find_python(self, benchmark):
        pattern = re.compile(PATTERN)
        data, expected = make_data(long=True)

        def find(data):
            return [pattern.search(s).group(0) for s in data]

        result = benchmark(find, data)
        assert result == expected
