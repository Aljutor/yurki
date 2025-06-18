
from copy import copy

import pytest

import yurki


JOBS = [1, 4, 16, 64]


def copy_python(data):
    new_data = [copy(item) for item in data]
    return new_data


def generate_test_data(size):
    """Generate test data and expected results together."""
    data = [f"long_long_long_and_unique_string_very_long_such_unique{i}" for i in range(size)]
    expected = [f"long_long_long_and_unique_string_very_long_such_unique{i}" for i in range(size)]
    return data, expected


class TestCopy:
    @pytest.mark.parametrize("jobs", JOBS)
    def test_copy(self, jobs):
        data, expected = generate_test_data(1000)
        result = yurki.test.copy(data=data, jobs=jobs)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_copy_v2(self, jobs):
        data, expected = generate_test_data(1000)
        result = yurki.test.copy_v2(data=data, jobs=jobs)
        assert result == expected

    @pytest.mark.parametrize("jobs", JOBS)
    def test_empty_list(self, jobs):
        assert yurki.test.copy(data=[], jobs=jobs) == []

    @pytest.mark.parametrize("jobs", JOBS)
    def test_empty_list_v2(self, jobs):
        assert yurki.test.copy_v2(data=[], jobs=jobs) == []


class TestBenchCopyShort:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(1_000)

    @pytest.mark.benchmark(group="copy-short")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_copy_rust_short(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.copy_string_list, data, jobs)
        assert len(result) == len(expected)

    @pytest.mark.benchmark(group="copy-short")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_copy_rust_short_v2(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.copy_string_list_v2, data, jobs)
        assert len(result) == len(expected)

    @pytest.mark.benchmark(group="copy-short")
    def test_find_python_short(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(copy_python, data)
        assert len(result) == len(expected)


class TestBenchCopyMedium:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(1_000_000)

    @pytest.mark.benchmark(group="copy-medium")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_copy_rust_medium(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.copy_string_list, data, jobs)
        assert len(result) == len(expected)

    @pytest.mark.benchmark(group="copy-medium")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_copy_rust_medium_v2(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.copy_string_list_v2, data, jobs)
        assert len(result) == len(expected)

    @pytest.mark.benchmark(group="copy-medium")
    def test_find_python_medium(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(copy_python, data)
        assert len(result) == len(expected)


class TestBenchCopyLong:
    @pytest.fixture
    def test_data(self):
        return generate_test_data(10_000_000)

    @pytest.mark.benchmark(group="copy-long")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_copy_rust_long(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.copy_string_list, data, jobs)
        assert len(result) == len(expected)

    @pytest.mark.benchmark(group="copy-long")
    @pytest.mark.parametrize("jobs", JOBS, ids=lambda j: f"jobs={j}")
    def test_copy_rust_long_v2(self, jobs, benchmark, test_data):
        data, expected = test_data
        result = benchmark(yurki.internal.copy_string_list_v2, data, jobs)
        assert len(result) == len(expected)

    @pytest.mark.benchmark(group="copy-long")
    def test_find_python_long(self, benchmark, test_data):
        data, expected = test_data
        result = benchmark(copy_python, data)
        assert len(result) == len(expected)
