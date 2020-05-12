import re
import yurki

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


# Benchmarks
def test_find_rust_4_bench_long(benchmark):
    data, expected = make_data(long=True)
    result = benchmark(yurki.find_in_string, data, PATTERN, 4)
    assert result == expected


def test_find_python_bench_long(benchmark):
    pattern = re.compile(PATTERN)

    def find(data):
        return [pattern.search(s).group(0) for s in data]

    data, expected = make_data(long=True)
    result = benchmark(find, data)
    assert result == expected
