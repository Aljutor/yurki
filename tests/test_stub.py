import yurki


def test_to_uppercase():
    size = 1000000
    data = ["hi" for _ in range(size)]
    expected = ["HI" for _ in range(size)]
    assert yurki.to_uppercase(data) == expected
