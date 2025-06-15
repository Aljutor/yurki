import yurki


def find(data: list[str], pattern: str, case: bool = False, jobs: int = 1, inplace: bool = False) -> list[str]:
    return yurki.internal.find_regex_in_string(data, pattern, case, jobs, inplace)


def is_match(data: list[str], pattern: str, case: bool = False, jobs: int = 1, inplace: bool = False) -> list[bool]:
    return yurki.internal.is_match_regex_in_string(data, pattern, case, jobs, inplace)


__all__ = ["find", "is_match"]
