import os

import yurki


def __auto_select_jobs(data: list[str]) -> int:
    if len(data) < 1000:
        return 1
    else:
        return os.cpu_count()


def find(
    data: list[str], pattern: str, case: bool = False, jobs: int | None = None, inplace: bool = False
) -> list[str]:
    """Find the first regex match in each string.

    Args:
        data: List of strings to search in
        pattern: Regular expression pattern to search for
        case: Whether to enable case-insensitive matching. Defaults to False
        jobs: Number of parallel jobs to use. Auto-selects based on data size if None
        inplace: Whether to modify the original list. Defaults to False

    Returns:
        List of strings containing the first match found in each input string.
        Empty strings are returned for strings with no matches.

    Examples:
        >>> yurki.regexp.find(['hello world', 'test 123'], r'\\d+')
        ['', '123']

        >>> yurki.regexp.find(['Hello', 'hello'], r'hello', case=True)
        ['Hello', 'hello']
    """
    if jobs is None:
        jobs = __auto_select_jobs(data)

    return yurki.internal.find_regex_in_string(data, pattern, case, jobs, inplace)


def is_match(
    data: list[str], pattern: str, case: bool = False, jobs: int | None = None, inplace: bool = False
) -> list[bool]:
    """Check if each string matches the regex pattern.

    Args:
        data: List of strings to test
        pattern: Regular expression pattern to match against
        case: Whether to enable case-insensitive matching. Defaults to False
        jobs: Number of parallel jobs to use. Auto-selects based on data size if None
        inplace: Whether to modify the original list. Defaults to False

    Returns:
        List of booleans indicating whether each string matches the pattern.

    Examples:
        >>> yurki.regexp.is_match(['test123', 'hello'], r'\\d+')
        [True, False]

        >>> yurki.regexp.is_match(['Hello', 'world'], r'^H', case=True)
        [True, False]
    """
    if jobs is None:
        jobs = __auto_select_jobs(data)

    return yurki.internal.is_match_regex_in_string(data, pattern, case, jobs, inplace)


def capture(
    data: list[str], pattern: str, case: bool = False, jobs: int | None = None, inplace: bool = False
) -> list[list[str]]:
    """Capture regex groups from each string.

    Args:
        data: List of strings to capture from
        pattern: Regular expression pattern with capture groups
        case: Whether to enable case-insensitive matching. Defaults to False
        jobs: Number of parallel jobs to use. Auto-selects based on data size if None
        inplace: Whether to modify the original list. Defaults to False

    Returns:
        List of lists containing captured groups for each string.
        Each inner list contains: [full_match, group1, group2, ...].
        Empty lists are returned for strings with no matches.
        Non-matching groups are represented as empty strings.

    Note:
        Uses Rust's group ordering: group 0 (full match) first, then numbered groups.

    Examples:
        >>> yurki.regexp.capture(['name: John'], r'name: (\\w+)')
        [['name: John', 'John']]

        >>> yurki.regexp.capture(['test 123'], r'(\\w+) (\\d+)')
        [['test 123', 'test', '123']]

        >>> yurki.regexp.capture(['no match'], r'(\\d+)')
        [[]]
    """
    if jobs is None:
        jobs = __auto_select_jobs(data)

    return yurki.internal.capture_regex_in_string(data, pattern, case, jobs, inplace)


def split(
    data: list[str], pattern: str, case: bool = False, jobs: int | None = None, inplace: bool = False
) -> list[list[str]]:
    """Split each string using a regex pattern as delimiter.

    Args:
        data: List of strings to split
        pattern: Regular expression pattern to use as delimiter
        case: Whether to enable case-insensitive matching. Defaults to False
        jobs: Number of parallel jobs to use. Auto-selects based on data size if None
        inplace: Whether to modify the original list. Defaults to False

    Returns:
        List of lists containing the split parts for each string.

    Examples:
        >>> yurki.regexp.split(['a,b;c', 'x,y'], r'[,;]')
        [['a', 'b', 'c'], ['x', 'y']]

        >>> yurki.regexp.split(['hello world test'], r'\\s+')
        [['hello', 'world', 'test']]

        >>> yurki.regexp.split(['no-delimiters'], r',')
        [['no-delimiters']]
    """
    if jobs is None:
        jobs = __auto_select_jobs(data)

    return yurki.internal.split_by_regexp_string(data, pattern, case, jobs, inplace)


def replace(
    data: list[str],
    pattern: str,
    replacement: str,
    count: int = 1,
    case: bool = False,
    jobs: int | None = None,
    inplace: bool = False,
) -> list[str]:
    """Replace regex matches in each string.

    Args:
        data: List of strings to perform replacements on
        pattern: Regular expression pattern to match
        replacement: String to replace matches with. Supports backreferences ($1, $2, etc.)
        count: Number of replacements to make per string:
            - 1 (default): Replace only the first match
            - N > 1: Replace the first N matches
            - 0: Replace all matches
        case: Whether to enable case-insensitive matching. Defaults to False
        jobs: Number of parallel jobs to use. Auto-selects based on data size if None
        inplace: Whether to modify the original list. Defaults to False

    Returns:
        List of strings with replacements applied.

    Note:
        Uses Rust's backreference syntax ($1, $2, etc.) instead of Python's (\\1, \\2, etc.).

    Examples:
        >>> yurki.regexp.replace(['test hello test'], r'test', 'TEST')
        ['TEST hello test']

        >>> yurki.regexp.replace(['test hello test'], r'test', 'TEST', count=0)
        ['TEST hello TEST']

        >>> yurki.regexp.replace(['name: John'], r'name: (\\w+)', r'Hello $1')
        ['Hello John']

        >>> yurki.regexp.replace(['a1b2c3'], r'(\\w)(\\d)', r'$2$1', count=2)
        ['1a2bc3']
    """
    if jobs is None:
        jobs = __auto_select_jobs(data)

    return yurki.internal.replace_regexp_in_string(data, pattern, replacement, count, case, jobs, inplace)


__all__ = ["find", "is_match", "capture", "split", "replace"]
