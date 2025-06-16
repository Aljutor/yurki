"""Type stubs for yurki.internal module (Rust implementation)."""

from typing import List

def find_regex_in_string(
    list: List[str],
    pattern: str,
    case: bool = False,
    jobs: int = 1,
    inplace: bool = False,
) -> List[str]:
    """Find first regex match in each string.

    Args:
        list: List of strings to process
        pattern: Regular expression pattern
        case: Case-insensitive matching when True
        jobs: Number of parallel workers
        inplace: Modify original list when True

    Returns:
        List of matched strings (empty string if no match)
    """
    ...

def is_match_regex_in_string(
    list: List[str],
    pattern: str,
    case: bool = False,
    jobs: int = 1,
    inplace: bool = False,
) -> List[bool]:
    """Check if each string matches regex pattern.

    Args:
        list: List of strings to process
        pattern: Regular expression pattern
        case: Case-insensitive matching when True
        jobs: Number of parallel workers
        inplace: Modify original list when True

    Returns:
        List of booleans indicating matches
    """
    ...

def capture_regex_in_string(
    list: List[str],
    pattern: str,
    case: bool = False,
    jobs: int = 1,
    inplace: bool = False,
) -> List[List[str]]:
    """Capture regex groups from each string.

    Args:
        list: List of strings to process
        pattern: Regular expression pattern with capture groups
        case: Case-insensitive matching when True
        jobs: Number of parallel workers
        inplace: Modify original list when True

    Returns:
        List of lists containing [full_match, group1, group2, ...]
    """
    ...

def split_by_regexp_string(
    list: List[str],
    pattern: str,
    case: bool = False,
    jobs: int = 1,
    inplace: bool = False,
) -> List[List[str]]:
    """Split strings by regex delimiter.

    Args:
        list: List of strings to process
        pattern: Regular expression pattern for splitting
        case: Case-insensitive matching when True
        jobs: Number of parallel workers
        inplace: Modify original list when True

    Returns:
        List of lists containing split parts
    """
    ...

def replace_regexp_in_string(
    list: List[str],
    pattern: str,
    replacement: str,
    count: int = 1,
    case: bool = False,
    jobs: int = 1,
    inplace: bool = False,
) -> List[str]:
    """Replace regex matches in strings.

    Args:
        list: List of strings to process
        pattern: Regular expression pattern
        replacement: Replacement string (supports backreferences $1, $2, etc.)
        count: Maximum number of replacements per string (0 for all)
        case: Case-insensitive matching when True
        jobs: Number of parallel workers
        inplace: Modify original list when True

    Returns:
        List of strings with replacements applied
    """
    ...
