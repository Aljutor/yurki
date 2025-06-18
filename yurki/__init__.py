"""Yurki - High-performance string processing library.

Yurki provides fast, parallel string processing operations implemented in Rust
with Python bindings. All functions support parallel processing and can work
in-place for memory efficiency.

Modules:
    regexp: Regular expression operations (find, match, capture, split, replace)
    internal: Low-level Rust functions (for advanced users)

Examples:
    >>> import yurki
    >>> data = ['hello world', 'test 123']
    >>> yurki.regexp.find(data, r'\\d+')
    ['', '123']
    >>> yurki.regexp.replace(data, r'hello', 'hi')
    ['hi world', 'test 123']
"""

import yurki.test as test
import yurki.regexp as regexp
from .yurki import internal


__all__ = ["regexp", "internal", "test"]
