# Yurki

High-performance regex operations for Python using Rust backends with parallel processing support.

**⚠️ Warning: This is a raw development project and may be unstable. Use at your own risk.**

## Requirements

- Python 3.12+
- Rust toolchain
- uv package manager

## Installation

```bash
git clone https://github.com/Aljutor/yurki.git
cd yurki
uv sync
maturin develop
```

For performance testing, use release build:
```bash
maturin develop --release
```

## Performance

Yurki significantly outperforms Python's standard regex on large datasets:

- **Find operations**: 5x faster (2.5s vs 12.7s on 10M strings)
- **Match operations**: 5.6x faster (2.1s vs 12.0s on 10M strings)  
- **Capture operations**: 2.5x faster (6.8s vs 17.2s on 10M strings)
- **Replace operations**: 1.6x faster (2.3s vs 3.6s on 10M strings)

Performance scales with parallel processing using the `jobs` parameter.

## Usage & API

```python
import yurki.regexp as regexp

data = ['hello world', 'test 123', 'no match here']

# Find first regex match in each string
# Returns list of matched strings (empty string if no match)
regexp.find(data, pattern, case=False, jobs=1, inplace=False)
regexp.find(data, r'\d+')  # ['', '123', '']

# Check if each string matches pattern  
# Returns list of booleans
regexp.is_match(data, pattern, case=False, jobs=1, inplace=False)
regexp.is_match(data, r'\d+')  # [False, True, False]

# Capture regex groups
# Returns list of lists: [full_match, group1, group2, ...]
regexp.capture(data, pattern, case=False, jobs=1, inplace=False)
regexp.capture(data, r'(\w+) (\d+)')  # [[], ['test 123', 'test', '123'], []]

# Split strings by regex delimiter
# Returns list of lists
regexp.split(data, pattern, case=False, jobs=1, inplace=False)
regexp.split(['a,b;c', 'x,y'], r'[,;]')  # [['a', 'b', 'c'], ['x', 'y']]

# Replace regex matches  
# Use count=0 for all matches. Supports backreferences ($1, $2)
regexp.replace(data, pattern, replacement, count=1, case=False, jobs=1, inplace=False)
regexp.replace(data, r'\d+', 'NUM')  # ['hello world', 'test NUM', 'no match here']

# Parallel processing for large datasets
regexp.find(large_data, pattern, jobs=4)

# In-place operations for memory efficiency
regexp.replace(data, pattern, replacement, inplace=True)
```

**Parameters:**
- `data`: List of strings to process
- `pattern`: Regex pattern string  
- `case`: Case-insensitive matching when True
- `jobs`: Number of parallel workers
- `inplace`: Modify original list when True

## License

MIT
