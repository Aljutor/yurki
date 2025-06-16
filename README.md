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

### Implementation Details

- **Custom Unicode Reader**: Thread-safe Python string conversion avoiding PyUnicode_AsUTF8AndSize limitations pre-Python 3.13
- **Bump Allocator**: bumpalo-based memory allocation with 256KB initial capacity per thread and automatic reset
- **Parallel Processing**: Rayon-based thread pool

### Benchmark Results (Large Datasets)

**Find Operations**:

- 4 jobs: 2.0s vs Python: 12.5s (6.2x speedup)
- 1 job: 3.2s vs Python: 12.5s (3.9x speedup)

**Match Operations**:

- 4 jobs: 1.8s vs Python: 11.8s (6.6x speedup)
- 1 job: 1.9s vs Python: 11.8s (6.2x speedup)

**Capture Operations**:

- 4 jobs: 5.9s vs Python: 16.9s (2.9x speedup)
- 1 job: 7.9s vs Python: 16.9s (2.1x speedup)

**Replace Operations**:

- 4 jobs: 2.0s vs Python: 3.6s (1.8x speedup)
- 1 job: 2.2s vs Python: 3.6s (1.6x speedup)

**Split Operations**:

- 4 jobs: 5.3s vs Python: 7.0s (1.3x speedup)
- 1 job: 4.7s vs Python: 7.0s (1.5x speedup)

Note: Performance varies by operation type and dataset characteristics. Threading overhead may impact performance on small datasets.

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
