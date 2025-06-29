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

### Implementation notes

- **Custom Python types**: `yurki.List` (immutable) and `yurki.String` match the Python 3.12 object layout but use a Rust-side allocator, avoiding the CPython heap.  
- **SIMD Unicode reader**: vectorised path that converts Python text to Rust `&str`.  
- **Bump allocator**: thread-local arena for short-lived allocations; resets automatically, minimising locking and fragmentation.  
- **Parallel processing**: Rayon work pool distributes work across available cores.

### Benchmark Results (Large Datasets)

**Find Operations**:

- 4 jobs: 0.72s vs Python: 12.49s (**17.4x speedup**)
- 1 job: 2.30s vs Python: 12.49s (5.4x speedup)

**Match Operations**:

- 4 jobs: 0.33s vs Python: 11.67s (**35.2x speedup**)
- 1 job: 1.27s vs Python: 11.67s (9.2x speedup)

**Capture Operations**:

- 4 jobs: 2.83s vs Python: 16.97s (**6.0x speedup**)
- 1 job: 6.58s vs Python: 16.97s (2.6x speedup)

**Replace Operations**:

- 4 jobs: 0.64s vs Python: 3.73s (**5.9x speedup**)
- 1 job: 1.76s vs Python: 3.73s (2.1x speedup)

**Split Operations**:

- 4 jobs: 1.33s vs Python: 7.15s (**5.4x speedup**)
- 1 job: 3.34s vs Python: 7.15s (2.1x speedup)

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
