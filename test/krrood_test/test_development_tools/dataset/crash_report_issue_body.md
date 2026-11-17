### What crashed

The parser, while reading a file.

### Script

```python
from pathlib import Path

### a comment that looks like a heading
Path("missing.txt").read_text()
```
