#!/usr/bin/env python
import os
import traceback

# Keep the wrapper headless: no viewer is spawned unless the caller opts in.
os.environ.setdefault("CORAPLEX_VISUALIZATION", "none")

try:
    import demo
except Exception:
    traceback.print_exc()
    exit(1)
