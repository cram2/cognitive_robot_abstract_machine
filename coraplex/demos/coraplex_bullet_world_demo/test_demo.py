#!/usr/bin/env python
import traceback

try:
    import demo
except Exception:
    traceback.print_exc()
    exit(1)
