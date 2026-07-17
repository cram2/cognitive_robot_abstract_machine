#!/usr/bin/env python
import traceback

try:
    from experiments.tool_based_actions.simple_demo import demo_cutting
except Exception:
    traceback.print_exc()
    exit(1)
