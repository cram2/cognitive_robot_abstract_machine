#!/usr/bin/env python3
"""
Stand-in for build_dashboard.py used by test_refresh_dashboard_sh.py: records its own
invocation arguments to a file (so the test can inspect what refresh_dashboard.sh passed
it, e.g. whether --tracking-url was forwarded) and prints a minimal summary.
"""

import json
import sys
from pathlib import Path

Path("build_dashboard_invocation.json").write_text(json.dumps(sys.argv[1:]))
print(json.dumps({"drift_count": 0}))
