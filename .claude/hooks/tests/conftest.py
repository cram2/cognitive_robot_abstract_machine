"""
Makes the hooks' Python scripts importable as plain modules.

They are single-file scripts, not an installed package - so their directory is added
to ``sys.path`` here rather than requiring an ``__init__.py``/packaging setup just for
tests. Mirrors ``.claude/skills/plan-dashboard/tests/conftest.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
