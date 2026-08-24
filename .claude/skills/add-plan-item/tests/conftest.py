"""
Makes ``check_scope_overlap.py`` and the shared scratch-repository helper importable as
plain modules.

``check_scope_overlap.py`` is a single-file script run via ``python3
check_scope_overlap.py ...``, not an installed package, so its directory is added to
``sys.path`` here rather than requiring an ``__init__.py``/packaging setup just for
tests - the same approach ``plan-dashboard/tests/conftest.py`` takes. The hooks
directories come along too so ``scratch_repository`` (and the ``plan_manifest_tools``
it imports) resolve, rather than a second scratch-repository helper being written.
"""

import sys
from pathlib import Path

SKILL_DIRECTORY = Path(__file__).parent.parent
PROJECT_ROOT = SKILL_DIRECTORY.parent.parent.parent
HOOKS_DIRECTORY = PROJECT_ROOT / ".claude" / "hooks"

for importable_directory in (
    SKILL_DIRECTORY,
    HOOKS_DIRECTORY,
    HOOKS_DIRECTORY / "tests",
):
    sys.path.insert(0, str(importable_directory))
