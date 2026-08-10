"""
Shared setup for the hook tests: module importability, and the scratch repository they
all run against.

The hooks' Python scripts are single-file scripts, not an installed package - so their
directory is added to ``sys.path`` here rather than requiring an ``__init__.py``/
packaging setup just for tests. Mirrors
``.claude/skills/plan-dashboard/tests/conftest.py``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scratch_repository import ScratchRepository  # noqa: E402


@pytest.fixture
def scratch_repository(tmp_path: Path) -> ScratchRepository:
    """
    An initialized scratch project root and its bare notes remote, with nothing
    committed and no hook scripts installed yet.

    Each test module layers on what its own hook needs - the scripts under test, the
    files they read, and the notes branch contents - so only what genuinely differs
    between them lives in the modules.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The scratch repository.
    """
    return ScratchRepository.create(tmp_path)
