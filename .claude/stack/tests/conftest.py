"""
Makes ``stack`` importable as a plain module, and the hooks-test suite's
``ScratchRepository`` importable for the config-layering tests - both are
single-file scripts run directly, not installed packages, so their
directories are added to ``sys.path`` here rather than requiring an
``__init__.py``/packaging setup just for tests. Mirrors
``.claude/skills/plan-dashboard/tests/conftest.py`` and
``.claude/hooks/tests/conftest.py``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "hooks"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "hooks" / "tests"))

import pytest  # noqa: E402
from scratch_repository import ScratchRepository  # noqa: E402


@pytest.fixture
def scratch_repository(tmp_path: Path) -> ScratchRepository:
    """
    An initialized scratch project root and its bare notes remote, with nothing
    committed and no personal-notes branch published yet.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The scratch repository.
    """
    return ScratchRepository.create(tmp_path)
