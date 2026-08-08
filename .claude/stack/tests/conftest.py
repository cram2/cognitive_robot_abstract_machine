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
from stack import BOARD_PATH  # noqa: E402


@pytest.fixture(autouse=True)
def board_snapshot_set_aside() -> None:
    """
    Hide any board snapshot this checkout happens to be carrying, for every test.

    ``board.json`` lives beside ``stack.py`` rather than in the scratch repository a
    test runs in, so a developer who has run a maintenance pass has one - and the tests
    that assert on a *missing* board would fail for a reason that has nothing to do with
    them. Setting it aside makes the suite independent of whether a pass has been run
    here, and restores it afterwards so running the tests never costs somebody their
    snapshot.
    """
    if not BOARD_PATH.exists():
        yield
        return
    set_aside = BOARD_PATH.with_suffix(".json.set-aside-for-tests")
    BOARD_PATH.rename(set_aside)
    yield
    set_aside.rename(BOARD_PATH)


@pytest.fixture
def scratch_repository(tmp_path: Path) -> ScratchRepository:
    """
    An initialized scratch project root and its bare notes remote, with nothing
    committed and no personal-notes branch published yet.

    :param tmp_path: pytest's per-test temporary directory.
    :return: The scratch repository.
    """
    return ScratchRepository.create(tmp_path)
