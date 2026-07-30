"""
Static type-checks for the ``an()``/``a()``/``the()`` overloads.

Existing tests exercise these factories at runtime, but the overloads themselves are a
purely static contract (:py:func:`typing.overload` bodies never execute) -- nothing
catches a change that silently makes mypy infer the wrong static type while runtime
behaviour stays correct. Running ``mypy`` against a fixture module full of
``assert_type`` calls closes that gap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from mypy import api as mypy_api

import krrood

from . import quantifier_overloads_fixture


@dataclass
class MypyCheckResult:
    """
    The outcome of running ``mypy`` against a single source file.
    """

    stdout: str
    """
    ``mypy``'s normal-report output (errors/notes, one per line).
    """

    stderr: str
    """
    ``mypy``'s own diagnostic output (e.g. internal errors), separate from ``stdout``.
    """

    exit_status: int
    """
    ``mypy``'s process exit status: ``0`` means no errors were found.
    """

    @property
    def succeeded(self) -> bool:
        """
        :return: Whether ``mypy`` found no errors.
        """
        return self.exit_status == 0


def _run_mypy_on_fixture(cache_dir: Path) -> MypyCheckResult:
    """
    Type-check the fixture module with ``mypy``, resolving ``krrood`` as a first-party
    package.

    ``krrood`` ships no ``py.typed`` marker, so mypy resolves it through its editable
    install and skips analysing it ("missing library stubs or py.typed marker") by
    default. Pointing MYPYPATH at the source root (derived from the already-imported
    package's own file, not a hardcoded relative path) makes mypy treat it that way
    anyway, exactly as it would once py.typed is added.

    :param cache_dir: A scratch directory for mypy's incremental cache, so repeated runs
        never write ``.mypy_cache`` into the repository.
    :return: The check's outcome.
    """
    fixture_path = Path(quantifier_overloads_fixture.__file__)
    krrood_src = Path(krrood.__file__).resolve().parent.parent

    previous_mypypath = os.environ.get("MYPYPATH")
    os.environ["MYPYPATH"] = str(krrood_src)
    try:
        stdout, stderr, exit_status = mypy_api.run(
            ["--cache-dir", str(cache_dir), str(fixture_path)]
        )
    finally:
        if previous_mypypath is None:
            os.environ.pop("MYPYPATH", None)
        else:
            os.environ["MYPYPATH"] = previous_mypypath
    return MypyCheckResult(stdout=stdout, stderr=stderr, exit_status=exit_status)


def test_quantifier_overloads_type_check(tmp_path) -> None:
    """
    Every ``assert_type`` call in the fixture module must hold under ``mypy``.
    """
    result = _run_mypy_on_fixture(cache_dir=tmp_path / "mypy_cache")
    assert result.succeeded, f"mypy reported problems:\n{result.stdout}{result.stderr}"
