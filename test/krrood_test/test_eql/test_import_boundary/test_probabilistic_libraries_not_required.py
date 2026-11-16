"""
The entity query language loads without the probabilistic libraries installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

import krrood.entity_query_language
from krrood.entity_query_language.enums import ProbabilisticLibrary
from krrood.entity_query_language.exceptions import ProbabilisticLibrariesNotInstalled
from krrood.entity_query_language.factories import probability_of, variable

# %% importing without them


def test_every_entity_query_language_module_imports_without_probabilistic_libraries():
    """
    A fresh interpreter, since this one has already loaded the libraries.
    """
    import_script = (
        Path(__file__).parent / "dataset" / "import_package_without_libraries.py"
    )
    report = subprocess.run(
        [
            sys.executable,
            str(import_script),
            krrood.entity_query_language.__name__,
            *ProbabilisticLibrary,
        ],
        capture_output=True,
        text=True,
    )

    assert report.returncode == 0, report.stderr


# %% building probabilistic queries without them


def test_building_a_probabilistic_query_without_libraries_raises_with_a_correction(
    monkeypatch,
):
    """
    The factories name the missing libraries and how to fix it instead of failing on an
    import.
    """
    monkeypatch.setattr(
        ProbabilisticLibrary, "missing", classmethod(lambda cls: tuple(cls))
    )

    with pytest.raises(ProbabilisticLibrariesNotInstalled) as raised:
        probability_of(variable(int, domain=[1]) > 0)

    assert raised.value.missing_libraries == tuple(ProbabilisticLibrary)
