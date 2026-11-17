"""
Building a verbalization clause is declarative: the inflected words it decides are
realized later, so building one loads none of the libraries realization depends on.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from ...test_core_boundary.test_core_boundary import packages_imported_while_running


# %% realization libraries


class RealizationLibrary(StrEnum):
    """
    The top-level packages only realizing a fragment tree as text may load.
    """

    INFLECT = "inflect"
    LEMMINFLECT = "lemminflect"
    TYPEGUARD = "typeguard"
    JINJA2 = "jinja2"
    IPYTHON = "IPython"


# %% building a clause


def test_building_clauses_loads_no_realization_library() -> None:
    """
    Every vocabulary path that decides an inflected word - a verb read off a class name,
    a set summarised by count, an indefinite article, an indexed collection hop - builds
    its clause without loading a realization library.
    """
    source = (Path(__file__).parent / "dataset" / "clause_building.py").read_text()

    loaded = {imported.package for imported in packages_imported_while_running(source)}

    assert loaded & set(RealizationLibrary) == set()
