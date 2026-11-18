"""
The JSON serializer loads without the libraries only krrood's database and sparse-array
support need.
"""

from __future__ import annotations

import subprocess
import sys
from enum import StrEnum
from pathlib import Path

import pytest

import krrood.adapters.json_serializer


class HeavyLibrary(StrEnum):
    """
    A library the JSON serializer must not need in order to load.
    """

    SQLALCHEMY = "sqlalchemy"
    """
    The database library of the ORM.
    """

    SCIPY = "scipy"
    """
    The scientific library that provides sparse arrays.
    """

    CASADI = "casadi"
    """
    The symbolic mathematics library, which has no Windows distribution.
    """


# %% loading the serializer


@pytest.mark.parametrize("library", list(HeavyLibrary))
def test_serializer_loads_without_a_heavy_library(library: HeavyLibrary) -> None:
    """
    Importing the serializer in a fresh interpreter succeeds when the library is blocked.
    """
    blocker_script = (
        Path(__file__).parent / "dataset" / "import_with_blocked_libraries.py"
    )

    loading = subprocess.run(
        [
            sys.executable,
            "-I",
            str(blocker_script),
            krrood.adapters.json_serializer.__name__,
            library,
        ],
        capture_output=True,
        text=True,
    )

    assert loading.returncode == 0, loading.stderr
