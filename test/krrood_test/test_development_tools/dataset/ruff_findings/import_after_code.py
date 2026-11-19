"""
A module importing after code has run, next to the import blocks that may follow
other imports.
"""

from __future__ import annotations

import os

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

home_directory = os.path.expanduser("~")

import json  # reported

__all__ = ["Path", "json"]
