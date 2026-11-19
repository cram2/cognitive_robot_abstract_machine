"""
A module importing from ``typing`` and annotating with its generic aliases.
"""

from __future__ import annotations

import typing  # reported
from typing import Dict  # reported

from typing_extensions import Optional, Union


def count_entries(table: Dict[str, int]) -> int:  # reported
    """
    A ``Dict`` instead of ``dict``.
    """
    return len(table)


def find_index(name: str) -> Optional[int]:  # reported
    """
    An ``Optional`` instead of ``| None``.
    """
    return None


def parse_number(text: Union[int, str]) -> int:  # reported
    """
    A ``Union`` instead of ``|``.
    """
    return int(text)
