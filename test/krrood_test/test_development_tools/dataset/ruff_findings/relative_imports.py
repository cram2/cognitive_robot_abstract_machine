"""
A module importing modules of its own package relatively.
"""

from __future__ import annotations

from . import assertion  # reported
from .mutable_default import collect  # reported

__all__ = ["assertion", "collect"]
