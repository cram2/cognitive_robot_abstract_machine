"""Sentinel value for the EQL-RDR subsystem."""

from __future__ import annotations

from enum import Enum


class _Unset(Enum):
    """Sentinel enum for a missing current/target conclusion.

    A single-member enum yields a hashable, identity-stable sentinel (compared
    with ``is UNSET``) without a hand-rolled singleton class. The type is private;
    only the :data:`UNSET` member is part of the public interface.
    """

    UNSET = "unset"
    """The sole sentinel member, exported module-wide as :data:`UNSET`."""

    def __repr__(self) -> str:
        return "UNSET"

    def __str__(self) -> str:
        return "UNSET"


UNSET: _Unset = _Unset.UNSET
"""Sentinel for "no current/target conclusion was supplied" (e.g. the ask-for-rule path)."""
