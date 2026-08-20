"""
The failure both of the executor's outside dependencies report the same way.

git and the GitHub API are the two things a pass depends on and neither of which it
controls. What a caller needs when either refuses is identical, so it is stated once
here and the two concrete failures differ only in how they name the call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExternalCallFailed(RuntimeError):
    """
    Base for a call to git or GitHub that this pass depended on and did not get.

    Both carry the same three things under different names - what was called, the
    status it came back with, and what it said - so they say so once here and differ
    only in how they name the call. Mirrors ``krrood``'s dataclass-exception idiom
    (typed context fields, an abstract message composed by the base) without importing
    it, since this module is deliberately dependency-free.
    """

    status: int
    """The status the call came back with."""

    detail: str
    """
    What the far side said about it.
    """

    @property
    def call(self) -> str:
        """:return: The call that failed, named the way its own caller named it."""
        raise NotImplementedError

    def __str__(self) -> str:
        """:return: The call, its status, and the reason given."""
        return f"{self.call} failed with {self.status}: {self.detail}"
