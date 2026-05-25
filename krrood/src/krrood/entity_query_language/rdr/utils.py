from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, eq=False, repr=False)
class _Unset:
    """
    Class for UNSET Sentinel for "no current/target conclusion was supplied" (useful, for example, for ask-for-rule path).
    """
    def __repr__(self) -> str:
        return "UNSET"
    def __eq__(self, other):
        return isinstance(other, _Unset)
    def __hash__(self):
        return hash(type(self))

#: Sentinel for "no current/target conclusion was supplied" (useful, for example, for ask-for-rule path).
UNSET: _Unset = _Unset()
