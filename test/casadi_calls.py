"""
Records the calls code makes into CasADi, for tests stating that something reaches its
numbers without building or evaluating anything symbolic.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import FrameType

import casadi
from typing_extensions import Any, Optional, Self

CASADI_PACKAGE = Path(casadi.__file__).parent
"""
Where the CasADi package is installed; a call into a file below it is a call into
CasADi.
"""

REPOSITORY_ROOT = Path(__file__).parents[1]
"""
The directory the callers a recorder names are stated relative to.
"""


class ProfiledEvent(StrEnum):
    """
    The kinds of event :func:`sys.setprofile` reports that can be a call into CasADi.
    """

    PYTHON_CALL = "call"
    """
    A call to a function written in Python, such as CasADi's generated wrappers.
    """

    C_CALL = "c_call"
    """
    A call to a function written in C, such as CasADi's compiled extension.
    """


class SymbolicMachinery(StrEnum):
    """
    Parts of the workspace that exist to build symbolic values, so a call into CasADi is
    attributed to whatever called into them rather than to them.
    """

    SYMBOLIC_MATH = "krrood/src/krrood/symbolic_math/"
    """
    krrood's wrappers around CasADi expressions.
    """

    SPATIAL_TYPES = (
        "semantic_digital_twin/src/semantic_digital_twin/spatial_types/spatial_types.py"
    )
    """
    The symbolic points, vectors, rotations and transforms of the digital twin.
    """


@dataclass
class CasadiCalls:
    """
    The calls into CasADi made on the current thread while this recorder is entered,
    counted by the line of the repository that led to each of them.

    Entering the recorder again adds to what it has recorded so far.
    """

    calls_by_caller: Counter[str] = field(default_factory=Counter)
    """
    How many calls into CasADi each repository line outside the symbolic machinery led
    to, named as ``path:line function``.
    """

    def __enter__(self) -> Self:
        sys.setprofile(self._record)
        return self

    def __exit__(self, *exception_info: Any) -> None:
        sys.setprofile(None)

    def _record(self, frame: FrameType, event: str, argument: Any) -> None:
        if not self._is_a_call_into_casadi(frame, event, argument):
            return
        self.calls_by_caller[self._caller_of(frame)] += 1

    @staticmethod
    def _is_a_call_into_casadi(frame: FrameType, event: str, argument: Any) -> bool:
        if event == ProfiledEvent.PYTHON_CALL:
            return CasadiCalls._is_casadi(frame)
        if event == ProfiledEvent.C_CALL:
            return str(argument.__module__).split(".")[-1].lstrip("_") == "casadi"
        return False

    @staticmethod
    def _is_casadi(frame: FrameType) -> bool:
        return Path(frame.f_code.co_filename).is_relative_to(CASADI_PACKAGE)

    @staticmethod
    def _is_symbolic_machinery(frame: FrameType) -> bool:
        return any(
            part in frame.f_code.co_filename for part in SymbolicMachinery
        ) or CasadiCalls._is_casadi(frame)

    @staticmethod
    def _is_in_the_repository(frame: FrameType) -> bool:
        return Path(frame.f_code.co_filename).is_relative_to(REPOSITORY_ROOT)

    @staticmethod
    def _caller_of(frame: FrameType) -> str:
        caller: Optional[FrameType] = frame
        while caller is not None and (
            CasadiCalls._is_symbolic_machinery(caller)
            or not CasadiCalls._is_in_the_repository(caller)
        ):
            caller = caller.f_back
        if caller is None:
            return "unknown caller"
        path = Path(caller.f_code.co_filename).relative_to(REPOSITORY_ROOT)
        return f"{path}:{caller.f_lineno} {caller.f_code.co_name}"
