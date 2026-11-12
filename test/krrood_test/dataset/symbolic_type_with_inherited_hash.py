"""
A symbolic type that keeps the hash
:class:`krrood.symbolic_math.symbolic_math.SymbolicMathType` gives it, the way the
spatial types built on it do.
"""

from __future__ import annotations

from dataclasses import dataclass

import krrood.symbolic_math.symbolic_math as sm

# %% a symbolic type with the inherited hash


@dataclass(kw_only=True, eq=False, init=False, repr=False)
class SymbolicTypeWithInheritedHash(sm.SymbolicMathType):
    """
    A symbolic type of any shape that neither compares nor hashes in a way of its own.
    """

    def _verify_type(self) -> None:
        """
        Accepts every shape.
        """
