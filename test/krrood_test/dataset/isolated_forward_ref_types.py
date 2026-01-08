"""
Isolated forward reference types that are NOT imported anywhere else.

These classes are specifically designed to NOT be in sys.modules when
the forward reference resolution test runs, to reproduce the bug where
multiple forward references can't be resolved iteratively.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.predicate import Symbol


@dataclass
class IsolatedTypeAlpha(Symbol):
    """An isolated type used for forward reference testing."""

    alpha_value: str = ""


@dataclass
class IsolatedTypeBeta(Symbol):
    """Another isolated type used for forward reference testing."""

    beta_value: int = 0
