"""
Classes that use isolated forward reference types under TYPE_CHECKING.

This module is specifically designed to reproduce the bug where multiple
forward references can't be resolved iteratively because they're not
in sys.modules when the resolution happens.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Optional

from typing_extensions import TYPE_CHECKING

from krrood.entity_query_language.predicate import Symbol

if TYPE_CHECKING:
    # These imports are only available to type checkers, not at runtime
    # They won't be in sys.modules when the test runs
    from .isolated_forward_ref_types import IsolatedTypeAlpha, IsolatedTypeBeta


@dataclass
class IsolatedMixinAlpha(ABC):
    """A mixin with an isolated forward reference type."""

    alpha_ref: Optional[IsolatedTypeAlpha] = None


@dataclass
class IsolatedMixinBeta(ABC):
    """Another mixin with a different isolated forward reference type."""

    beta_ref: Optional[IsolatedTypeBeta] = None


@dataclass
class IsolatedClassWithMultipleMixins(IsolatedMixinAlpha, IsolatedMixinBeta, Symbol):
    """
    A class that inherits from multiple mixins with isolated forward references.

    This reproduces the bug where:
    1. get_type_hints fails with NameError for IsolatedTypeAlpha
    2. The code catches the error and searches for IsolatedTypeAlpha
    3. If IsolatedTypeAlpha is NOT in sys.modules, manually_search_for_class_name fails
    4. Even if Alpha is found, the second get_type_hints call may fail for IsolatedTypeBeta
    """

    own_value: str = ""
