"""
Module that demonstrates forward reference resolution issues.

This module mimics the pattern found in semantic_digital_twin where:
1. `from __future__ import annotations` makes all annotations forward references
2. Some types are imported only under TYPE_CHECKING
3. When get_type_hints is called, these forward references can't be resolved
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Optional

from typing_extensions import TYPE_CHECKING

from krrood.entity_query_language.predicate import Symbol

if TYPE_CHECKING:
    # These imports are only available to type checkers, not at runtime
    # This mimics the pattern in semantic_digital_twin/semantic_annotations/mixins.py
    from .example_classes import ForwardRefTypeA, ForwardRefTypeB

# Import a type that will be used in method signatures but NOT in TYPE_CHECKING
# This mimics how World is imported in mixins.py
from .example_classes import Position


@dataclass
class MixinWithForwardRef(ABC):
    """
    A mixin class that has a field with a forward reference type.
    The type is imported only under TYPE_CHECKING, so it's not available at runtime.
    """

    type_a_ref: Optional[ForwardRefTypeA] = None

    def method_with_forward_ref_param(self, position: Position) -> None:
        """
        Method with a parameter that uses a forward reference type.
        Due to `from __future__ import annotations`, this is also a forward reference.
        """
        pass


@dataclass
class AnotherMixinWithForwardRef(ABC):
    """Another mixin with a different forward reference."""

    type_b_ref: Optional[ForwardRefTypeB] = None


@dataclass
class ClassWithMultipleForwardRefMixins(
    MixinWithForwardRef, AnotherMixinWithForwardRef, Symbol
):
    """
    A class that inherits from multiple mixins, each with forward reference types.

    This reproduces the issue where:
    1. get_type_hints fails with NameError for ForwardRefTypeA
    2. The code catches the error and adds ForwardRefTypeA to the namespace
    3. get_type_hints is called again but fails with NameError for ForwardRefTypeB
       or Position (which is used in method signatures)
    """

    own_field: str = ""
