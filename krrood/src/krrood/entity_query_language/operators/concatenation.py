"""
Concatenation utilities for the Entity Query Language.

This module defines the Concatenation operator that merges values from multiple selectable expressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Iterable

from ..core.base_expressions import Bindings, OperationResult, Selectable
from ..operators.set_operations import Union
from ..utils import T
from ..core.mapped_variable import CanBehaveLikeAVariable


@dataclass(eq=False, repr=False)
class Concatenation(Union, CanBehaveLikeAVariable[T]):
    """
    Concatenation of two or more variables.
    """

    _operation_children_: Tuple[Selectable, ...] = field(default_factory=tuple)
    """
    The children of the concatenate operation. They must be selectables.
    """

    def __post_init__(self):
        if not all(
            isinstance(child, Selectable) for child in self._operation_children_
        ):
            raise ValueError(
                f"All children of Concatenate must be Selectable instances."
            )
        super().__post_init__()
        self._var_ = self

    def _evaluate__(self, sources: Bindings) -> Iterable[OperationResult]:
        yield from (
            result.update({self._binding_id_: result.previous_operation_result.value})
            for result in super()._evaluate__(sources)
        )

    @property
    def _variables_(self) -> Tuple[Selectable[T], ...]:
        """
        The variables to concatenate.
        """
        return self._operation_children_
