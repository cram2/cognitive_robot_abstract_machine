"""
Conclusion rules for the Entity Query Language.

This module defines side-effecting clauses that adjust outputs or bindings (for example, Add) during query evaluation.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from functools import cached_property

from typing_extensions import Any, List, Iterable

from ..core.base_expressions import (
    Bindings,
    OperationResult,
    SymbolicExpression,
    Selectable,
    BinaryExpression,
)
from ..core.variable import Variable


@dataclass(eq=False)
class Conclusion(BinaryExpression, ABC):
    """
    Base for side-effecting/action clauses that adjust outputs (e.g., Set, Add).
    """

    left: Selectable
    """
    The variable being affected by the conclusion.
    """
    right: Any
    """
    The value added or set to the variable by the conclusion.
    """

    def __post_init__(self):

        super().__post_init__()

        current_parent = SymbolicExpression._current_parent_in_context_stack_()
        if current_parent is None:
            current_parent = self._conditions_root_
        self._parent_ = current_parent
        self._parent_._conclusions_.add(self)

    @property
    def variable(self) -> Selectable:
        return self.left

    @property
    def value(self) -> Any:
        return self.right

    @property
    def _name_(self) -> str:
        value_str = (
            self.value._type_.__name__
            if isinstance(self.value, Variable)
            else str(self.value)
        )
        return f"{self.__class__.__name__}({self.variable._var_._name_}, {value_str})"


@dataclass(eq=False)
class Add(Conclusion):
    """Add a new value to the domain of a variable."""

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:

        v = next(self.value._evaluate_(sources, parent=self)).value
        sources[self.variable._binding_id_] = v
        yield OperationResult(sources, False, self)
