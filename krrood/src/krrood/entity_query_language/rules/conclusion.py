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
)
from ..core.variable import Variable


@dataclass(eq=False)
class Conclusion(SymbolicExpression, ABC):
    """
    Base for side-effecting/action clauses that adjust outputs (e.g., Set, Add).
    """

    variable: Selectable
    """
    The variable being affected by the conclusion.
    """
    value: Any
    """
    The value added or set to the variable by the conclusion.
    """

    def __post_init__(self):

        self.variable, self.value = self._update_children_(self.variable, self.value)

        self.value._is_inferred_ = True

        current_parent = SymbolicExpression._current_parent_in_context_stack_()
        if current_parent is None:
            current_parent = self._conditions_root_
        self._parent_ = current_parent
        self._parent_._add_conclusion_(self)

    def _replace_child_field_(
        self, old_child: SymbolicExpression, new_child: SymbolicExpression
    ):
        if self.variable is old_child:
            self.variable = new_child
        elif self.value is old_child:
            self.value = new_child

    @property
    def _name_(self) -> str:
        value_str = (
            self.value._type_.__name__
            if isinstance(self.value, Variable)
            else str(self.value)
        )
        return f"{self.__class__.__name__}({self.variable._var_._name_}, {value_str})"


@dataclass(eq=False)
class Set(Conclusion):
    """Set the value of a variable in the current solution binding."""

    def _evaluate__(
        self,
        sources: Bindings,
    ) -> Iterable[OperationResult]:

        if self.variable._binding_id_ not in sources:
            parent_value = next(iter(self.variable._evaluate_(sources, parent=self)))[
                self.variable._binding_id_
            ]
            sources[self.variable._binding_id_] = parent_value
        sources[self.variable._binding_id_] = next(
            iter(self.value._evaluate_(sources, parent=self))
        )[self.value._binding_id_]
        yield OperationResult(sources, False, self)


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
