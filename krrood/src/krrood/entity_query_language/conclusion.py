from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache

from typing_extensions import Any, Optional, List, Dict, Iterable

from .enums import RDREdge
from .rxnode import ColorLegend
from .symbolic import (
    SymbolicExpression,
    Variable,
    OperationResult,
    ResultQuantifier,
    Selectable,
)
from .utils import T


@dataclass(eq=False)
class Conclusion(SymbolicExpression[T], ABC):
    """
    Base for side-effecting/action clauses that adjust outputs (e.g., Set, Add).

    :ivar var: The variable being affected by the conclusion.
    :ivar value: The value or expression used by the conclusion.
    """

    var: Selectable[T]
    value: Any
    _child_: Optional[SymbolicExpression[T]] = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()

        self.var, self.value = self._update_children_(self.var, self.value)

        self.value._is_inferred_ = True

        self._node_.weight = RDREdge.Then

        current_parent = SymbolicExpression._current_parent_()
        if current_parent is None:
            current_parent = self._conditions_root_
        self._node_.parent = current_parent._node_
        self._parent_._add_conclusion_(self)

    @property
    @lru_cache(maxsize=None)
    def _all_variable_instances_(self) -> List[Variable]:
        return self.var._all_variable_instances_ + self.value._all_variable_instances_

    @property
    def _name_(self) -> str:
        value_str = (
            self.value._type_.__name__
            if isinstance(self.value, Variable)
            else str(self.value)
        )
        return f"{self.__class__.__name__}({self.var._var_._name_}, {value_str})"

    def _reset_cache_(self) -> None: ...

    @property
    def _plot_color_(self) -> ColorLegend:
        return ColorLegend("Conclusion", "#8cf2ff")


@dataclass(eq=False)
class Set(Conclusion[T]):
    """Set the value of a variable in the current solution binding."""

    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        self._eval_parent_ = parent
        self._yield_when_false_ = False
        if self.var._var_._id_ not in sources:
            parent_value = next(iter(self.var._evaluate__(sources, parent=self)))[
                self.var._var_._id_
            ]
            sources[self.var._var_._id_] = parent_value
        sources[self.var._var_._id_] = next(
            iter(self.value._evaluate__(sources, parent=self))
        )[self.value._id_]
        yield OperationResult(sources, False, self)


@dataclass(eq=False)
class Add(Conclusion[T]):
    """Add a new value to the domain of a variable."""

    def _evaluate__(
        self,
        sources: Optional[Dict[int, Any]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        self._eval_parent_ = parent
        self._yield_when_false_ = False
        v = next(iter(self.value._evaluate__(sources, parent=self)))[self.value._id_]
        sources[self.var._var_._id_] = v
        yield OperationResult(sources, False, self)
