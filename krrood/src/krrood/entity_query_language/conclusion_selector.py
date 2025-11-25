from __future__ import annotations

import typing
from abc import ABC
from dataclasses import dataclass, field
from functools import lru_cache
from typing_extensions import Dict, Optional, Iterable

from .cache_data import SeenSet
from .conclusion import Conclusion
from .hashed_data import HashedIterable, HashedValue
from .rxnode import ColorLegend
from .symbolic import (
    SymbolicExpression,
    ElseIf,
    Union as EQLUnion,
    Literal,
    OperationResult,
    LogicalBinaryOperator,
)


@dataclass(eq=False)
class ConclusionSelector(LogicalBinaryOperator, ABC):
    """
    Base class for logical operators that may carry and select conclusions.

    Tracks whether certain conclusion-combinations were already produced so
    they are not duplicated across truth branches.
    """

    concluded_before: Dict[bool, SeenSet] = field(
        default_factory=lambda: {True: SeenSet(), False: SeenSet()}, init=False
    )

    def update_conclusion(
        self, output: OperationResult, conclusions: typing.Set[Conclusion]
    ) -> None:
        """
        Update conclusions if this combination hasn't been seen before.

        Uses canonical tuple keys for stable deduplication.
        """
        if not conclusions:
            return
        required_vars = HashedIterable()
        for conclusion in conclusions:
            vars_ = conclusion._unique_variables_.filter(
                lambda v: not isinstance(v.value, Literal)
            )
            required_vars.update(vars_)
        required_output = {
            k: v for k, v in output.bindings.items() if k in required_vars
        }

        if not self.concluded_before[not self._is_false_].check(required_output):
            self._conclusion_.update(conclusions)
            self.concluded_before[not self._is_false_].add(required_output)

    @property
    def _plot_color_(self) -> ColorLegend:
        return ColorLegend("ConclusionSelector", "#eded18")


@dataclass(eq=False)
class ExceptIf(ConclusionSelector):
    """
    Conditional branch that yields left unless the right side produces values.

    This encodes an "except if" behavior: when the right condition matches,
    the left branch's conclusions/outputs are excluded; otherwise, left flows through.
    """

    @lru_cache(maxsize=None)
    def _projection_(self, when_true: Optional[bool] = True) -> HashedIterable[int]:
        """
        Return the projection for ExceptIf operators.

        Includes variables from both branches and their conclusions based on truth values.
        """
        projection = HashedIterable()

        # When true, we need right's variables to check the exception condition
        if when_true:
            projection.update(self.right._unique_variables_)

        # Include conclusions from both branches
        for conclusion in self.left._conclusion_.union(self.right._conclusion_):
            projection.update(conclusion._unique_variables_)

        if self._parent_:
            projection.update(self._parent_._projection_(when_true))

        for conclusion in self._conclusion_:
            projection.update(conclusion._unique_variables_)

        return projection

    def _evaluate__(
        self,
        sources: Optional[Dict[int, HashedValue]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        """
        Evaluate the ExceptIf condition and yield the results.
        """
        self._eval_parent_ = parent
        # init an empty source if none is provided
        sources = sources or HashedIterable()

        # constrain left values by available sources
        left_values = self.left._evaluate__(sources, parent=self)
        for left_value in left_values:

            self._is_false_ = left_value.is_false
            if self._is_false_:
                yield left_value
                continue

            right_yielded = False
            for right_value in self.right._evaluate__(left_value.bindings, parent=self):
                if right_value.is_false:
                    continue
                right_yielded = True
                yield from self.yield_and_update_conclusion(
                    right_value, self.right._conclusion_
                )
            if not right_yielded:
                yield from self.yield_and_update_conclusion(
                    left_value, self.left._conclusion_
                )

    def yield_and_update_conclusion(
        self, result: OperationResult, conclusion: typing.Set[Conclusion]
    ) -> Iterable[OperationResult]:
        self.update_conclusion(result, conclusion)
        yield OperationResult(result.bindings, self._is_false_, self)
        self._conclusion_.clear()


@dataclass(eq=False)
class Alternative(ElseIf, ConclusionSelector):
    """
    A conditional branch that behaves like an "else if" clause where the left branch
    is selected if it is true, otherwise the right branch is selected if it is true else
    none of the branches are selected.

    Uses both variable-based deduplication (from base class via projection) and
    conclusion-based deduplication (via update_conclusion).
    """

    def _evaluate__(
        self,
        sources: Optional[Dict[int, HashedValue]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        outputs = super()._evaluate__(sources, parent=parent)
        for output in outputs:
            # Only yield if conclusions were successfully added (not duplicates)
            if not self.left._is_false_:
                self.update_conclusion(output, self.left._conclusion_)
            elif not self.right._is_false_:
                self.update_conclusion(output, self.right._conclusion_)
            yield OperationResult(output.bindings, self._is_false_, self)
            self._conclusion_.clear()


@dataclass(eq=False)
class Next(EQLUnion, ConclusionSelector):
    """
    A Union conclusion selector that always evaluates the left and right branches and combines their results.
    """

    def _evaluate__(
        self,
        sources: Optional[Dict[int, HashedValue]] = None,
        parent: Optional[SymbolicExpression] = None,
    ) -> Iterable[OperationResult]:
        outputs = super()._evaluate__(sources, parent=parent)
        for output in outputs:
            if self.left_evaluated:
                self.update_conclusion(output, self.left._conclusion_)
            if self.right_evaluated:
                self.update_conclusion(output, self.right._conclusion_)
            yield OperationResult(output.bindings, self._is_false_, self)
            self._conclusion_.clear()
