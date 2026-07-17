"""
Composable result-stream transformers for the Entity Query Language.

A query produces a stream of result rows from its cartesian product; ordering and
quantification are expressed as :class:`ResultTransformer` stages applied to that
stream. A query owns an ordered list of transformers and runs them in sequence, so each
stage stays single-responsibility and the query node itself stays thin and inspectable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Any, Callable, Iterator, Optional, TYPE_CHECKING

from krrood.entity_query_language.core.base_expressions import (
    OperationResult,
    Selectable,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.query.quantifiers import (
        ResultQuantificationConstraint,
    )


@dataclass
class ResultTransformer(ABC):
    """
    A single stage in a query's result pipeline that maps a stream of result rows to
    another stream.
    """

    @abstractmethod
    def transform(
        self, results: Iterator[OperationResult]
    ) -> Iterator[OperationResult]:
        """
        :param results: The incoming result rows.
        :return: The transformed result rows.
        """


@dataclass
class Ordering(ResultTransformer):
    """
    Sorts a query's results by the value of an ordering variable.
    """

    variable: Selectable
    """
    The variable whose value the results are ordered by.
    """
    descending: bool = False
    """
    Whether to order the results in descending order.
    """

    key: Optional[Callable] = None
    """
    An optional function extracting the comparison key from the ordering variable's
    value.
    """

    def transform(
        self, results: Iterator[OperationResult]
    ) -> Iterator[OperationResult]:
        yield from sorted(results, key=self._sort_key_, reverse=self.descending)

    def _sort_key_(self, result: OperationResult) -> Any:
        """
        :param result: A result row to order.
        :return: The comparison key for *result*, resolving the ordering variable against the row's
            bindings (evaluating it when not already bound) and applying the optional key function.
        """
        variable_id = self.variable._id_
        if variable_id not in result.all_bindings:
            ordering_value = next(
                self.variable._evaluate_(OperationResult(result.all_bindings))
            ).value
        else:
            ordering_value = result.all_bindings[variable_id]
        return self.key(ordering_value) if self.key else ordering_value


@dataclass
class Quantification(ResultTransformer):
    """
    Enforces a query's result-count constraint (``an``/``the`` and custom
    quantifications), passing rows through while asserting the constraint is satisfied.
    """

    quantifier_type: type
    """
    The quantifier kind (``An`` / ``The``) this stage enforces, kept so the pipeline
    stays inspectable.
    """

    constraint: Optional[ResultQuantificationConstraint] = None
    """
    The result-count constraint to enforce, or ``None`` to accept any number of results.
    """

    owner: Optional[Any] = None
    """
    The query this stage quantifies, reported in quantification errors.
    """

    def transform(
        self, results: Iterator[OperationResult]
    ) -> Iterator[OperationResult]:
        if self.constraint is None:
            yield from results
            return
        number_of_results = 0
        for result in results:
            number_of_results += 1
            self.constraint.assert_satisfaction(
                number_of_results, self.owner, done=False
            )
            yield result
        self.constraint.assert_satisfaction(number_of_results, self.owner, done=True)
