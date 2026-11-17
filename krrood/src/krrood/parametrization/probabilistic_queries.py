"""
Entity queries that ask a probabilistic model directly, built by
:func:`~krrood.entity_query_language.factories.distribution_of` and
:func:`~krrood.entity_query_language.factories.probability_of`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import cached_property
from typing_extensions import Any, Iterator, TYPE_CHECKING

from krrood.entity_query_language.core.base_expressions import (
    HasExpression,
    SymbolicExpression,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.exceptions import (
    BackendCannotEvaluateProbabilisticQuery,
    EvidenceHasZeroProbability,
    NoSolutionFound,
)
from krrood.entity_query_language.factories import and_, count, entity
from krrood.parametrization.exceptions import JointQueryAcrossClassesNotSupported
from krrood.parametrization.parameterizer import (
    ConditionParameters,
    UnderspecifiedParameters,
)
from krrood.parametrization.random_events_translator import (
    WhereExpressionToRandomEventTranslator,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.mapped_variable import Attribute
    from krrood.entity_query_language.factories import ConditionType
    from krrood.entity_query_language.query.match import Match
    from krrood.parametrization.model_registries import ModelRegistry


@dataclass(kw_only=True, eq=False, repr=False)
class ProbabilisticQuery(Evaluable, HasExpression, ABC):
    """
    A query that asks a probabilistic model directly rather than selecting or generating
    instances.

    Only
    :class:`~krrood.parametrization.probabilistic_backend.ProbabilisticBackend` resolves
    one, through :meth:`_resolve_`, except :class:`Probability`, which also counts rows.
    """

    def _evaluate_natively_(self) -> Iterator:
        raise BackendCannotEvaluateProbabilisticQuery(expression=self)

    @abstractmethod
    def _resolve_(self, *, model_registry: ModelRegistry) -> Any:
        """
        Resolve this query against a model obtained from ``model_registry``.

        :param model_registry: The registry to resolve a model with.
        :return: Whatever the underlying ``ProbabilisticModel`` operation naturally
            returns -- a plain value, unwrapped.
        """


@dataclass(kw_only=True, eq=False, repr=False)
class Probability(ProbabilisticQuery):
    """
    The probability of a condition, optionally given evidence, each anything a
    ``.where(...)`` clause accepts.

    Resolved exactly from a model by
    :meth:`_resolve_`, or by counting matching rows under any other backend; see
    :doc:`/krrood/doc/eql/user/probabilistic_queries`.
    """

    condition: SymbolicExpression
    """
    The condition to compute the probability of.
    """

    evidence: SymbolicExpression | None = None
    """
    The condition the probability is conditioned on, or ``None`` for the unconditional
    probability.
    """

    def __post_init__(self):
        """
        Wrap a plain value given as condition or evidence in a literal.
        """
        if not isinstance(self.condition, SymbolicExpression):
            self.condition = Literal(_value_=self.condition)
        if self.evidence is not None and not isinstance(
            self.evidence, SymbolicExpression
        ):
            self.evidence = Literal(_value_=self.evidence)

    @cached_property
    def condition_and_evidence(self) -> SymbolicExpression:
        """
        :return: The conjunction of :attr:`condition` and :attr:`evidence`, or just
            :attr:`condition` without evidence.
        """
        if self.evidence is None:
            return self.condition
        return and_(self.condition, self.evidence)

    def _evaluate_natively_(self) -> Iterator[float]:
        """
        Counts the rows that satisfy :attr:`condition_and_evidence` and divides by the
        rows that satisfy :attr:`evidence` (all rows without evidence) -- works under
        any backend with an enumerable domain.

        :raises JointQueryAcrossClassesNotSupported: If the condition and evidence
            reference attributes reached from more than one ``variable(...)`` root, or
            none.
        :raises EvidenceHasZeroProbability: If no row satisfies the evidence.
        """
        referenced_attributes = WhereExpressionToRandomEventTranslator(
            conditions_root=self.condition_and_evidence
        ).variables.keys()
        roots = {attribute._chain_root_ for attribute in referenced_attributes}
        if len(roots) != 1:
            raise JointQueryAcrossClassesNotSupported(
                owner_classes={root._type_ for root in roots}
            )
        [root_variable] = roots

        matching_count = (
            entity(count(root_variable)).where(self.condition_and_evidence).first()
        )
        evidence_query = entity(count(root_variable))
        if self.evidence is not None:
            evidence_query = evidence_query.where(self.evidence)
        evidence_count = evidence_query.first()
        if evidence_count == 0:
            raise EvidenceHasZeroProbability(expression=self)
        yield matching_count / evidence_count

    def _resolve_(self, *, model_registry: ModelRegistry) -> float:
        """
        :raises EvidenceHasZeroProbability: If the model assigns the evidence zero
            probability.
        """
        parameters = ConditionParameters(condition=self.condition_and_evidence)
        model = model_registry.get_model(parameters=parameters)
        probability = model.probability(parameters.event)
        if self.evidence is None:
            return probability

        evidence_probability = model.probability(
            ConditionParameters(condition=self.evidence).event
        )
        if evidence_probability == 0:
            raise EvidenceHasZeroProbability(expression=self)
        return probability / evidence_probability

    def _get_expression_(self) -> SymbolicExpression:
        return self.condition_and_evidence

    def __repr__(self) -> str:
        """
        :return: The :func:`~krrood.entity_query_language.factories.probability_of` call that builds this query.
        """
        evidence = "" if self.evidence is None else f", evidence={self.evidence!r}"
        return f"probability_of({self.condition!r}{evidence})"


@dataclass(kw_only=True, eq=False, repr=False)
class Distribution(ProbabilisticQuery):
    """
    The distribution a :class:`~krrood.entity_query_language.query.match.Match`'s
    conditions describe: literal-valued keyword arguments condition the model,
    ``.where(...)`` conditions truncate it, and underspecified (``...``) fields are its
    free variables, e.g. ``distribution_of(a(Pick)(arm=0.3, outcome=...))``.
    """

    match: Match
    """
    The match whose conditions describe the distribution.
    """

    marginalize_for: tuple[Attribute, ...] = field(default_factory=tuple)
    """
    The match's variables to narrow the result to; empty keeps every free variable.
    """

    def __post_init__(self):
        """
        Reroot each of :attr:`marginalize_for` that is an attribute of the match's query
        on the query's selection.
        """
        query = self.match._get_expression_()
        self.marginalize_for = tuple(
            (
                query._rerooted_on_selection_(chain=attribute)
                if query._is_attribute_of_self_(expression=attribute)
                else attribute
            )
            for attribute in self.marginalize_for
        )

    def _resolve_(self, *, model_registry: ModelRegistry) -> Any:
        parameters = UnderspecifiedParameters(statement=self.match)
        model = model_registry.get_model(parameters=parameters)
        result = parameters.resolve_conditioned_and_truncated_model(model=model)
        if result is None:
            raise NoSolutionFound(expression=self)

        if self.marginalize_for:
            selected = [parameters.variables[v._name_] for v in self.marginalize_for]
            result = result.marginal(selected)
            if result is None:
                raise NoSolutionFound(expression=self)

        return result

    def _get_expression_(self) -> SymbolicExpression:
        return self.match._get_expression_()

    def __repr__(self) -> str:
        """
        :return: The :func:`~krrood.entity_query_language.factories.distribution_of` call that builds this query.
        """
        args = (
            f", marginalize_for={self.marginalize_for!r}"
            if self.marginalize_for
            else ""
        )
        return f"distribution_of({self.match!r}{args})"
