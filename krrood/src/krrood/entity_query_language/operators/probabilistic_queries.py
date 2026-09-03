"""
Constructs for querying a probabilistic model directly from the Entity Query Language:
``distribution_of`` and ``probability_of``.

A third case -- the expectation of an attribute -- reuses the existing
:class:`~krrood.entity_query_language.operators.aggregators.Average` aggregator
(``average(...)``) instead of a bespoke construct here:
:class:`~krrood.entity_query_language.backends.ProbabilisticBackend` recognizes a bare
``average(...)`` selection and answers it in closed form via
``ProbabilisticModel.moment`` instead of sampling and averaging rows, so the same
declarative call reads correctly under either backend. See
:meth:`~krrood.entity_query_language.backends.ProbabilisticBackend._resolve_average`.

Bundled in one module rather than one file each (unlike ``operators/causal.py``, which
is one coupled do()-intervention feature) because these are parallel operations on the
same base, the same shape ``operators/aggregators.py`` bundles ``Sum``/``Max``/``Min``/
... in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing_extensions import Any, Iterator, Tuple, TYPE_CHECKING

from krrood.entity_query_language.core.base_expressions import (
    HasExpression,
    SymbolicExpression,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.exceptions import (
    BackendCannotEvaluateProbabilisticQuery,
    NoSolutionFound,
)
from krrood.entity_query_language.factories import count, entity
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
    from krrood.entity_query_language.query.match import Match
    from krrood.parametrization.model_registries import ModelRegistry


@dataclass(eq=False, repr=False)
class ProbabilisticQuery(Evaluable, HasExpression, ABC):
    """
    Shared base for EQL constructs that query a probabilistic model directly --
    ``distribution_of``, ``probability_of`` -- rather than selecting or generating
    rows/instances.

    This is a probabilistic operation, not a data selection, so by default none of
    these resolve natively or under any backend other than
    :class:`~krrood.entity_query_language.backends.ProbabilisticBackend`, which
    dispatches to :meth:`_resolve_` for whichever subclass it is given.
    :class:`Probability` is the one exception -- see its own
    :meth:`~Probability._evaluate_natively_`.
    """

    def _evaluate_natively_(self) -> Iterator:
        raise BackendCannotEvaluateProbabilisticQuery(self)

    @abstractmethod
    def _resolve_(self, model_registry: ModelRegistry) -> Any:
        """
        Resolve this query against a model obtained from ``model_registry``.

        :param model_registry: The registry to resolve a model with.
        :return: Whatever the underlying ``ProbabilisticModel`` operation naturally
            returns -- a plain value, unwrapped.
        """


@dataclass(eq=False, repr=False)
class Probability(ProbabilisticQuery):
    """
    The probability of a condition, e.g. ``probability_of(x.A > 5)`` for
    ``x = variable(MyClass)``. Accepts any condition a ``.where(...)`` clause does.

    Resolves two ways: exactly via :meth:`_resolve_` under a
    :class:`~krrood.entity_query_language.backends.ProbabilisticBackend`, or by
    counting matching rows via :meth:`_evaluate_natively_` under any other backend.
    See :doc:`/krrood/doc/eql/user/probabilistic_queries` for the full walkthrough.
    """

    condition: SymbolicExpression
    """
    The condition to compute the probability of.
    """

    def __post_init__(self):
        if not isinstance(self.condition, SymbolicExpression):
            self.condition = Literal(_value_=self.condition)

    def _evaluate_natively_(self) -> Iterator[float]:
        """
        Counts matching rows via ``entity(count(root)).where(condition)`` instead of
        resolving a model -- works under any backend with an enumerable domain.

        :raises JointQueryAcrossClassesNotSupported: If the condition references
            attributes reached from more than one ``variable(...)`` root, or none.
        """
        referenced_attributes = WhereExpressionToRandomEventTranslator(
            self.condition
        ).variables.keys()
        roots = {attribute._chain_root_ for attribute in referenced_attributes}
        if len(roots) != 1:
            raise JointQueryAcrossClassesNotSupported(
                {root._type_ for root in roots}
            )
        [root_variable] = roots

        matching_count = entity(count(root_variable)).where(self.condition).first()
        total_count = entity(count(root_variable)).first()
        yield matching_count / total_count

    def _resolve_(self, model_registry: ModelRegistry) -> float:
        parameters = ConditionParameters(self.condition)
        model = model_registry.get_model(parameters)
        return model.probability(parameters.event)

    def _get_expression_(self) -> SymbolicExpression:
        return self.condition

    def __repr__(self) -> str:
        return f"probability_of({self.condition!r})"


@dataclass(eq=False, repr=False)
class Distribution(ProbabilisticQuery):
    """
    Requests the distribution a :class:`~krrood.entity_query_language.query.match.Match`'s
    conditions describe -- the probabilistic interpretation of
    :py:func:`~krrood.entity_query_language.factories.a`/:py:func:`~krrood.entity_query_language.factories.an`/
    :py:func:`~krrood.entity_query_language.factories.the`. Exactly the same sequence
    :class:`~krrood.entity_query_language.backends.ProbabilisticBackend` already
    applies before *sampling* from a match -- literal-valued kwargs condition the
    circuit (``arm=0.3``), ``.where(...)`` conditions truncate it, and underspecified
    (``...``) fields are the joint's free variables -- just returned directly instead
    of sampled from:

    ``distribution_of(a(Pick)(arm=0.3, outcome=...))`` -- the distribution over
    ``outcome`` given ``arm == 0.3``.

    Optional keyword-only ``marginalize_for`` narrows the result to a subset of the
    match's free variables (further marginalization), e.g.
    ``distribution_of(match, marginalize_for=(match.variable.outcome,))``. Without it,
    every one of the match's free variables is kept.
    """

    match: Match
    """
    The match whose conditions describe the distribution.
    """

    marginalize_for: Tuple[Attribute, ...] = field(default_factory=tuple)
    """
    The match's variables to narrow the result to. Empty keeps every one of the
    match's free variables.
    """

    def _resolve_(self, model_registry: ModelRegistry) -> Any:
        parameters = UnderspecifiedParameters(self.match)
        model = model_registry.get_model(parameters)
        result = parameters.resolve_conditioned_and_truncated_model(model)
        if result is None:
            raise NoSolutionFound(self)

        if self.marginalize_for:
            selected = [
                parameters.variables[v._name_] for v in self.marginalize_for
            ]
            result = result.marginal(selected)
            if result is None:
                raise NoSolutionFound(self)

        return result

    def _get_expression_(self) -> SymbolicExpression:
        return self.match._get_expression_()

    def __repr__(self) -> str:
        args = (
            f", marginalize_for={self.marginalize_for!r}"
            if self.marginalize_for
            else ""
        )
        return f"distribution_of({self.match!r}{args})"
