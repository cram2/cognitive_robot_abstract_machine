"""
Constructs for expressing Pearl-style causal (``do()``) queries in the Entity Query
Language.

Kept in its own module rather than folded into ``operators/core_logical_operators.py``
or ``core/variable.py`` (the modules :class:`Cause`/:class:`CausesEffect` otherwise
resemble) so causal-specific code has its own, easily reviewable surface, mirroring how
``probabilistic_model`` isolates its own causal code under a ``causal`` subpackage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Any, Iterable, List, Optional, Type, Self

import random_events.variable
from krrood.entity_query_language.core.base_expressions import (
    BinaryExpression,
    OperationResult,
    UnaryExpression,
)
from krrood.entity_query_language.core.variable import Literal
from krrood.entity_query_language.exceptions import (
    CausesEffectRequiresEqualityComparator,
)
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalOperator,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.mapped_variable import Attribute
    from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
        ProbabilisticCircuit,
    )


@dataclass(eq=False, repr=False)
class CausalRoleMarker(Literal):
    """
    Marks a keyword argument of a match pattern as a variable whose value the query
    searches for rather than one it observes. Always wraps ``Ellipsis``.

    Each kind is written as one shared instance with no type of its own, so every
    attribute it marks gets a fresh copy carrying that attribute's type, and marking
    several attributes never makes them interfere.
    """

    _value_: Any = field(default=Ellipsis, init=False)

    def _as_assigned_variable_(self, attribute_type: Optional[Type]) -> Self:
        """
        :param attribute_type: The type of the attribute this marks.
        :return: This marker if it already has a type, otherwise a fresh copy of it
            carrying the attribute's type.
        """
        if self._type_ is not None:
            return self
        return type(self)(_type_=attribute_type)


@dataclass(eq=False, repr=False)
class Cause(CausalRoleMarker):
    """
    Marks a :class:`~krrood.entity_query_language.query.match.Match` keyword argument as
    a ``do()``-intervention target searched for by the query, rather than an observed
    value.

    ``arm=cause`` means: find the value of ``arm`` whose intervention (Pearl's
    ``do(arm=value)``) best explains the effect declared via
    :meth:`~krrood.entity_query_language.query.match.Match.causes_effect`. There is no
    pinned-value form; pin a value with a plain assignment (``arm=0.3``) instead.
    """

    def __repr__(self) -> str:
        return "cause"


cause = Cause()
"""
Marks a :class:`~krrood.entity_query_language.query.match.Match` keyword argument as a
``do()``-intervention target searched for by the query (``arm=cause``).
"""


@dataclass(eq=False, repr=False)
class Confounder(CausalRoleMarker):
    """
    Marks a :class:`~krrood.entity_query_language.query.match.Match` keyword argument
    as a variable to adjust for when searching a :class:`Cause` intervention -- Pearl's
    backdoor-criterion adjustment set Z in
    ``P(effect | do(cause=v)) = sum_z P(effect | cause=v, Z=z) * P(Z=z)``.

    ``season=confounder`` means: season is a common cause of the searched
    :class:`Cause` and the declared effect, and must be summed back out rather than
    left baked into the correlation between them.
    """

    def __repr__(self) -> str:
        return "confounder"


confounder = Confounder()
"""
Marks a :class:`~krrood.entity_query_language.query.match.Match` keyword argument as a
variable to adjust for when searching a :class:`Cause` intervention -- see
:class:`Confounder`.
"""


@dataclass(eq=False, repr=False)
class CausesEffect(LogicalOperator, UnaryExpression):
    """
    Tags a condition as the effect side of a causal query.

    Evaluates transparently -- the same truth value as its wrapped condition -- under
    every backend, so filtering behaves identically whether or not a condition is
    wrapped in :meth:`~krrood.entity_query_language.query.match.Match.causes_effect`.
    Only :class:`~krrood.entity_query_language.backends.ProbabilisticBackend`
    additionally reads it, to find which variable(s) a :class:`Cause` search should
    optimize for.
    """

    cause_attributes: List[Attribute] = field(default_factory=list, kw_only=True)
    """
    The ``cause``-marked attribute(s) this effect condition explains, so its
    verbalization can name them (*"the arm causes its outcome to be SUCCESS"*) instead
    of a generic *"what causes"*.

    Empty when built directly rather than through
    :meth:`~krrood.entity_query_language.query.match.Match.causes_effect`.
    """

    def __post_init__(self):
        super().__post_init__()
        if (
            not isinstance(self._child_, BinaryExpression)
            or not self._child_._is_equality_literal_comparator_or_conjunction_()
        ):
            raise CausesEffectRequiresEqualityComparator(self._child_)

    def _evaluate__(
        self,
        sources: OperationResult,
    ) -> Iterable[OperationResult]:
        for child_result in self._evaluate_child_as_condition_(self._child_, sources):
            yield self._build_operation_result_with_truth_(
                child_result.is_true, child_result.bindings, child_result
            )


@dataclass
class CauseEffectVariables:
    """
    The cause candidates and effect variable a ``cause`` search resolves to.
    """

    cause_variables: List[random_events.variable.Variable]
    """
    The variable(s) a ``cause`` intervention is searched over.

    When there is more than one, each is tried independently -- there is no joint,
    multi-variable intervention -- and the one with the highest
    :attr:`~krrood.entity_query_language.operators.causal.ScoredIntervention.effect_probability_given_region`
    becomes the primary cause: the candidate whose own best region gives the highest
    ``P(effect | do(cause in best_region))``.
    """

    effect_variable: random_events.variable.Variable
    """
    The variable a ``causes_effect(...)`` condition declares as the effect.
    """

    confounder_variables: List[random_events.variable.Variable]
    """
    Variables assigned a ``confounder`` marker: Pearl's backdoor-criterion adjustment
    set, summed back out of each cause candidate's interventional probability so it is
    not left baked into the correlation between cause and effect.

    Empty for a query with no confounders declared, so the interventional search falls
    back to an empty adjustment set.
    """


@dataclass
class ScoredIntervention:
    """
    One cause candidate's best-region search result, scored for comparison against the
    other candidates when a query has more than one ``cause`` field.
    """

    cause_variable: random_events.variable.Variable
    """
    The candidate cause variable this result is for.
    """

    effect_probability_given_region: float
    """
    The interventional probability that the effect holds, given this variable is
    restricted to its best region -- how *reliably* this candidate's best value produces
    the effect.

    Comparable across candidates regardless of how much of each candidate's own domain
    that region happens to cover: the higher one is the better explanation.
    """

    narrowed_circuit: ProbabilisticCircuit
    """
    The interventional joint, truncated to the effect condition and this candidate's
    best region -- what the query samples from if this candidate turns out to be the
    primary cause.
    """
