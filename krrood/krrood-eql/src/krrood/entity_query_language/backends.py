"""
The backends that answer entity queries: by evaluating them in this process or by
generating the instances they describe.
"""

from __future__ import annotations

import enum
import uuid
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing_extensions import Any, ClassVar, Iterable, Iterator, Self, TypeVar

import logging
from krrood.entity_query_language.verbalization.vocabulary.english import Directive

from krrood.entity_query_language.core.base_expressions import (
    Bindings,
    OperationResult,
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.enums import CombinationOrder
from krrood.entity_query_language.operators.causal import (
    Cause,
)
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.exceptions import (
    BackendCannotEvaluateCause,
    GenerativeBackendQueryIsNotUnderspecifiedVariable,
    SelectiveBackendCannotResolveEllipsisMatch,
    UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration,
)
from krrood.entity_query_language.factories import set_of, variable
from krrood.entity_query_language.query.match import Match, AttributeMatch
from krrood.patterns.cartesian_product import CartesianProductByRankSum
from krrood.patterns.classproperty import classproperty

logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass(kw_only=True)
class QueryBackend(ABC):
    """
    Base class for all query backends.

    Query backends are objects that answer queries by different means.
    """

    opening_directive: ClassVar[Directive | None] = None
    """
    The opening verb a verbalization uses when this backend evaluates the expression
    (``None`` keeps the query-type default).

    A backend declares its own performative so the verbalization layer never inspects
    concrete backend types.
    """

    raise_on_unresolvable_cause: bool = field(default=False, kw_only=True)
    """
    Whether to raise instead of warning when an expression contains a ``cause``
    intervention this backend cannot resolve causally; otherwise the cause is treated as
    an ordinary unspecified field.

    Read only by :class:`SelectiveBackend` and
    :class:`EntityQueryLanguageGenerativeBackend`.
    """

    @classproperty
    def reads_state(cls) -> bool:
        """
        Whether this backend's evaluations read the values of ground terms from the
        state they are given.
        """
        return False

    @abstractmethod
    def evaluate(self, *, expression: Evaluable) -> Iterable[T]:
        """
        Generate answers that match the expression.

        :param expression: The expression to generate answers for.
        :return: An iterable of answers.
        """

    def _warn_or_raise_on_unresolved_cause_(self, *, expression: Evaluable) -> None:
        """
        Warn (or, if :attr:`raise_on_unresolvable_cause` is set, raise) when
        *expression* is a :class:`~krrood.entity_query_language.query.match.Match`
        containing a `Cause` this backend has no causal graph to resolve.

        :param expression: The expression about to be evaluated.
        """
        if not (isinstance(expression, Match) and expression._has_cause_attributes_):
            return
        if self.raise_on_unresolvable_cause:
            raise BackendCannotEvaluateCause(match=expression, backend_type=type(self))
        logger.warning(
            BackendCannotEvaluateCause(match=expression, backend_type=type(self))
        )


@dataclass(kw_only=True)
class SelectiveBackend(QueryBackend, ABC):
    """
    Selective backends are backends that select elements from existing data.

    These can take any query as input.
    """

    opening_directive: ClassVar[Directive | None] = Directive.FIND
    """
    Selecting from existing data reads as *"Find …"*.
    """

    def evaluate(self, *, expression: Evaluable) -> Iterable[T]:
        if isinstance(expression, Match) and expression._has_ellipsis_attributes_:
            raise SelectiveBackendCannotResolveEllipsisMatch(match=expression)
        self._warn_or_raise_on_unresolved_cause_(expression=expression)
        yield from self._evaluate(expression=expression)

    @abstractmethod
    def _evaluate(self, *, expression: Evaluable) -> Iterable[T]: ...


@dataclass(kw_only=True)
class GenerativeBackend(QueryBackend, ABC):
    """
    Generative backends are backends that generate new elements.

    Generative backends have to take match expressions as input, since they need to construct new objects, and currently
    {py:class}`~krrood.entity_query_language.query.match.Match` is the only way to do so.
    """

    opening_directive: ClassVar[Directive | None] = Directive.GENERATE
    """
    Generating new elements reads as *"Generate …"*.
    """

    def evaluate(self, *, expression: Evaluable) -> Iterable[T]:
        self._require_match(expression=expression)
        yield from self._evaluate(expression=expression)

    @staticmethod
    def _require_match(*, expression: Evaluable) -> None:
        """
        :param expression: An expression handed to this backend.
        :raises GenerativeBackendQueryIsNotUnderspecifiedVariable: If `expression` is not
            a match, the only expression instances can be generated from.
        """
        if not isinstance(expression, Match):
            raise GenerativeBackendQueryIsNotUnderspecifiedVariable(
                expression=expression
            )

    @abstractmethod
    def _evaluate(self, *, expression: Match[T]) -> Iterable[T]: ...


# %% answering a match under values given for some of its variables


@dataclass(kw_only=True)
class Answer:
    """
    An instance generated for a match, with the values its variables took for it.
    """

    value: Any
    """
    The generated instance.
    """

    variable_values: Bindings
    """
    The value each variable of the match took for :attr:`value`, the variables of its
    pattern among them, keyed by the identifier of the variable.
    """


@dataclass(kw_only=True)
class EvaluatesUnderBindings(ABC):
    """
    A backend that answers a match under values given for some of its variables, the way
    a correlated subquery is answered under the bindings of the query around it, and
    reports the values the variables took for every answer.
    """

    @abstractmethod
    def evaluate_under(
        self, *, expression: Evaluable, variable_values: Bindings
    ) -> Iterator[Answer]:
        """
        :param expression: The match to answer.
        :param variable_values: The value of each variable of `expression` that is
            fixed already, keyed by the identifier of the variable; every other variable
            ranges over its domain.
        :return: The answers, in the order :meth:`QueryBackend.evaluate` yields their
            values.
        """


@dataclass(kw_only=True)
class EntityQueryLanguageBackend(SelectiveBackend):
    """
    A backend that selects elements in this python process.

    This is just ordinary EQL: each expression evaluates itself natively (queries and matches both select over their domains).
    Constructing new instances is the job of a :class:`GenerativeBackend`.
    """

    @classproperty
    def reads_state(cls) -> bool:
        """
        Unlike :meth:`QueryBackend.reads_state`, holds: evaluation runs in this process.
        """
        return True

    def _evaluate(self, *, expression: Evaluable) -> Iterable:
        yield from expression._evaluate_natively_()


@dataclass(kw_only=True)
class EntityQueryLanguageGenerativeBackend(GenerativeBackend, EvaluatesUnderBindings):
    """
    A generative backend that constructs new instances deterministically: it treats a
    match's unspecified leaves as variables, enumerates every combination over their
    (discrete) domains, constructs an instance per combination via the type's
    constructor, and keeps those that satisfy the match: the values its pattern states
    and its ``where`` conditions.
    """

    combination_order: CombinationOrder = CombinationOrder.NESTED
    """
    The order the combinations of the values of the match's variables are enumerated in.
    """

    @classproperty
    def reads_state(cls) -> bool:
        """
        Unlike :meth:`QueryBackend.reads_state`, holds: generation selects in this
        process.
        """
        return True

    def _evaluate(self, *, expression: Match[T]) -> Iterable[T]:
        for answer in self._answers(expression=expression, variable_values={}):
            yield answer.value

    def evaluate_under(
        self, *, expression: Evaluable, variable_values: Bindings
    ) -> Iterator[Answer]:
        self._require_match(expression=expression)
        yield from self._answers(expression=expression, variable_values=variable_values)

    def _answers(
        self, *, expression: Match[T], variable_values: Bindings
    ) -> Iterator[Answer]:
        """
        :param expression: The match to answer.
        :param variable_values: See :meth:`EvaluatesUnderBindings.evaluate_under`.
        :return: The instances constructed for the combinations of the values of the
            variables of `expression` that satisfy it, with those values.
        """
        self._warn_or_raise_on_unresolved_cause_(expression=expression)
        enumerated_variables: dict[uuid.UUID, Selectable] = {}
        for attribute_match in expression._matches_with_variables_:
            self._check_attribute_match_is_suitable_for_generation(
                attribute_match=attribute_match
            )
            enumerated_variables[attribute_match.assigned_variable._id_] = (
                self._convert_attribute_match_to_variable(
                    attribute_match=attribute_match
                )
            )

        values_of_instance: dict[int, Bindings] = {}

        def constructed_instances() -> Iterator[T]:
            """
            :return: An instance per combination whose nested matches hold for it,
                whose values are kept in `values_of_instance` until it is selected.
            """
            for combination in self._enumerate_combinations(
                enumerated_variables=enumerated_variables,
                variable_values=variable_values,
            ):
                for instance in expression._instances_constructed_from_bindings_(
                    bindings=combination.pattern_bindings
                ):
                    values_of_instance[id(instance)] = (
                        combination.pattern_bindings | combination.variable_values
                    )
                    yield instance

        satisfying = expression._select_satisfying_(instances=constructed_instances())
        for value in satisfying._quantify_(
            quantifier_type=expression._quantifier_type_
        )._evaluate_natively_():
            yield Answer(value=value, variable_values=values_of_instance.pop(id(value)))

    @staticmethod
    def _check_attribute_match_is_suitable_for_generation(
        *,
        attribute_match: AttributeMatch,
    ) -> None:
        """
        Raise if an assignment in the match cannot be used to generate solutions.

        :param attribute_match: The attribute match to check.
        :raises UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration: If a
            non-enum leaf is left fully unspecified (``...`` or ``cause``), which
            deterministic generation cannot enumerate (use the
            :class:`~krrood.parametrization.probabilistic_backend.ProbabilisticBackend` instead).
        """
        if isinstance(
            attribute_match.assigned_value, (type(Ellipsis), Cause)
        ) and not issubclass(attribute_match.assigned_variable._type_, enum.Enum):
            raise UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration(
                attribute_match=attribute_match
            )

    @staticmethod
    def _convert_attribute_match_to_variable(
        *,
        attribute_match: AttributeMatch,
    ) -> Selectable:
        """
        Convert an attribute match into a variable to enumerate, handling ellipsis (and,
        identically, ``cause``) assignments for enum fields and concrete values.

        :param attribute_match: The attribute match to convert.
        :return: A variable (or symbolic expression) representing the attribute match.
        """
        if isinstance(
            attribute_match.assigned_value, (type(Ellipsis), Cause)
        ) and issubclass(attribute_match.assigned_variable._type_, enum.Enum):
            return variable(
                attribute_match.assigned_variable._type_,
                list(attribute_match.assigned_variable._type_),
            )
        if isinstance(attribute_match.assigned_value, SymbolicExpression):
            return attribute_match.assigned_value
        return variable(
            type(attribute_match.assigned_value),
            [attribute_match.assigned_value],
        )

    def _enumerate_combinations(
        self,
        *,
        enumerated_variables: dict[uuid.UUID, Selectable],
        variable_values: Bindings,
    ) -> Iterator[EnumeratedCombination]:
        """
        :param enumerated_variables: The variable whose values each pattern variable
            takes, keyed by the identifier of that pattern variable.
        :param variable_values: See :meth:`EvaluatesUnderBindings.evaluate_under`.
        :return: The combinations of the enumerated values, in
            :attr:`combination_order`.
        """
        if self.combination_order == CombinationOrder.NESTED:
            yield from VariablesEnumeratedTogether.from_expressions(
                expressions=enumerated_variables
            ).combinations(variable_values=variable_values)
            return
        groups = VariablesEnumeratedTogether.groups_sharing_variables(
            expressions=enumerated_variables
        )
        for parts in CartesianProductByRankSum(
            factors=[
                group.combinations(variable_values=variable_values) for group in groups
            ]
        ):
            yield EnumeratedCombination.merged(parts=parts)


# %% enumerating the values of the variables of a match


@dataclass(kw_only=True)
class EnumeratedCombination:
    """
    One combination of the values of the variables a generative backend enumerates.
    """

    pattern_bindings: Bindings
    """
    The value of each pattern variable, keyed by the identifier of the pattern variable,
    which the instance is constructed from.
    """

    variable_values: Bindings
    """
    The value each variable the pattern variables are read from took, keyed by the
    identifier of the variable.
    """

    @classmethod
    def merged(cls, *, parts: Iterable[EnumeratedCombination]) -> Self:
        """
        :param parts: Combinations of disjoint sets of variables.
        :return: The combination holding the values of all of `parts`.
        """
        merged = cls(pattern_bindings={}, variable_values={})
        for part in parts:
            merged.pattern_bindings |= part.pattern_bindings
            merged.variable_values |= part.variable_values
        return merged


@dataclass(kw_only=True)
class VariablesEnumeratedTogether:
    """
    Expressions whose values are enumerated in one nested product, since they share
    variables that have to take one value in each combination.
    """

    expressions: dict[uuid.UUID, Selectable]
    """
    The expression each pattern variable takes its values from, keyed by the identifier
    of the pattern variable.
    """

    variables: dict[uuid.UUID, Variable]
    """
    The variables :attr:`expressions` read, keyed by their identifier.
    """

    @classmethod
    def from_expressions(cls, *, expressions: dict[uuid.UUID, Selectable]) -> Self:
        """
        :param expressions: The expression each pattern variable takes its values from.
        :return: The expressions enumerated together, with the variables they read.
        """
        return cls(
            expressions=dict(expressions),
            variables={
                variable_._id_: variable_
                for expression in expressions.values()
                for variable_ in (expression, *expression._descendants_)
                if isinstance(variable_, Variable)
            },
        )

    @classmethod
    def groups_sharing_variables(
        cls, *, expressions: dict[uuid.UUID, Selectable]
    ) -> list[Self]:
        """
        :param expressions: The expression each pattern variable takes its values from.
        :return: The fewest groups of `expressions` such that no two groups read the
            same variable, in the order their first expression comes in.
        """
        groups: list[Self] = []
        for pattern_variable_id, expression in expressions.items():
            group = cls.from_expressions(expressions={pattern_variable_id: expression})
            sharing = [
                index
                for index, other in enumerate(groups)
                if other.variables.keys() & group.variables.keys()
            ]
            for index in reversed(sharing):
                group = groups.pop(index).merged_with(other=group)
            groups.insert(sharing[0] if sharing else len(groups), group)
        return groups

    def merged_with(self, *, other: VariablesEnumeratedTogether) -> Self:
        """
        :param other: Expressions sharing a variable with these.
        :return: These expressions followed by those of `other`.
        """
        return type(self)(
            expressions=self.expressions | other.expressions,
            variables=self.variables | other.variables,
        )

    def combinations(
        self, *, variable_values: Bindings
    ) -> Iterator[EnumeratedCombination]:
        """
        :param variable_values: See :meth:`EvaluatesUnderBindings.evaluate_under`.
        :return: The combinations of the values of :attr:`expressions`, in nested
            order, each variable in `variable_values` taking the value given for it.
        """
        selected = {
            expression._id_: expression for expression in self.expressions.values()
        } | self.variables
        selection = set_of(*selected.values())
        for result in selection._create_new_context_and_evaluate_(
            sources=OperationResult(bindings=dict(variable_values))
        ):
            row = selection._process_result_(result=result)
            yield EnumeratedCombination(
                pattern_bindings={
                    pattern_variable_id: row[expression]
                    for pattern_variable_id, expression in self.expressions.items()
                },
                variable_values={
                    variable_id: row[variable_]
                    for variable_id, variable_ in self.variables.items()
                },
            )
