"""
The backends that answer entity queries: by evaluating them in this process or by
generating the instances they describe.
"""

import enum
import uuid
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing_extensions import ClassVar, Iterable, Iterator, TypeVar

import logging
from krrood.entity_query_language.verbalization.vocabulary.english import Directive

from krrood.entity_query_language.core.base_expressions import (
    Bindings,
    Selectable,
    SymbolicExpression,
)
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
        if not isinstance(expression, Match):
            raise GenerativeBackendQueryIsNotUnderspecifiedVariable(
                expression=expression
            )
        yield from self._evaluate(expression=expression)

    @abstractmethod
    def _evaluate(self, *, expression: Match[T]) -> Iterable[T]: ...


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
class EntityQueryLanguageGenerativeBackend(GenerativeBackend):
    """
    A generative backend that constructs new instances deterministically: it treats a
    match's unspecified leaves as variables, enumerates every combination over their
    (discrete) domains, constructs an instance per combination via the type's
    constructor, and keeps those that satisfy the match: the values its pattern states
    and its ``where`` conditions.
    """

    @classproperty
    def reads_state(cls) -> bool:
        """
        Unlike :meth:`QueryBackend.reads_state`, holds: generation selects in this
        process.
        """
        return True

    def _evaluate(self, *, expression: Match[T]) -> Iterable[T]:
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

        instances = (
            expression._construct_instance_from_bindings_(bindings=bindings)
            for bindings in self._enumerate_bindings(
                enumerated_variables=enumerated_variables
            )
        )
        satisfying = expression._select_satisfying_(instances=instances)
        yield from satisfying._quantify_(
            quantifier_type=expression._quantifier_type_
        )._evaluate_natively_()

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

    @staticmethod
    def _enumerate_bindings(
        *,
        enumerated_variables: dict[uuid.UUID, Selectable],
    ) -> Iterator[Bindings]:
        """
        :param enumerated_variables: The variable whose values each pattern variable
            takes, keyed by the identifier of that pattern variable.
        :return: The bindings of the pattern variables, one per combination of the
            enumerated values.
        """
        for combination in set_of(*enumerated_variables.values())._evaluate_natively_():
            yield {
                pattern_variable_id: combination[enumerated_variable]
                for pattern_variable_id, enumerated_variable in enumerated_variables.items()
            }
