"""
The backends that answer entity queries: by evaluating them in this process or by
generating the instances they describe.
"""

import enum
import uuid
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from functools import cached_property
from typing_extensions import Any, ClassVar, Iterable, Iterator, TypeVar

import logging
from krrood.entity_query_language.verbalization.vocabulary.english import Directive

from krrood.entity_query_language.core.base_expressions import (
    Bindings,
    OperationResult,
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
from krrood.entity_query_language.query.quantifiers import An, ResultQuantifier
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


# %% generated instances


@dataclass(kw_only=True)
class GeneratedInstance:
    """
    An instance a generative backend constructed for a match, with the values the
    match's variables took for it.
    """

    instance: Any
    """
    The constructed instance.
    """

    variable_values: Bindings
    """
    The value each expression the match was given for an attribute, such as a variable,
    took for :attr:`instance`, keyed by the identifier of that expression.
    """


@dataclass(kw_only=True)
class ConstructedInstance:
    """
    An instance constructed for a match, with the row of values it was constructed from.
    """

    instance: Any
    """
    The constructed instance.
    """

    row: Bindings
    """
    The values of the pattern's variables :attr:`instance` was constructed from, keyed
    by the identifier of the variable assigned to each attribute.
    """


@dataclass(kw_only=True)
class InstanceConstruction:
    """
    Constructs the instances of a match from rows of values of its pattern's variables,
    remembering the row each instance was constructed from.

    The instances are expected to be handed out in the order they were constructed, see
    :meth:`generated`.
    """

    match: Match
    """
    The match whose instances are constructed.
    """

    rows: Iterable[Bindings]
    """
    The values of the pattern's variables, one row per instance, keyed by the identifier
    of the variable assigned to each attribute.
    """

    _not_handed_out: dict[int, ConstructedInstance] = field(
        default_factory=dict, init=False, repr=False
    )
    """
    The instances constructed since the one :meth:`generated` handed out last, in the
    order they were constructed, keyed by the identity of the instance.
    """

    def __iter__(self) -> Iterator[Any]:
        """
        :return: An instance constructed from each of :attr:`rows`, in their order.
        """
        for row in self.rows:
            instance = self.match._construct_instance_from_bindings_(bindings=row)
            self._not_handed_out[id(instance)] = ConstructedInstance(
                instance=instance, row=row
            )
            yield instance

    @cached_property
    def given_expressions(self) -> dict[uuid.UUID, uuid.UUID]:
        """
        :return: The identifier of the pattern variable of every attribute the match was
            given an expression for, such as a variable, keyed by the identifier of that
            expression.
        """
        return {
            attribute_match.assigned_value._id_: attribute_match.assigned_variable._id_
            for attribute_match in self.match._matches_with_variables_
            if isinstance(attribute_match.assigned_value, SymbolicExpression)
        }

    def generated(self, *, instance: Any) -> GeneratedInstance:
        """
        Forget `instance` and the instances constructed before it, which were rejected
        or handed out already.

        :param instance: An instance this construction yielded and did not hand out yet.
        :return:`instance` with the values the expressions the match was given took in
            the row it was constructed from.
        """
        constructed_ids = list(self._not_handed_out)
        for rejected_id in constructed_ids[: constructed_ids.index(id(instance))]:
            del self._not_handed_out[rejected_id]
        row = self._not_handed_out.pop(id(instance)).row
        return GeneratedInstance(
            instance=instance,
            variable_values={
                expression_id: row[pattern_variable_id]
                for expression_id, pattern_variable_id in self.given_expressions.items()
                if pattern_variable_id in row
            },
        )


# %% generative backends


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
        for generated in self.generate(expression=expression, bound_values={}):
            yield generated.instance

    def generate(
        self, *, expression: Evaluable, bound_values: Bindings
    ) -> Iterator[GeneratedInstance]:
        """
        Generate the instances a match describes, each with the values its variables
        took, while every variable `bound_values` binds keeps its bound value.

        :param expression: The match to generate instances of.
        :param bound_values: Values already chosen for variables, keyed by the
            identifier of each variable; values of variables the match does not depend
            on are ignored.
        :return: The generated instances.
        :raises GenerativeBackendQueryIsNotUnderspecifiedVariable: If `expression` is
            not a match.
        """
        self._check_is_match(expression=expression)
        yield from self._generate(
            expression=expression,
            bound_values=bound_values,
            quantifier_type=expression._quantifier_type_,
        )

    def generate_without_quantifier(
        self, *, expression: Evaluable, bound_values: Bindings
    ) -> Iterator[GeneratedInstance]:
        """
        Differs from :meth:`generate` in generating every instance the match describes,
        whatever its quantifier requires of their number.
        """
        self._check_is_match(expression=expression)
        yield from self._generate(
            expression=expression, bound_values=bound_values, quantifier_type=An
        )

    @staticmethod
    def _check_is_match(*, expression: Evaluable) -> None:
        """
        :param expression: The expression to generate instances of.
        :raises GenerativeBackendQueryIsNotUnderspecifiedVariable: If `expression` is
            not a match.
        """
        if not isinstance(expression, Match):
            raise GenerativeBackendQueryIsNotUnderspecifiedVariable(
                expression=expression
            )

    @abstractmethod
    def _generate(
        self,
        *,
        expression: Match[T],
        bound_values: Bindings,
        quantifier_type: type[ResultQuantifier],
    ) -> Iterator[GeneratedInstance]:
        """
        :meth:`generate` once `expression` is known to be a match.

        :param quantifier_type: The quantifier the generated instances satisfy.
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

    def _generate(
        self,
        *,
        expression: Match[T],
        bound_values: Bindings,
        quantifier_type: type[ResultQuantifier],
    ) -> Iterator[GeneratedInstance]:
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

        construction = InstanceConstruction(
            match=expression,
            rows=self._enumerate_bindings(
                enumerated_variables=enumerated_variables, bound_values=bound_values
            ),
        )
        satisfying = expression._select_satisfying_(instances=construction)
        for instance in satisfying._quantify_(
            quantifier_type=quantifier_type
        )._evaluate_natively_():
            yield construction.generated(instance=instance)

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
        bound_values: Bindings,
    ) -> Iterator[Bindings]:
        """
        :param enumerated_variables: The variable whose values each pattern variable
            takes, keyed by the identifier of that pattern variable.
        :param bound_values: Values the enumerated variables, and the expressions their
            domains are read from, keep instead of being enumerated.
        :return: The bindings of the pattern variables, one per combination of the
            enumerated values.
        """
        combinations = set_of(*enumerated_variables.values())
        combinations.build()
        for result in combinations._create_new_context_and_evaluate_(
            sources=OperationResult(bindings=bound_values)
        ):
            if not result.is_true:
                continue
            combination = combinations._process_result_(result=result)
            yield {
                pattern_variable_id: combination[enumerated_variable]
                for pattern_variable_id, enumerated_variable in enumerated_variables.items()
            }
