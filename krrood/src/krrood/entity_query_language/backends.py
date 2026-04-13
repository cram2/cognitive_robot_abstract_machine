import enum
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import Iterable, TypeVar

from sqlalchemy.orm import sessionmaker
from typing_extensions import Dict

from krrood.entity_query_language.core.base_expressions import (
    Selectable,
    SymbolicExpression,
)
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.exceptions import (
    NoSolutionFound,
    GenerativeBackendQueryIsNotUnderspecifiedVariable,
    UnderspecifiedStatementInfeasibleForEntityQueryLanguageGeneration,
)
from krrood.entity_query_language.factories import (
    set_of,
    variable,
    variable_from,
    entity,
    an,
)
from krrood.entity_query_language.query.match import (
    Match,
    AttributeMatch,
    MatchVariable,
)
from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.query_graph import QueryGraph
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.parametrization.model_registries import (
    ModelRegistry,
    FullyFactorizedRegistry,
)
from krrood.parametrization.parameterizer import (
    UnderspecifiedParameters,
)

T = TypeVar("T")


@dataclass
class QueryBackend(ABC):
    """
    Base class for all query backends.
    Query backends are objects that answer queries by different means.
    """

    @abstractmethod
    def evaluate(self, expression: Query) -> Iterable[T]:
        """
        Generate answers that match the expression.

        :param expression: The expression to generate answers for.
        :return: An iterable of answers.
        """


@dataclass
class SelectiveBackend(QueryBackend, ABC):
    """
    Selective backends are backends that select elements from existing data.
    These can take any query as input.
    """


@dataclass
class GenerativeBackend(QueryBackend, ABC):
    """
    Generative backends are backends that generate new elements.
    Generative backends have to take match expressions as input, since they need to construct new objects, and currently
    {py:class}`~krrood.entity_query_language.query.match.Match` is the only way to do so.
    """

    def evaluate(self, expression: Query) -> Iterable[T]:
        if not isinstance(expression, Match):
            raise GenerativeBackendQueryIsNotUnderspecifiedVariable(expression)
        yield from self._evaluate(expression)

    @abstractmethod
    def _evaluate(self, expression: Match[T]) -> Iterable[T]: ...


@dataclass
class SQLAlchemyBackend(SelectiveBackend):
    """
    A backend that selects elements from a database that is available via SQLAlchemy.
    """

    session_maker: sessionmaker
    """
    The session maker used for the database interactions.
    """

    def evaluate(self, expression: Query) -> Iterable:
        session = self.session_maker()
        translator = eql_to_sql(expression, session)
        yield from translator.evaluate()


@dataclass
class EntityQueryLanguageBackend(SelectiveBackend):
    """
    A domain that selects elements from a python process. This is just ordinary EQL.
    """

    def evaluate(self, expression: Query) -> Iterable:
        if isinstance(expression, Match) and not isinstance(expression, MatchVariable):
            yield from self._evaluate_underspecified(expression)
            return
        yield from expression.evaluate()

    def _evaluate_underspecified(self, expression: Match[T]) -> Iterable[T]:
        """
        Evaluate an underspecified match expression by generating results from its constructor.

        :param expression: The underspecified match expression.
        :return: A newly generated instance of `T` that is compliant with the match expression's constraints.
        """

        variables: Dict[str, Variable] = {}

        for attribute_match in expression.matches_with_variables:
            self._check_if_attribute_match_is_suitable_for_generation(attribute_match)
            variables[attribute_match.name_from_variable_access_path] = (
                self._convert_attribute_match_to_variable(attribute_match)
            )

        expression.variable._update_domain_(
            self._generate_raw_results(expression, variables)
        )

        filtered_results = an(entity(expression.variable))

        if expression._where_conditions_:
            filtered_results = filtered_results.where(*expression._where_conditions_)
        yield from filtered_results.evaluate()

    def _check_if_attribute_match_is_suitable_for_generation(
        self, attribute_match: AttributeMatch
    ):
        """
        Raise an error if an assignment in the match cannot be used to generate solutions.
        :param attribute_match: The attribute match to check.
        """
        if isinstance(
            attribute_match.assigned_value, type(Ellipsis)
        ) and not issubclass(attribute_match.assigned_variable._type_, enum.Enum):
            raise UnderspecifiedStatementInfeasibleForEQLGeneration(attribute_match)

    def _convert_attribute_match_to_variable(self, attribute_match: AttributeMatch):
        """
        Convert an attribute match to a variable, handling ellipsis assignments for enum fields.
        :param attribute_match: The attribute match to convert.
        :return: A variable representing the attribute match.
        """
        # convert ellipsis assignments for enum fields to symbolic expressions
        if isinstance(attribute_match.assigned_value, type(Ellipsis)) and issubclass(
            attribute_match.assigned_variable._type_, enum.Enum
        ):
            result = variable(
                attribute_match.assigned_variable._type_,
                list(attribute_match.assigned_variable._type_),
            )

        # keep symbolic expressions as is
        elif isinstance(attribute_match.assigned_value, SymbolicExpression):
            result = attribute_match.assigned_value

        # convert concrete objects to symbolic expressions
        else:
            result = variable(
                type(attribute_match.assigned_value),
                [attribute_match.assigned_value],
            )
        return result

    def _generate_raw_results(
        self, expression: Match[T], variables: Dict[str, Variable]
    ) -> Iterable[T]:
        """
        Generate instances from a given match expression and variables.
        :param expression: The match expression to generate instances from.
        :param variables: The variables used in the match expression.
        :return: A generator yielding instances generated from the match expression.
        """
        all_combinations = set_of(*variables.values())
        for combination in all_combinations.evaluate():
            for variable_name, value in zip(variables, combination.values()):
                mapped_variable = expression._get_mapped_variable_by_name(variable_name)
                mapped_variable._value_ = value

            expression._update_kwargs_from_literal_values()
            yield expression.construct_instance()


@dataclass
class ProbabilisticBackend(GenerativeBackend):
    """
    A backend that generates elements from a tractable probabilistic model using a model registry.
    """

    model_registry: ModelRegistry = field(default_factory=FullyFactorizedRegistry)
    """
    A model registry that can be used to resolve match statements to probabilistic models.
    """

    number_of_samples: int = field(kw_only=True, default=50)
    """
    The number of samples to generate.
    This is only used if the query does not specify a limit.
    """

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:

        # generate parameters from example instance values
        parameters = UnderspecifiedParameters(expression)

        model = self.model_registry.get_model(parameters)

        # apply conditions from the parameters
        conditioned, _ = model.conditional(parameters.assignments_for_conditioning)

        if conditioned is None:
            raise NoSolutionFound(expression.expression)

        # apply conditions from the where statements
        if parameters.truncation_event:
            truncated, _ = conditioned.truncated(parameters.truncation_event)

            if truncated is None:
                raise NoSolutionFound(expression.expression)
        else:
            truncated = conditioned

        number_of_samples = expression.expression._limit_ or self.number_of_samples

        samples = truncated.sample(number_of_samples)

        # create new objects with the values from the samples
        for sample in samples:
            instance = parameters.create_instance_from_variables_and_sample(
                truncated.variables, sample
            )
            yield instance

        raise StopIteration()
