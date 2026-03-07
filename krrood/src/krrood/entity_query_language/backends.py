from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Iterable, TypeVar

from sqlalchemy.orm import sessionmaker

from krrood.entity_query_language.failures import (
    NoSolutionFound,
    GenerativeBackendQueryIsNotUnderspecifiedVariable,
)
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.query.query import Query
from krrood.ormatic.eql_interface import eql_to_sql
from krrood.parametrization.model_registries import ModelRegistry
from krrood.parametrization.parameterizer import (
    MatchVariable,
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
class PythonBackend(SelectiveBackend):
    """
    A domain that selects elements from a python process. This is just ordinary EQL.
    """

    def evaluate(self, expression: Query) -> Iterable:
        yield from expression.evaluate()


@dataclass
class ProbabilisticBackend(GenerativeBackend):
    """
    A backend that generates elements from a tractable probabilistic model.
    """

    model_registry: ModelRegistry
    """
    A model registry that can be used to resolve match statements to probabilistic models.
    """

    number_of_samples: int = 50
    """
    The number of samples to generate.
    """

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:

        # generate parameters from example instance values
        parameters = UnderspecifiedParameters(expression)

        # apply conditions from the parameters
        conditioned, _ = self.model_registry.get_model(expression).conditional(
            parameters.assignments_for_conditioning
        )

        if conditioned is None:
            raise NoSolutionFound(expression.expression)

        # apply conditions from the where statements
        if parameters.truncation_event:
            truncated, _ = conditioned.truncated(parameters.truncation_event)

            if truncated is None:
                raise NoSolutionFound(expression.expression)
        else:
            truncated = conditioned

        samples = truncated.sample(self.number_of_samples)

        # create new objects with the values from the samples
        for sample in samples:
            instance = parameters.create_instance_from_variables_and_sample(
                truncated.variables, sample
            )
            yield instance
