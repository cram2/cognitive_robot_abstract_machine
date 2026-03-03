from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import Type, Dict

from krrood.entity_query_language.query.match import Match
from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class ModelRegistry(ABC):
    """
    A registry that selects probabilistic models for given match-queries.
    """

    @abstractmethod
    def get_model(self, expression: Match) -> ProbabilisticModel:
        """
        :param expression: The expression to get a model for.
        :return: A probabilistic model that can be used to generate answers for the given expression.
        """


@dataclass
class DictRegistry(ModelRegistry):
    """
    A registry that uses a dictionary to keep all models.
    """

    models: Dict[Type, ProbabilisticModel]
    """
    A dictionary that maps classes to probabilistic models.
    """

    def get_model(self, expression: Match) -> ProbabilisticModel:
        return self.models[expression._expression.selected_variable._type_]
