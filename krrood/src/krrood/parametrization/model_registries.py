from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import Type, Dict

from krrood.parametrization.parameterizer import UnderspecifiedParameters
from krrood.utils import get_class_and_attribute_name
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class ModelRegistry(ABC):
    """
    A registry that selects probabilistic models for given underspecified parameters of
    match-queries.
    """

    @abstractmethod
    def get_model(self, parameters: UnderspecifiedParameters) -> ProbabilisticModel:
        """
        :param parameters: The parameters to get a model for.
        :return: A probabilistic model that can be used to generate answers for the given expression.
        """


@dataclass
class FullyFactorizedRegistry(ModelRegistry):
    """
    A registry that always returns a fully factorized model.
    """

    def get_model(self, parameters: UnderspecifiedParameters) -> ProbabilisticModel:
        return fully_factorized(parameters.variables.values())


@dataclass
class DictRegistry(ModelRegistry):
    """
    A registry that uses a dictionary to keep all models.
    """

    models: Dict[Type, ProbabilisticModel]
    """
    A dictionary that maps classes to probabilistic models.
    """

    def get_model(self, parameters: UnderspecifiedParameters) -> ProbabilisticModel:
        return self.models[parameters.statement._expression.selected_variable._type_]


@dataclass
class RelationalCircuitRegistry(ModelRegistry):
    """
    A registry that grounds a RelationalProbabilisticCircuit for the queried statement
    and aligns its variable names to the UnderspecifiedParameters convention before
    returning.
    """

    relational_probabilistic_circuit: RelationalProbabilisticCircuit
    """
    The trained relational probabilistic circuit to ground.
    """

    def get_model(self, parameters: UnderspecifiedParameters) -> ProbabilisticModel:
        grounded = self.relational_probabilistic_circuit.ground(parameters.statement)
        class_prefix = self.relational_probabilistic_circuit.class_.__name__
        rename_map = {}
        for circuit_var in grounded.variables:
            qualified_name = get_class_and_attribute_name(
                class_prefix, circuit_var.name
            )
            if qualified_name in parameters.variables:
                rename_map[circuit_var] = parameters.variables[qualified_name]
        grounded.update_variables(rename_map)
        return grounded
