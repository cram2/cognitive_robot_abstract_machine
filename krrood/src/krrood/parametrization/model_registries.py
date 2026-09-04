from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing_extensions import Type, Dict

from krrood.parametrization.exceptions import RelationalCircuitRegistryRequiresMatch
from krrood.parametrization.parameterizer import (
    ModelQueryParameters,
    UnderspecifiedParameters,
)
from krrood.utils import get_class_and_attribute_name
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    GroundingMode,
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class ModelRegistry(ABC):
    """
    A registry that selects probabilistic models for given underspecified parameters of
    match-queries (or other probabilistic queries).
    """

    @abstractmethod
    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
        """
        :param parameters: The parameters to get a model for.
        :return: A probabilistic model that can be used to generate answers for the given expression.
        """


@dataclass
class FullyFactorizedRegistry(ModelRegistry):
    """
    A registry that always returns a fully factorized model.
    """

    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
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

    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
        return self.models[parameters.queried_class]


@dataclass
class RelationalCircuitRegistry(ModelRegistry):
    """
    A registry that grounds a RelationalProbabilisticCircuit for the queried statement
    and aligns its variable names to the UnderspecifiedParameters convention before
    returning.

    Only supports :class:`~krrood.parametrization.parameterizer.UnderspecifiedParameters`
    (i.e. a ``Match``, directly or wrapped by ``distribution_of(...)``): grounding needs
    the full match statement, which ``average``'s/``probability_of``'s lighter
    parameter classes don't carry -- resolving one of those raises
    :class:`~krrood.parametrization.exceptions.RelationalCircuitRegistryRequiresMatch`
    rather than failing on a missing attribute.

    When the query also declares ``cause``/``causes_effect``/``confounder`` markers,
    grounding uses :attr:`causal_grounding_mode` instead of
    :attr:`~probabilistic_model.probabilistic_circuit.relational.rspn.GroundingMode.PREDICTIVE`,
    so the aggregation latents those markers name survive grounding instead of being
    integrated out, and the result is wrapped as a verified
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`
    instead of a plain circuit -- one more kind of cause this same registry can answer
    about, not a parallel system. Non-causal queries are unaffected.
    """

    relational_probabilistic_circuit: RelationalProbabilisticCircuit
    """
    The trained relational probabilistic circuit to ground.
    """

    causal_grounding_mode: GroundingMode = GroundingMode.CAUSAL_SAMPLED
    """
    Grounding mode used only when the query declares
    ``cause``/``causes_effect``/``confounder`` markers. Non-causal queries always
    ground with ``GroundingMode.PREDICTIVE``, unchanged from before.
    """

    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
        if not isinstance(parameters, UnderspecifiedParameters):
            raise RelationalCircuitRegistryRequiresMatch(parameters)
        is_causal_query = bool(parameters.search_cause_variables)
        grounding_mode = (
            self.causal_grounding_mode if is_causal_query else GroundingMode.PREDICTIVE
        )
        grounded = self.relational_probabilistic_circuit.ground(
            parameters.statement, grounding_mode
        )
        class_prefix = self.relational_probabilistic_circuit.class_.__name__
        rename_map = {}
        for circuit_var in grounded.variables:
            qualified_name = get_class_and_attribute_name(
                class_prefix, circuit_var.name
            )
            if qualified_name in parameters.variables:
                rename_map[circuit_var] = parameters.variables[qualified_name]
        grounded.update_variables(rename_map)
        if not is_causal_query:
            return grounded

        effect_variables = list(parameters.effect_variables_from_causes_effect)
        tree = MarginalDeterminismTreeNode.from_causal_graph(
            parameters.search_cause_variables, effect_variables
        )
        causal_circuit = CausalCircuit.from_probabilistic_circuit(
            grounded, tree, parameters.search_cause_variables, effect_variables
        )
        causal_circuit.verify_support_determinism()
        return causal_circuit


@dataclass
class CausalCircuitRegistry(ModelRegistry):
    """
    A registry that maps target classes directly to pre-built causal circuits, so a
    ``cause``/``causes_effect()`` query can be routed through that circuit's
    ``backdoor_adjustment`` method.

    See
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.
    """

    circuits: Dict[Type, CausalCircuit]
    """
    A dictionary that maps classes to pre-built causal circuits.
    """

    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
        return self.circuits[parameters.queried_class]
