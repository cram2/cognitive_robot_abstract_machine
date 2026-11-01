from abc import ABC, abstractmethod
from dataclasses import dataclass

from krrood.parametrization.exceptions import (
    RelationalCircuitRegistryRequiresMatch,
    UnboundedParameterError,
)
from krrood.parametrization.parameterizer import (
    ModelQueryParameters,
    UnderspecifiedParameters,
)
from krrood.utils import get_class_and_attribute_name
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.causal import (
    RelationalCausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    GroundingMode,
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.groundable_circuit import (
    GroundableCircuit,
)
from probabilistic_model.probabilistic_circuit.probabilistic_circuit_representation import (
    RepresentationType,
)
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.tensorized.factories import (
    uniform_measure_of_event,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.exceptions import UnboundedEventError
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
class UniformPriorRegistry(ModelRegistry):
    """
    A registry that knows nothing but what a statement allows: every parameter is drawn
    uniformly from where the statement's ``where`` conditions permit it.

    The model is the uniform measure of the values the conditions allow, also when they
    form several separate regions.
    """

    def get_model(
        self, parameters: ModelQueryParameters
    ) -> LayeredProbabilisticCircuit:
        """
        :param parameters: The parameters of the statement. Only a ``Match`` brings
            ``where`` conditions; the bare selection of ``average(...)`` and the
            condition of ``probability_of(...)`` bring none.
        :raises UnboundedParameterError: If the conditions leave a parameter without a
            lower or an upper bound, which they always do when there are none.
        """
        if not isinstance(parameters, UnderspecifiedParameters):
            raise UnboundedParameterError(
                parameter_name=", ".join(parameters.variables)
            )
        allowed = parameters.truncation_assignments_from_where_conditions
        if allowed is None:
            raise UnboundedParameterError(
                parameter_name=", ".join(parameters.variables)
            )
        for name, variable in parameters.variables.items():
            if variable not in allowed.variables:
                raise UnboundedParameterError(parameter_name=name)
        try:
            return uniform_measure_of_event(allowed)
        except UnboundedEventError as error:
            raise UnboundedParameterError(parameter_name=error.variable.name) from error


@dataclass
class DictRegistry(ModelRegistry):
    """
    A registry that uses a dictionary to keep all models.
    """

    models: dict[type, ProbabilisticModel]
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
    returning. The grounding comes back in the circuit type the relational circuit was
    fitted in, and a causal query is wrapped in that same type.

    Only supports :class:`~krrood.parametrization.parameterizer.UnderspecifiedParameters`
    (i.e. a ``Match``, directly or wrapped by ``distribution_of(...)``): grounding needs
    the full match statement, which ``average``'s/``probability_of``'s lighter
    parameter classes don't carry -- resolving one of those raises
    :class:`~krrood.parametrization.exceptions.RelationalCircuitRegistryRequiresMatch`
    rather than failing on a missing attribute.

    Every query grounds the same way, with :attr:`grounding_mode`: undetermined
    aggregation latents are always retained, never integrated out. Whether the query
    also declares a ``cause`` marker only decides what happens to the grounded circuit
    afterward -- a postprocessing step, not a different way of grounding. A causal
    query is wrapped as a verified
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`
    registering the marked variables; any accompanying ``causes_effect``/``confounder``
    markers are read later, by ``ProbabilisticBackend``, once it has this
    ``CausalCircuit`` in hand. A non-causal query gets the grounded circuit back
    unchanged, retained latents included.
    """

    relational_probabilistic_circuit: RelationalProbabilisticCircuit
    """
    The trained relational probabilistic circuit to ground.
    """

    grounding_mode: GroundingMode = GroundingMode.SAMPLED
    """
    How undetermined aggregation latents are represented during grounding.

    See
    :class:`~probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit.GroundingMode`.
    """

    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
        if not isinstance(parameters, UnderspecifiedParameters):
            raise RelationalCircuitRegistryRequiresMatch(parameters)
        grounded = self._ground_and_rename(parameters)
        if not parameters.search_cause_variables:
            return grounded
        return self._as_causal_circuit(grounded, parameters)

    def _ground_and_rename(
        self, parameters: UnderspecifiedParameters
    ) -> GroundableCircuit:
        """
        Ground the relational circuit for ``parameters.statement`` and align its
        variable names to the ``UnderspecifiedParameters`` convention.

        :param parameters: The parameters extracted from the queried statement.
        :return: The grounded, renamed circuit.
        """
        grounded = self.relational_probabilistic_circuit.ground(
            parameters.statement, self.grounding_mode
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
        return grounded

    @staticmethod
    def _as_causal_circuit(
        grounded: RepresentationType, parameters: UnderspecifiedParameters
    ) -> CausalCircuit[RepresentationType]:
        """
        Wrap an already-grounded, already-renamed circuit as a verified
        ``CausalCircuit`` registering the query's declared cause, effect and confounder
        variables, trimmed down to just those.

        Trimming matters for cost, not correctness (see
        ``RelationalCausalCircuit.from_grounded_circuit``): a class circuit fitted
        over many unrelated scalar and exchangeable variables otherwise makes
        ``verify_support_determinism`` compute over all of them, not just the ones
        this query actually declared. Confounders are trimmed in alongside the cause
        and effect, not just registered later, so a caller's own
        ``backdoor_adjustment(..., adjustment_variables=...)`` call still finds them
        on the returned circuit.

        :param grounded: The grounded, renamed circuit to wrap.
        :param parameters: The parameters extracted from the queried statement, carrying
            the declared cause, effect and confounder variables.
        :return: A verified, support-deterministic ``CausalCircuit``.
        """
        return RelationalCausalCircuit().from_grounded_circuit(
            grounded,
            parameters.search_cause_variables,
            list(parameters.effect_variables_from_causes_effect),
            adjustment_variables=parameters.search_confounder_variables,
            trim_to_registered_variables=True,
        )


@dataclass
class CausalCircuitRegistry(ModelRegistry):
    """
    A registry that maps target classes directly to pre-built causal circuits, so a
    ``cause``/``causes_effect()`` query can be routed through that circuit's
    ``backdoor_adjustment`` method.

    See
    :class:`~probabilistic_model.probabilistic_circuit.causal.causal_circuit.CausalCircuit`.
    """

    circuits: dict[type, CausalCircuit]
    """
    A dictionary that maps classes to pre-built causal circuits.
    """

    def get_model(self, parameters: ModelQueryParameters) -> ProbabilisticModel:
        return self.circuits[parameters.queried_class]
