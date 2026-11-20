"""
The values the parameters of a trial may take, and the models that draw them.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from typing_extensions import TYPE_CHECKING, Generic, Self

from krrood.bounds import Bounds
from krrood.entity_query_language.factories import ConditionType
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.exceptions import UnboundedParameterError
from krrood.parametrization.model_registries import ModelRegistry
from krrood.parametrization.parameterizer import ModelQueryParameters
from krrood.parametrization.probabilistic_backend import ProbabilisticBackend
from krrood.parametrization.trials.exceptions import (
    NoSuccessfulTrials,
    ParameterWithoutCentre,
)
from krrood.parametrization.trials.records import TParameters, Trial, TrialParameters
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

if TYPE_CHECKING:
    from krrood.entity_query_language.core.mapped_variable import Attribute

# %% the values the parameters may take


@dataclass(kw_only=True)
class ParameterSpace(Generic[TParameters], SubClassSafeGeneric, ABC):
    """
    The values the parameters of a trial may take, as a statement a model answers.
    """

    @abstractmethod
    def conditions(self, *, parameters: Match[TParameters]) -> list[ConditionType]:
        """
        :param parameters: The pattern of the parameters, to reach every parameter
            from.
        :return: The conditions every draw meets, such as the bounds of each parameter.

        .. note::
            Reach a nested parameter through ``parameters``, such as
            ``parameters.offset.forward``, so the model's backend can bound it.
        """

    def pattern(self) -> Match[TParameters]:
        """
        :return: The pattern of the parameters, with every number left to be drawn.
        :raises FieldOfUnsupportedType: If a field is neither a number nor a
            dataclass.
        """
        return self.get_type_of_generic_parameter(parameter=TParameters).pattern()

    def statement(self) -> Match[TParameters]:
        """
        :return: The statement of every parameters this space allows.
        """
        parameters = self.pattern()
        return parameters.where(*self.conditions(parameters=parameters))

    def accepts(self, *, parameters: TParameters) -> bool:
        """
        :param parameters: Drawn parameters.
        :return: Whether they are tried; by default every draw is.
        """
        return True

    def drawn_parameters(
        self, *, number_of_draws: int, model_registry: ModelRegistry
    ) -> list[TParameters]:
        """
        :param number_of_draws: How many parameters to draw.
        :param model_registry: Answers the statement of :meth:`statement`.
        :return: The draws :meth:`accepts`; fewer than asked for where some are not.
        """
        drawn = (
            self.statement()
            .limit(n=number_of_draws)
            .evaluate(backend=ProbabilisticBackend(model_registry=model_registry))
        )
        return [
            parameters for parameters in drawn if self.accepts(parameters=parameters)
        ]

    @staticmethod
    def conditions_within(
        *, value: Attribute, bounds: Bounds[float]
    ) -> list[ConditionType]:
        """
        :param value: A parameter, reached from the pattern of the parameters.
        :param bounds: Where it lies; the upper bound is excluded.
        :return: The conditions that keep the parameter within the bounds.
        """
        return [value >= bounds.lower, value < bounds.upper]


# %% drawing near parameters that worked


@dataclass(kw_only=True)
class NearCentresRegistry(ModelRegistry):
    """
    Answers a statement of trial parameters with an equal mixture of one Gaussian around
    each of :attr:`centres`, whose standard deviation along every parameter is
    :attr:`spread` times the range the statement allows that parameter.
    """

    centres: list[TrialParameters]
    """
    The parameters the draws are biased towards, such as those of successful trials.
    """

    spread: float = 0.1
    """
    The standard deviation of each Gaussian along a parameter, as a fraction of the
    range the statement allows the parameter.
    """

    @classmethod
    def from_successful_trials(
        cls, *, trials: list[Trial], spread: float = 0.1
    ) -> Self:
        """
        :param trials: The trials of a campaign.
        :param spread: The spread of the registry, as in :attr:`spread`.
        :return: A registry that draws near the parameters of every trial that
            succeeded, so that the next campaign tries more like them.
        :raises NoSuccessfulTrials: If no trial succeeded.
        """
        centres = [trial.parameters for trial in trials if trial.succeeded]
        if not centres:
            raise NoSuccessfulTrials(number_of_trials=len(trials))
        return cls(centres=centres, spread=spread)

    def get_model(self, *, parameters: ModelQueryParameters) -> ProbabilisticCircuit:
        """
        :raises UnboundedParameterError: If the statement leaves a parameter without a
            lower or an upper bound.
        :raises ParameterWithoutCentre: If a parameter of the statement is not one of
            :meth:`TrialParameters.values_by_attribute_path`.
        """
        allowed = parameters.truncation_assignments_from_where_conditions
        if allowed is None:
            raise UnboundedParameterError(
                parameter_name=", ".join(parameters.variables)
            )
        [box] = allowed.simple_sets
        prefix = f"{parameters.queried_class.__name__}."
        circuit = ProbabilisticCircuit()
        mixture = SumUnit(probabilistic_circuit=circuit)
        for centre in self.centres:
            values = centre.values_by_attribute_path()
            component = ProductUnit(probabilistic_circuit=circuit)
            for name, variable in parameters.variables.items():
                path = name.removeprefix(prefix)
                if path not in values:
                    raise ParameterWithoutCentre(parameter_name=name)
                if variable not in box.keys():
                    raise UnboundedParameterError(parameter_name=name)
                [interval] = box[variable].simple_sets
                component.add_subcircuit(
                    leaf(
                        GaussianDistribution(
                            variable=variable,
                            location=values[path],
                            scale=self.spread * (interval.upper - interval.lower),
                        ),
                        circuit,
                    )
                )
            mixture.add_subcircuit(component, -math.log(len(self.centres)))
        return circuit


# %% the parameters a campaign tries


@dataclass(kw_only=True)
class ParameterSource(Generic[TParameters], SubClassSafeGeneric, ABC):
    """
    Gives the parameters a campaign tries.
    """

    @abstractmethod
    def parameters_to_try(self) -> list[TParameters]:
        """
        :return: The parameters to try, each once, in the order they are tried.
        """


@dataclass(kw_only=True)
class DrawnParameters(ParameterSource[TParameters]):
    """
    Parameters a model draws from a space.
    """

    parameter_space: ParameterSpace[TParameters]
    """
    What the parameters are drawn from.
    """

    model_registry: ModelRegistry
    """
    Answers the statement of :attr:`parameter_space`.
    """

    number_of_draws: int = 10
    """
    How many parameters to draw; fewer are tried where the space does not accept some.
    """

    draw_seed: int = 0
    """
    Seeds the draws, so the same source draws the same parameters.
    """

    def parameters_to_try(self) -> list[TParameters]:
        # the probabilistic models draw from numpy's global random state
        np.random.seed(self.draw_seed)
        return self.parameter_space.drawn_parameters(
            number_of_draws=self.number_of_draws, model_registry=self.model_registry
        )


@dataclass(kw_only=True)
class GivenParameters(ParameterSource[TParameters]):
    """
    Parameters chosen beforehand, such as the most promising ones of an earlier
    campaign, to retry them.
    """

    parameters: list[TParameters]
    """
    The parameters to try, in order.
    """

    def parameters_to_try(self) -> list[TParameters]:
        return self.parameters
