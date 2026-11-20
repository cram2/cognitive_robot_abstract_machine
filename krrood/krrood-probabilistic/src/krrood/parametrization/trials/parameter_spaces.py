"""
The values the parameters of a trial may take, and the parameters a campaign tries.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from typing_extensions import TYPE_CHECKING, Generic, get_type_hints

from krrood.bounds import Bounds
from krrood.entity_query_language.factories import ConditionType, a
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.model_registries import ModelRegistry
from krrood.parametrization.probabilistic_backend import ProbabilisticBackend
from krrood.parametrization.trials.records import (
    AdoptedValue,
    InformedField,
    TParameters,
)
from krrood.patterns.classproperty import classproperty
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
        :return: The pattern of the parameters, with every field left to be drawn.
        """
        return self.underspecified_pattern(
            parameter_type=self.get_type_of_generic_parameter(parameter=TParameters)
        )

    @classmethod
    def underspecified_pattern(cls, *, parameter_type: type) -> Match:
        """
        :param parameter_type: The parameters or a dataclass nested in them.
        :return: The pattern of the dataclass, with every field that is no dataclass
            left to be drawn.
        """
        hints = get_type_hints(parameter_type)
        return a(parameter_type)(
            **{
                parameter_field.name: (
                    cls.underspecified_pattern(
                        parameter_type=hints[parameter_field.name]
                    )
                    if dataclasses.is_dataclass(hints[parameter_field.name])
                    else ...
                )
                for parameter_field in dataclasses.fields(parameter_type)
            }
        )

    @classproperty
    def informed_fields(cls) -> list[InformedField]:
        """
        The fields of actions whose defaults the parameters inform; none by default.
        """
        return []

    def adopted_values(self, *, parameters: TParameters) -> list[AdoptedValue]:
        """
        :param parameters: Parameters of this space, such as those that worked best.
        :return: Their value for every field they inform.
        """
        return [
            AdoptedValue(
                informed_field=informed_field,
                value=informed_field.parameter.apply_mapping_on_external_root(
                    instance=parameters
                ),
            )
            for informed_field in self.informed_fields
        ]

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
