"""
Finding out by trials what makes a behaviour succeed: parameters drawn from a space by a
model are tried from the same starting state, many at once, and the outcomes tell which
parameters matter and which values to draw the next trials near.
"""

from __future__ import annotations

import dataclasses
import math
import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from typing_extensions import TYPE_CHECKING, Any, Generic, TypeVar, get_type_hints

from krrood.bounds import Bounds
from krrood.entity_query_language.factories import ConditionType, a
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.exceptions import (
    NotEnoughMemoryForWorkers,
    NoTrialOutcomes,
    ParameterOfUnsupportedType,
    ParameterWithoutCentre,
    UnboundedParameterError,
)
from krrood.parametrization.model_registries import ModelRegistry
from krrood.parametrization.parameterizer import ModelQueryParameters
from krrood.parametrization.probabilistic_backend import ProbabilisticBackend
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

if TYPE_CHECKING:
    from krrood.entity_query_language.core.mapped_variable import Attribute

# %% the parameters of a trial


@dataclass(kw_only=True)
class TrialParameters:
    """
    Everything one trial of a behaviour varies by: numbers, directly or in nested
    dataclasses of numbers.
    """

    def values_by_attribute_path(self) -> dict[str, float]:
        """
        :return: Every number, keyed by the path of attributes that leads to it from
            this object, such as ``stand_offset.forward``.
        :raises ParameterOfUnsupportedType: If a field holds neither a number nor a
            dataclass.
        """
        return self.numbers_below(values=dataclasses.asdict(self), prefix="")

    @classmethod
    def numbers_below(cls, *, values: dict[str, Any], prefix: str) -> dict[str, float]:
        """
        :param values: The fields of the parameters or of a dataclass nested in them, by
            name, with nested dataclasses as dictionaries of their fields.
        :param prefix: The path of attributes that leads to the fields, ending in a dot,
            or empty for the parameters themselves.
        :return: Every number among the fields, keyed by its path of attributes.
        :raises ParameterOfUnsupportedType: If a field holds neither a number nor a
            dataclass.
        """
        numbers = {}
        for name, value in values.items():
            path = f"{prefix}{name}"
            if isinstance(value, dict):
                numbers.update(cls.numbers_below(values=value, prefix=f"{path}."))
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ParameterOfUnsupportedType(
                    owner=cls, field_name=path, field_type=type(value)
                )
            numbers[path] = float(value)
        return numbers


TParameters = TypeVar("TParameters", bound=TrialParameters)
"""
The parameters a trial varies by.
"""

TTrial = TypeVar("TTrial")
"""
A trial, with what came of it.
"""


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
        :raises ParameterOfUnsupportedType: If a field is neither a number nor a
            dataclass.
        """
        return self.underspecified_pattern(
            parameter_type=self.get_type_of_generic_parameter(parameter=TParameters)
        )

    def underspecified_pattern(self, *, parameter_type: type) -> Match:
        """
        :param parameter_type: A dataclass of numbers, directly or nested.
        :return: The pattern of the dataclass with every number left to be drawn.
        :raises ParameterOfUnsupportedType: If a field is neither a number nor a
            dataclass.
        """
        hints = get_type_hints(parameter_type)
        assigned = {}
        for parameter_field in dataclasses.fields(parameter_type):
            hint = hints[parameter_field.name]
            if dataclasses.is_dataclass(hint):
                assigned[parameter_field.name] = self.underspecified_pattern(
                    parameter_type=hint
                )
                continue
            if hint not in (float, int):
                raise ParameterOfUnsupportedType(
                    owner=parameter_type,
                    field_name=parameter_field.name,
                    field_type=hint,
                )
            assigned[parameter_field.name] = ...
        return a(parameter_type)(**assigned)

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


# %% trying and storing


@dataclass(kw_only=True)
class TrialPerformer(Generic[TParameters, TTrial], SubClassSafeGeneric, ABC):
    """
    Tries a behaviour with given parameters, each time from the same starting state.
    """

    @abstractmethod
    def perform(self, *, parameters: TParameters) -> TTrial:
        """
        :param parameters: The parameters to try.
        :return: The trial, with what came of it.
        """


@dataclass(kw_only=True)
class TrialStore(Generic[TTrial], SubClassSafeGeneric, ABC):
    """
    Keeps trials, each as soon as it is tried.
    """

    @abstractmethod
    def add(self, *, trial: TTrial) -> None:
        """
        :param trial: The trial to keep.
        """


@dataclass(kw_only=True)
class TrialSetup(Generic[TParameters, TTrial], SubClassSafeGeneric, ABC):
    """
    Everything a worker of a campaign tries trials with; it is sent to each worker's
    process, so it holds only what can be pickled and builds the rest there.
    """

    parameter_space: ParameterSpace[TParameters]
    """
    What the parameters are drawn from.
    """

    model_registry: ModelRegistry
    """
    Answers the statement of :attr:`parameter_space`.
    """

    @abstractmethod
    def performing(self) -> AbstractContextManager[TrialPerformer[TParameters, TTrial]]:
        """
        :return: A context in which the performer can try trials, such as a running
            simulation of the starting state, closed once the worker is done.
        """

    @abstractmethod
    def create_store(self) -> TrialStore[TTrial]:
        """
        :return: Where the worker keeps its trials.
        """


# %% many trials at once


@dataclass(kw_only=True)
class TrialCampaign(Generic[TParameters, TTrial], SubClassSafeGeneric):
    """
    Tries parameters drawn by a setup's model in several worker processes at once, each
    drawing its own with a seed of its own and keeping every trial as soon as it is
    tried.
    """

    setup: TrialSetup[TParameters, TTrial]
    """
    What every worker tries its trials with.
    """

    number_of_workers: int = 1
    """
    How many processes try trials at once; a single worker tries them in this process.
    """

    trials_per_worker: int = 10
    """
    How many parameters each worker draws.
    """

    first_draw_seed: int = 0
    """
    The seed the first worker draws with; each further worker's seed is one more.
    """

    memory_per_worker: int | None = None
    """
    How much working memory a worker needs at most, in bytes; ``None`` when it is not
    checked.
    """

    def run(self) -> int:
        """
        :return: How many trials were kept.
        :raises NotEnoughMemoryForWorkers: If the workers would need more working
            memory than is free.
        """
        self.raise_unless_the_workers_fit_into_memory()
        seeds = [
            self.first_draw_seed + worker for worker in range(self.number_of_workers)
        ]
        if self.number_of_workers == 1:
            return self.run_worker(draw_seed=seeds[0])
        with ProcessPoolExecutor(
            max_workers=self.number_of_workers, mp_context=get_context("spawn")
        ) as pool:
            workers = [pool.submit(self.run_worker, draw_seed=seed) for seed in seeds]
            return sum(worker.result() for worker in workers)

    def run_worker(self, *, draw_seed: int) -> int:
        """
        :param draw_seed: Seeds the worker's draws.
        :return: How many trials the worker kept.
        """
        # the probabilistic models draw from numpy's global random state
        np.random.seed(draw_seed)
        drawn = self.setup.parameter_space.drawn_parameters(
            number_of_draws=self.trials_per_worker,
            model_registry=self.setup.model_registry,
        )
        store = self.setup.create_store()
        with self.setup.performing() as performer:
            for parameters in drawn:
                store.add(trial=performer.perform(parameters=parameters))
        return len(drawn)

    def raise_unless_the_workers_fit_into_memory(self) -> None:
        """
        :raises NotEnoughMemoryForWorkers: If :attr:`memory_per_worker` is given and the
            workers would need more working memory than is free.
        """
        if self.memory_per_worker is None:
            return
        free_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        if self.number_of_workers * self.memory_per_worker > free_memory:
            raise NotEnoughMemoryForWorkers(
                number_of_workers=self.number_of_workers,
                memory_per_worker=self.memory_per_worker,
                free_memory=free_memory,
            )


# %% what made trials succeed


@dataclass(kw_only=True)
class TrialOutcome:
    """
    The parameters of one trial and whether it succeeded.
    """

    parameters: TrialParameters
    """
    What the trial varied by.
    """

    succeeded: bool
    """
    Whether the trial did what it was tried for.
    """


@dataclass(kw_only=True)
class SuccessRateInRange:
    """
    How often the trials whose parameter lay in a range succeeded.
    """

    bounds: Bounds[float]
    """
    The range of the parameter; the upper bound is included only for the last range.
    """

    number_of_trials: int
    """
    How many trials had the parameter in the range.
    """

    success_rate: float
    """
    The fraction of those trials that succeeded.
    """


@dataclass(kw_only=True)
class ParameterEffect:
    """
    How the success rate of trials changes along one parameter.
    """

    attribute_path: str
    """
    The parameter, as :meth:`TrialParameters.values_by_attribute_path` names it.
    """

    ranges: list[SuccessRateInRange]
    """
    The success rate in each range of the parameter, from its lowest values up.
    """

    @property
    def success_rate_difference(self) -> float:
        """
        How far apart the highest and the lowest success rate of the ranges are.
        """
        rates = [success.success_rate for success in self.ranges]
        return max(rates) - min(rates)


@dataclass(kw_only=True)
class TrialAnalysis:
    """
    Reads from the outcomes of trials what made them succeed.
    """

    outcomes: list[TrialOutcome]
    """
    The outcomes analysed, all of parameters of the same type.
    """

    number_of_ranges: int = 4
    """
    Into how many ranges of equally many trials each parameter is split.
    """

    def __post_init__(self) -> None:
        """
        :raises NoTrialOutcomes: If there are no outcomes.
        """
        if not self.outcomes:
            raise NoTrialOutcomes()

    @property
    def success_rate(self) -> float:
        """
        The fraction of all trials that succeeded.
        """
        return float(np.mean([outcome.succeeded for outcome in self.outcomes]))

    def parameter_effects(self) -> list[ParameterEffect]:
        """
        :return: The effect of every parameter, the one along which the success rate
            changes most first.
        """
        values = [
            outcome.parameters.values_by_attribute_path() for outcome in self.outcomes
        ]
        succeeded = np.array([outcome.succeeded for outcome in self.outcomes])
        effects = [
            ParameterEffect(
                attribute_path=path,
                ranges=self.success_rates_in_ranges(
                    values=np.array([value[path] for value in values]),
                    succeeded=succeeded,
                ),
            )
            for path in values[0]
        ]
        return sorted(
            effects, key=lambda effect: effect.success_rate_difference, reverse=True
        )

    def success_rates_in_ranges(
        self, *, values: np.ndarray, succeeded: np.ndarray
    ) -> list[SuccessRateInRange]:
        """
        :param values: The value of one parameter in each trial.
        :param succeeded: Whether each trial succeeded.
        :return: The success rate in each of :attr:`number_of_ranges` ranges of equally
            many trials; a range no trial falls into is left out.
        """
        edges = np.quantile(values, np.linspace(0.0, 1.0, self.number_of_ranges + 1))
        rates = []
        for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            last = index == self.number_of_ranges - 1
            inside = (values >= lower) & (
                (values <= upper) if last else (values < upper)
            )
            if not inside.any():
                continue
            rates.append(
                SuccessRateInRange(
                    bounds=Bounds(lower=float(lower), upper=float(upper)),
                    number_of_trials=int(inside.sum()),
                    success_rate=float(succeeded[inside].mean()),
                )
            )
        return rates

    def successful_parameters(self) -> list[TrialParameters]:
        """
        :return: The parameters of every trial that succeeded, such as the centres to
            draw the next trials near.
        """
        return [outcome.parameters for outcome in self.outcomes if outcome.succeeded]
