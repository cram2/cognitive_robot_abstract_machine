"""
Reading from trials what made them succeed.
"""

from __future__ import annotations

from dataclasses import dataclass

from functools import cached_property

import numpy as np
import pandas as pd
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from scipy.stats import chi2_contingency
from typing_extensions import Any

from krrood.bounds import Bounds
from krrood.entity_query_language.factories import a, cause, confounder
from krrood.parametrization.causal_search import ScoredIntervention
from krrood.parametrization.model_registries import CausalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from krrood.parametrization.probabilistic_backend import ProbabilisticBackend
from krrood.parametrization.trials.exceptions import NoTrials
from krrood.parametrization.trials.records import (
    InformedField,
    Trial,
    TrialParameters,
)

# %% success along one value


@dataclass(kw_only=True)
class SuccessRateInRange:
    """
    How often the trials whose value lay in a range succeeded.
    """

    bounds: Bounds[float]
    """
    The range of the value; the upper bound is included only for the last range.
    """

    number_of_trials: int
    """
    How many trials had the value in the range.
    """

    success_rate: float
    """
    The fraction of those trials that succeeded.
    """


@dataclass(kw_only=True)
class EffectOnSuccess:
    """
    How the success rate of trials changes along one of their parameters or
    measurements.
    """

    attribute_path: str
    """
    The parameter or measurement, as :meth:`Trial.values_by_attribute_path` names it.
    """

    ranges: list[SuccessRateInRange]
    """
    The success rate in each range of the value, from its lowest values up.
    """

    @property
    def success_rate_difference(self) -> float:
        """
        How far apart the highest and the lowest success rate of the ranges are.
        """
        rates = [success.success_rate for success in self.ranges]
        return max(rates) - min(rates)

    @property
    def p_value(self) -> float | None:
        """
        How likely success rates at least this different arise by chance where the value
        has no effect, by a chi-square test of the ranges against the outcome; ``None``
        when every trial came out the same, so no effect can be told.
        """
        successes = [
            round(success.success_rate * success.number_of_trials)
            for success in self.ranges
        ]
        failures = [
            success.number_of_trials - successful
            for success, successful in zip(self.ranges, successes)
        ]
        if sum(successes) == 0 or sum(failures) == 0:
            return None
        return float(chi2_contingency([successes, failures]).pvalue)


@dataclass(kw_only=True)
class SuccessRateOfParameters:
    """
    How often the trials of the same parameters succeeded.
    """

    parameters: TrialParameters
    """
    The parameters the trials shared.
    """

    number_of_trials: int
    """
    How many trials had these parameters.
    """

    success_rate: float
    """
    The fraction of those trials that succeeded.
    """


@dataclass(kw_only=True)
class AdoptedValue:
    """
    The value of a parameter that worked best, to become the default of the field of an
    action it informs.
    """

    informed_field: InformedField
    """
    The field whose default the parameter informs.
    """

    value: float
    """
    The value of the parameter in the parameters with the highest success rate.
    """


# %% what made trials succeed


@dataclass(kw_only=True)
class TrialAnalysis:
    """
    Reads from trials what made them succeed.
    """

    trials: list[Trial]
    """
    The trials analysed, all of the same type.
    """

    number_of_ranges: int = 4
    """
    Into how many ranges of equally many trials each value is split.
    """

    def __post_init__(self) -> None:
        """
        :raises NoTrials: If there are no trials.
        """
        if not self.trials:
            raise NoTrials()

    @property
    def success_rate(self) -> float:
        """
        The fraction of all trials that succeeded.
        """
        return float(np.mean([trial.succeeded for trial in self.trials]))

    def effects_on_success(self) -> list[EffectOnSuccess]:
        """
        :return: The effect of every parameter and measurement, the one along which the
            success rate changes most first.
        """
        values = [trial.values_by_attribute_path() for trial in self.trials]
        succeeded = np.array([trial.succeeded for trial in self.trials])
        effects = [
            EffectOnSuccess(
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
        :param values: One value of each trial.
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

    def success_rates_of_parameters(self) -> list[SuccessRateOfParameters]:
        """
        :return: The success rate of every distinct parameters, the highest first.
        """
        trials_by_parameters: dict[tuple[tuple[str, float], ...], list[Trial]] = {}
        for trial in self.trials:
            values = tuple(trial.parameters.values_by_attribute_path().items())
            trials_by_parameters.setdefault(values, []).append(trial)
        rates = [
            SuccessRateOfParameters(
                parameters=trials[0].parameters,
                number_of_trials=len(trials),
                success_rate=float(np.mean([trial.succeeded for trial in trials])),
            )
            for trials in trials_by_parameters.values()
        ]
        return sorted(rates, key=lambda rate: rate.success_rate, reverse=True)

    def adopted_values(self) -> list[AdoptedValue]:
        """
        :return: For every field the parameters inform, the value of the parameter in
            the parameters with the highest success rate.
        """
        [best, *_] = self.success_rates_of_parameters()
        values = best.parameters.values_by_attribute_path()
        return [
            AdoptedValue(
                informed_field=informed_field,
                value=values[informed_field.attribute_path],
            )
            for informed_field in type(best.parameters).informed_fields
        ]


# %% what causes trials to succeed


@dataclass(kw_only=True)
class CauseOfSuccess:
    """
    How likely trials succeed when one of their parameters or measurements is set to the
    values that make success most likely.
    """

    attribute_path: str
    """
    The parameter or measurement, as :meth:`Trial.values_by_attribute_path` names it.
    """

    region: Bounds[float]
    """
    The smallest range that holds the values that make success most likely.
    """

    success_probability: float
    """
    The probability that a trial succeeds when the value is set to the region.
    """


@dataclass(kw_only=True)
class CausalTrialAnalysis:
    """
    Reads from trials what causes them to succeed, through a joint probability tree
    fitted to their parameters, measurements and outcomes.

    Parameters are drawn independently, so they are searched as causes as they are; a
    measurement is searched adjusted for the parameters, which come before it.
    """

    trials: list[Trial]
    """
    The trials analysed, all of the same type.
    """

    minimum_trials_per_leaf: int = 20
    """
    How many trials each leaf of the tree holds at least, so that no success probability
    rests on a handful of trials.
    """

    def __post_init__(self) -> None:
        """
        :raises NoTrials: If there are no trials.
        """
        if not self.trials:
            raise NoTrials()

    @cached_property
    def trial_type(self) -> type[Trial]:
        """
        The type of the trials.
        """
        return type(self.trials[0])

    @cached_property
    def rows(self) -> pd.DataFrame:
        """
        Every trial as a row of its numbers and its outcome, each column named like the
        variable a query of the trials has for it.
        """
        prefix = self.trial_type.__name__
        return pd.DataFrame(
            [
                {
                    **{
                        f"{prefix}.{path}": value
                        for path, value in trial.values_by_attribute_path().items()
                    },
                    f"{prefix}.succeeded": trial.succeeded,
                }
                for trial in self.trials
            ]
        )

    def fitted_circuit(self) -> ProbabilisticCircuit:
        """
        :return: A joint probability tree of the parameters, measurements and outcomes
            of the trials.
        """
        return JointProbabilityTree(
            minimum_samples_per_leaf=self.minimum_trials_per_leaf
        ).fit(
            self.rows,
            infer_variables_from_dataframe(self.rows),
        )

    def causes_of_success(self) -> list[CauseOfSuccess]:
        """
        :return: Every parameter and measurement as a cause of success, the one that
            makes success most likely first.
        """
        causes = self.ranked_causes(parameters_marker=cause, measurements_marker=...)
        if self.trials[0].measurements.values_by_attribute_path():
            causes += self.ranked_causes(
                parameters_marker=confounder, measurements_marker=cause
            )
        return sorted(causes, key=lambda found: found.success_probability, reverse=True)

    def ranked_causes(
        self, *, parameters_marker: Any, measurements_marker: Any
    ) -> list[CauseOfSuccess]:
        """
        :param parameters_marker: What every parameter is matched with in the query.
        :param measurements_marker: What every measurement is matched with in the query.
        :return: Every number marked ``cause`` as a cause of success.
        """
        example = self.trials[0]
        query = a(self.trial_type)(
            parameters=type(example.parameters).pattern(marker=parameters_marker),
            measurements=type(example.measurements).pattern(marker=measurements_marker),
            succeeded=...,
        )
        query.causes_effect(query.succeeded == True)  # noqa: E712
        circuit = self.fitted_circuit()
        circuit_variables = {variable.name: variable for variable in circuit.variables}
        cause_variables = [
            circuit_variables[variable.name]
            for variable in UnderspecifiedParameters(
                statement=query
            ).search_cause_variables
        ]
        causal_circuit = CausalCircuit.from_probabilistic_circuit(
            circuit,
            MarginalDeterminismTreeNode.from_variable_order(cause_variables),
            cause_variables,
            [circuit_variables[f"{self.trial_type.__name__}.succeeded"]],
        )
        backend = ProbabilisticBackend(
            model_registry=CausalCircuitRegistry(
                circuits={self.trial_type: causal_circuit}
            )
        )
        return [
            self.cause_of_success(scored=scored)
            for scored in backend.rank_causes(expression=query)
        ]

    def cause_of_success(self, *, scored: ScoredIntervention) -> CauseOfSuccess:
        """
        :param scored: The best region of one cause, as the causal search found it.
        :return: The cause of success it tells of.
        """
        support = scored.narrowed_circuit.support.marginal([scored.cause_variable])
        intervals = [
            interval
            for simple_event in support.simple_sets
            for interval in simple_event[scored.cause_variable].simple_sets
        ]
        return CauseOfSuccess(
            attribute_path=scored.cause_variable.name.removeprefix(
                f"{self.trial_type.__name__}."
            ),
            region=Bounds(
                lower=min(interval.lower for interval in intervals),
                upper=max(interval.upper for interval in intervals),
            ),
            success_probability=scored.effect_probability_given_region,
        )
