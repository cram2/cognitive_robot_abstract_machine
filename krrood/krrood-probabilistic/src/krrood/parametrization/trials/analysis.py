"""
Reading from trials what made them succeed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from krrood.bounds import Bounds
from krrood.parametrization.trials.exceptions import NoTrials
from krrood.parametrization.trials.records import Trial, TrialParameters

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

    def successful_parameters(self) -> list[TrialParameters]:
        """
        :return: The parameters of every trial that succeeded, such as the centres to
            draw the next trials near.
        """
        return [trial.parameters for trial in self.trials if trial.succeeded]
