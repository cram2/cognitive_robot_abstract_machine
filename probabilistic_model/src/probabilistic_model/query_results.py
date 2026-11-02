from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from random_events.product_algebra import Event
from random_events.sigma_algebra import AbstractCompositeSet
from typing_extensions import Generic, Self, TypeVar

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

DistributionType = TypeVar("DistributionType")
"""
What represents the distribution a query leaves behind, such as a model or the root
layer of a circuit.
"""

# %% modes


@dataclass
class AbstractMode(ABC):
    """
    The most likely events of a model.
    """

    event: Event
    """
    The set of most likely points.
    """


@dataclass
class Mode(AbstractMode):
    """
    The most likely events of a model and their likelihood.
    """

    likelihood: float
    """
    The likelihood of every point in :attr:`event`.
    """


@dataclass
class LogMode(AbstractMode):
    """
    The most likely events of a model and their log-likelihood.
    """

    log_likelihood: float
    """
    The log-likelihood of every point in :attr:`event`.
    """

    def exponentiated(self) -> Mode:
        """
        :return: This mode with its likelihood instead of its log-likelihood.
        """
        return Mode(self.event, float(np.exp(self.log_likelihood)))


@dataclass
class UnivariateLogMode:
    """
    The most likely values of a univariate distribution and their log-likelihood.
    """

    assignment: AbstractCompositeSet
    """
    The set of most likely values of the variable.
    """

    log_likelihood: float
    """
    The log-likelihood of every value in :attr:`assignment`.
    """


# %% posteriors


@dataclass
class AbstractPosterior(Generic[DistributionType], SubClassSafeGeneric, ABC):
    """
    A model restricted to evidence, P(* | evidence), and how likely that evidence is.
    """

    distribution: DistributionType | None
    """
    The distribution given the evidence, or *None* if the evidence is impossible.
    """

    @classmethod
    @abstractmethod
    def impossible(cls) -> Self:
        """
        :return: The posterior given evidence that cannot occur.
        """


# %% truncations


@dataclass
class Truncation(AbstractPosterior[DistributionType]):
    """
    A model truncated to an event, P(* | event), and the probability of that event.
    """

    probability: float
    """
    The probability of the event.
    """

    @classmethod
    def impossible(cls) -> Self:
        return cls(None, 0.0)


@dataclass
class LogTruncation(AbstractPosterior[DistributionType]):
    """
    A model truncated to an event, P(* | event), and the log-probability of that event.
    """

    log_probability: float
    """
    The log-probability of the event.
    """

    @classmethod
    def impossible(cls) -> Self:
        return cls(None, -np.inf)

    def exponentiated(self) -> Truncation[DistributionType]:
        """
        :return: This truncation with the probability instead of the log-probability.
        """
        return Truncation(self.distribution, float(np.exp(self.log_probability)))


# %% conditionings


@dataclass
class Conditioning(AbstractPosterior[DistributionType]):
    """
    A model conditioned on a partial point, P(* | point), and the likelihood of that
    point.
    """

    likelihood: float
    """
    The likelihood of the point.
    """

    @classmethod
    def impossible(cls) -> Self:
        return cls(None, 0.0)


@dataclass
class LogConditioning(AbstractPosterior[DistributionType]):
    """
    A model conditioned on a partial point, P(* | point), and the log-likelihood of that
    point.
    """

    log_likelihood: float
    """
    The log-likelihood of the point.
    """

    @classmethod
    def impossible(cls) -> Self:
        return cls(None, -np.inf)

    def exponentiated(self) -> Conditioning[DistributionType]:
        """
        :return: This conditioning with the likelihood instead of the log-likelihood.
        """
        return Conditioning(self.distribution, float(np.exp(self.log_likelihood)))
