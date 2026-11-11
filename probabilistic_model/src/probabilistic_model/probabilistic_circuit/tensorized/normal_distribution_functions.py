"""
The functions of the normal distribution that the numeric queries of the layers need and
the Python Array API standard does not define, for every supported array backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType

import jax.scipy.special
import numpy as np
import scipy.special
from array_api_compat import array_namespace, is_jax_namespace, is_numpy_namespace
from krrood.ormatic.utils import classproperty
from typing_extensions import Any

from probabilistic_model.probabilistic_circuit.tensorized.exceptions import (
    UnsupportedArrayBackendError,
)

# %% normal distribution functions of one array backend


@dataclass
class NormalDistributionFunctions(ABC):
    """
    The cumulative distribution function and the log-density of normal distributions,
    computed by one array backend.

    Every function takes and returns arrays of that backend and broadcasts its
    arguments.
    """

    namespace: ModuleType
    """
    The Array API namespace of the backend.
    """

    @classmethod
    def from_arrays(cls, *arrays: Any) -> NormalDistributionFunctions:
        """
        :param arrays: Arrays of one backend.
        :return: The normal distribution functions of that backend.
        :raises UnsupportedArrayBackendError: If no subclass computes them for that
            backend.
        """
        namespace = array_namespace(*arrays)
        for subclass in cls.__subclasses__():
            if subclass.supports(namespace):
                return subclass(namespace)
        raise UnsupportedArrayBackendError(namespace.__name__)

    @classmethod
    @abstractmethod
    def supports(cls, namespace: ModuleType) -> bool:
        """
        :param namespace: An Array API namespace.
        :return: Whether this class computes the functions for that backend.
        """
        raise NotImplementedError

    @abstractmethod
    def standard_cumulative_distribution(self, values: Any) -> Any:
        """
        :param values: Any array.
        :return: The cumulative distribution function of the standard normal
            distribution at every value.
        """
        raise NotImplementedError

    @classproperty
    def log_normalization_constant(cls) -> float:
        """
        The logarithm of the normalization constant of the standard normal density,
        ``log(sqrt(2 pi))``.
        """
        # spelled as scipy.stats.norm spells it, so that the densities agree bit for bit
        return float(np.log(np.sqrt(2 * np.pi)))

    def cumulative_distribution(self, values: Any, location: Any, scale: Any) -> Any:
        """
        :param values: The values.
        :param location: The mean of the normal distribution.
        :param scale: The standard deviation of the normal distribution.
        :return: The cumulative distribution function of the normal distribution at
            every value.
        """
        return self.standard_cumulative_distribution((values - location) / scale)

    def log_density(self, values: Any, location: Any, scale: Any) -> Any:
        """
        :param values: The values.
        :param location: The mean of the normal distribution.
        :param scale: The standard deviation of the normal distribution.
        :return: The logarithm of the density of the normal distribution at every
            value.
        """
        standardized = (values - location) / scale
        return (
            -(standardized**2) / 2.0 - self.log_normalization_constant
        ) - self.namespace.log(scale)


# %% backends


@dataclass
class ScipyNormalDistributionFunctions(NormalDistributionFunctions):
    """
    The normal distribution functions of numpy arrays, computed by scipy.
    """

    @classmethod
    def supports(cls, namespace: ModuleType) -> bool:
        return is_numpy_namespace(namespace)

    def standard_cumulative_distribution(self, values: Any) -> Any:
        return scipy.special.ndtr(values)


@dataclass
class JaxNormalDistributionFunctions(NormalDistributionFunctions):
    """
    The normal distribution functions of jax arrays, computed by :mod:`jax.scipy`.
    """

    @classmethod
    def supports(cls, namespace: ModuleType) -> bool:
        return is_jax_namespace(namespace)

    def standard_cumulative_distribution(self, values: Any) -> Any:
        return jax.scipy.special.ndtr(values)
