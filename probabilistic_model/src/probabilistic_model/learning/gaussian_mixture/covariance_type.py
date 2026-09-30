"""
The shapes a Gaussian mixture's covariances can have.
"""

from __future__ import annotations

import enum

import numpy as np
import numpy.typing as npt


class CovarianceType(enum.StrEnum):
    """
    The shape of each component's covariance, with scikit-learn's names as values.
    """

    FULL = "full"
    """
    Every component has its own covariance matrix.
    """

    TIED = "tied"
    """
    All components share one covariance matrix.
    """

    DIAGONAL = "diag"
    """
    Every component has its own diagonal covariance matrix.
    """

    SPHERICAL = "spherical"
    """
    Every component has its own single variance.
    """

    @property
    def stepmix_model(self) -> str:
        """
        :return: The name of StepMix's measurement model for this type.
        """
        return f"gaussian_{self.value}"

    def full_covariances(
        self,
        covariances: npt.NDArray,
        number_of_components: int,
        number_of_features: int,
    ) -> npt.NDArray:
        """
        :param covariances: The covariances in scikit-learn's layout for this type.
        :param number_of_components: The number of components.
        :param number_of_features: The number of continuous variables.
        :return: The covariance matrix of each component, of shape
            ``(number_of_components, number_of_features, number_of_features)``.
        """
        match self:
            case CovarianceType.FULL:
                return covariances
            case CovarianceType.TIED:
                return np.broadcast_to(
                    covariances,
                    (number_of_components, number_of_features, number_of_features),
                )
            case CovarianceType.DIAGONAL:
                return np.stack([np.diag(variances) for variances in covariances])
            case CovarianceType.SPHERICAL:
                return np.stack(
                    [np.eye(number_of_features) * variance for variance in covariances]
                )
