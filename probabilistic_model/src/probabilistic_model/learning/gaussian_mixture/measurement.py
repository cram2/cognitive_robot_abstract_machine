"""
The description of the columns a StepMix mixture is fitted on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from random_events.variable import Continuous, Integer, Symbolic
from typing_extensions import Any, Dict, List, Optional, Union

from probabilistic_model.learning.gaussian_mixture.covariance_type import (
    CovarianceType,
)


@dataclass
class MeasurementBlock(ABC):
    """
    Columns that StepMix models with one measurement model.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        :return: The key of the block in StepMix's description and parameters.
        """

    @property
    @abstractmethod
    def model(self) -> str:
        """
        :return: StepMix's name of the measurement model.
        """

    @property
    @abstractmethod
    def number_of_columns(self) -> int:
        """
        :return: The number of columns of the block.
        """

    def to_stepmix(self) -> Dict[str, Any]:
        """
        :return: The block as StepMix describes it.
        """
        return {"model": self.model, "n_columns": self.number_of_columns}


@dataclass
class GaussianBlock(MeasurementBlock):
    """
    The continuous variables, modelled by one Gaussian per component.
    """

    variables: List[Continuous]
    """
    The variables, in the order of their columns.
    """

    covariance_type: CovarianceType
    """
    The shape of the covariances.
    """

    covariance_regularization: float
    """
    The regularization added to the diagonal of the covariances.
    """

    @property
    def name(self) -> str:
        return "continuous"

    @property
    def model(self) -> str:
        return self.covariance_type.stepmix_model

    @property
    def number_of_columns(self) -> int:
        return len(self.variables)

    def to_stepmix(self) -> Dict[str, Any]:
        return super().to_stepmix() | {"reg_covar": self.covariance_regularization}


@dataclass
class CategoricalBlock(MeasurementBlock):
    """
    One symbolic or integer variable, modelled by one categorical distribution per
    component.
    """

    variable: Union[Symbolic, Integer]
    """
    The variable, whose column holds the code of each row's outcome.
    """

    @property
    def name(self) -> str:
        return f"discrete {self.variable.name}"

    @property
    def model(self) -> str:
        return "categorical"

    @property
    def number_of_columns(self) -> int:
        return 1


@dataclass
class Measurement:
    """
    All columns of the data, as StepMix's ``measurement`` parameter describes them.
    """

    gaussian: Optional[GaussianBlock]
    """
    The block of the continuous columns, which lead the data, if there are any.
    """

    categorical: List[CategoricalBlock]
    """
    One block per discrete column, in the order of the columns.
    """

    @property
    def blocks(self) -> List[MeasurementBlock]:
        """
        :return: All blocks, in the order of their columns.
        """
        return ([self.gaussian] if self.gaussian else []) + self.categorical

    def to_stepmix(self) -> Dict[str, Dict[str, Any]]:
        """
        :return: The value of StepMix's ``measurement`` parameter.
        """
        return {block.name: block.to_stepmix() for block in self.blocks}
