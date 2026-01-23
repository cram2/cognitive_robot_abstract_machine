from enum import IntEnum

import numpy as np
import pandas as pd
from random_events.set import SetElement, Set
from random_events.variable import Variable, Continuous as REContinuous, Integer as REInteger, Symbolic
from typing_extensions import Self, List, Any, Dict


def infer_variables_from_dataframe(data: pd.DataFrame, scale_continuous_types: bool = False,
                                   min_likelihood_improvement: float = 0.1, min_samples_per_quantile: int = 10) \
        -> List[Variable]:
    """
    Infer the variables from a dataframe.
    The variables are inferred by the column names and types of the dataframe.

    :param data: The dataframe to infer the variables from.
    :param scale_continuous_types: Whether to scale numeric types.
    :param min_likelihood_improvement: The minimum likelihood improvement passed to the Continuous Variables.
    :param min_samples_per_quantile: The minimum number of samples per quantile passed to the Continuous Variables.
    :return: The inferred variables.
    """
    result = []

    for column, datatype in zip(data.columns, data.dtypes):

        unique_values = data[column].unique()

        # handle continuous variables
        if np.issubdtype(datatype, np.number) and datatype != int:

            if len(unique_values) == 1:
                minimal_distance_between_values = 1.
            else:
                minimal_distance_between_values = np.diff(np.sort(unique_values)).min()
            mean = data[column].mean()
            std = data[column].std()

            # select the correct class type
            if scale_continuous_types:
                variable_class = ScaledContinuous

            else:
                variable_class = Continuous

            variable = variable_class(column, mean, std, minimal_distance_between_values, min_likelihood_improvement,
                                      min_samples_per_quantile)

        # handle discrete variables
        elif datatype in [object, int]:
            if datatype == int:
                mean = data[column].mean()
                std = data[column].std()
                variable = Integer(column, mean, std)
            elif datatype == object:
                unique_values = data[column].unique()
                # unique_values.sort()
                # enum = IntEnum(column, {value: index for index, value in enumerate(unique_values)})
                variable = Symbolic(column, Set.from_iterable(unique_values))
            else:
                raise ValueError(f"Datatype {datatype} of column {column} is not supported.")

        else:
            raise ValueError(f"Datatype {datatype} of column {column} is not supported.")

        result.append(variable)

    return result


class Integer(REInteger):
    mean: float
    """
    Mean of the random variable.
    """

    std: float
    """
    Standard Deviation of the random variable.
    """

    def __init__(self, name: str, mean, std):
        super().__init__(name)
        self.mean = mean
        self.std = std

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["mean"] = self.mean
        result["std"] = self.std
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(name=data["name"], mean=data["mean"], std=data["std"])

    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.name)


class Continuous(REContinuous):
    """
    Base class for continuous variables in JPTs. This class does not standardize the data,
    but needs to know mean and std anyway.
    """

    mean: float
    """
    Mean of the random variable.
    """

    std: float
    """
    Standard Deviation of the random variable.
    """

    minimal_distance: float
    """
    The minimal distance between two values of the variable.
    """

    min_likelihood_improvement: float
    """
    The minimum likelihood improvement passed to the Nyga Distributions.
    """

    min_samples_per_quantile: int
    """
    The minimum number of samples per quantile passed to the Nyga Distributions.
    """

    def __init__(self, name: str, mean: float, std: float, minimal_distance: float = 1.,
                 min_likelihood_improvement: float = 0.1, min_samples_per_quantile: int = 10):
        super().__init__(name)
        self.mean = mean
        self.std = std
        self.minimal_distance = minimal_distance
        self.min_likelihood_improvement = min_likelihood_improvement
        self.min_samples_per_quantile = min_samples_per_quantile

    def to_json(self) -> Dict[str, Any]:
        result = super().to_json()
        result["mean"] = self.mean
        result["std"] = self.std
        result["minimal_distance"] = self.minimal_distance
        result["min_likelihood_improvement"] = self.min_likelihood_improvement
        result["min_samples_per_quantile"] = self.min_samples_per_quantile
        return result

    @classmethod
    def _from_json(cls, data: Dict[str, Any]) -> Self:
        return cls(name=data["name"], mean=data["mean"], std=data["std"], minimal_distance=data["minimal_distance"],
                   min_likelihood_improvement=data["min_likelihood_improvement"],
                   min_samples_per_quantile=data["min_samples_per_quantile"])


class ScaledContinuous(Continuous):
    """
    A continuous variable that is standardized.
    """

    def __init__(self, name: str, mean: float, std: float, minimal_distance: float = 1.,
                 min_likelihood_improvement: float = 0.1, min_samples_per_quantile: int = 10):
        super().__init__(name, mean, std, minimal_distance, min_likelihood_improvement, min_samples_per_quantile)

    def encode(self, value: Any):
        return (value - self.mean) / self.std

    def decode(self, value: float) -> float:
        return value * self.std + self.mean

    def __str__(self):
        return f"{self.__class__.__name__}({self.name}, {self.mean}, {self.std}, {self.minimal_distance})"
