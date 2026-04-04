from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from krrood.utils import get_default_value
from random_events.variable import Continuous, Integer, Symbolic, Variable
from pandas.core.dtypes.common import is_integer_dtype, is_float_dtype, is_bool_dtype
from random_events.set import Set
from typing_extensions import List, Any


@dataclass
class AnnotatedVariable:
    """
    AnnotatedVariable is a wrapper around the variables that are used in the JPTs.
    They consist of an association object and some additional parameters.
    """

    variable: Variable
    """
    The variable that is annotated.
    """

    mean: Optional[float] = field(default=0)
    """
    Mean of the random variable.
    """

    standard_deviation: Optional[float] = field(default=1)
    """
    Standard Deviation of the random variable.
    """

    minimal_distance: Optional[float] = field(default=1.0)
    """
    The minimal distance between two values of the variable.
    """

    min_likelihood_improvement: Optional[float] = field(default=0.1)
    """
    The minimum likelihood improvement passed to the Nyga Distributions.
    """

    min_samples_per_quantile: Optional[int] = field(default=10)
    """
    The minimum number of samples per quantile passed to the Nyga Distributions.
    """

    min_impurity_improvement: Optional[float] = field(default=0)
    """
    The minimum impurity improvement for JPT learning.
    """

    def __lt__(self, other):
        return self.variable < other.variable


def infer_variables_from_dataframe(
        data: pd.DataFrame,
        minimal_distance: float = 1.0,
        min_likelihood_improvement: float = 0.1,
        min_samples_per_quantile: int = 10,
        min_impurity_improvement: float = 0,
) -> List[AnnotatedVariable]:
    """
    Infer the variables from a dataframe.
    The variables are inferred by the column names and types of the dataframe.

    :param data: The dataframe to infer the variables from.
    :param minimal_distance: The minimal distance between two values of the variable.
    :param min_likelihood_improvement: The minimum likelihood improvement passed to the Continuous Variables.
    :param min_samples_per_quantile: The minimum number of samples per quantile passed to the Continuous Variables.
    :param min_impurity_improvement: The minimum impurity improvement for JPT learning.
    :return: The inferred variables.
    """
    result = []

    for column, datatype in zip(data.columns, data.dtypes):
        domain = None
        mean = get_default_value(AnnotatedVariable, "mean")
        standard_deviation = get_default_value(AnnotatedVariable, "standard_deviation")
        if is_integer_dtype(datatype):
            variable_class = Integer
            mean = data[column].mean()
            standard_deviation = data[column].std()
        elif is_float_dtype(datatype):
            variable_class = Continuous
            mean = data[column].mean()
            standard_deviation = data[column].std()
        elif is_bool_dtype(datatype):
            variable_class = Symbolic
            domain = Set.from_iterable([True, False])
        elif data[column].dtype == object:
            unique_values = data[column].unique()
            variable_class = Symbolic
            domain = Set.from_iterable(unique_values)
        else:
            raise ValueError(f"Unsupported datatype: {datatype}")

        domain = domain or get_default_value(variable_class, "domain")
        variable = variable_class(name=column, domain=domain)
        annotated_variable = AnnotatedVariable(variable=variable,
                                               mean=mean,
                                               standard_deviation=standard_deviation,
                                               minimal_distance=minimal_distance,
                                               min_likelihood_improvement=min_likelihood_improvement,
                                               min_samples_per_quantile=min_samples_per_quantile,
                                               min_impurity_improvement=min_impurity_improvement, )

        result.append(annotated_variable)
    return result

