from random_events.variable import Continuous, Symbolic, Integer

from probabilistic_model.distributions.distributions import (
    DiracDeltaDistribution,
    SymbolicDistribution,
    IntegerDistribution,
    UnivariateDistribution,
)
from probabilistic_model.utils import MissingDict


def make_dirac(variable, value) -> UnivariateDistribution:
    """
    Creates a Dirac distribution for a given variable and value.

    This function constructs a Dirac distribution tailored to the type of
    the input variable. It supports creating distributions for continuous,
    symbolic, and integer variables.

    :param variable: The variable for which the Dirac distribution is created.
    :param value: The value at which the Dirac distribution is centered.
    :return: The dirac-like distribution
    :raises NotImplementedError: When the variable type is unsupported.
    """
    if isinstance(variable, Continuous):
        return DiracDeltaDistribution(
            variable=variable, location=value, density_cap=1.0
        )
    elif isinstance(variable, Symbolic):
        return SymbolicDistribution(
            variable=variable, probabilities=MissingDict(float, {hash(value): 1.0})
        )
    elif isinstance(variable, Integer):
        return IntegerDistribution(
            variable=variable, probabilities=MissingDict(float, {value: 1.0})
        )
    else:
        raise NotImplementedError
