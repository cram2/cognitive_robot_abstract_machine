"""
Fitting a circuit that puts a kernel on every row, so that samples of it lie near the
rows it was fitted on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from random_events.variable import Continuous, Integer
from typing_extensions import Any, Iterable

from krrood.patterns.called_positionally import called_positionally
from probabilistic_model.distributions.binomial import BinomialDistribution
from probabilistic_model.distributions.distributions import (
    SymbolicDistribution,
    UnivariateDistribution,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.learning.exceptions import NoRowsToFitError
from probabilistic_model.learning.jpt.variables import (
    AnnotatedVariable,
    infer_variables_from_dataframe,
)
from probabilistic_model.learning.learning_method import LearningMethod
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.utils import MissingDict

# %% one kernel per row


@dataclass(kw_only=True)
class RowKernelMixture(LearningMethod):
    """
    Fits an equal mixture of one product of kernels per row, each centred on the row's
    value of its variable: a Gaussian for a continuous variable and a symmetric binomial
    for an integer, both :attr:`bandwidth` times the variable's annotated standard
    deviation wide, and for a symbol most of the probability on the row's symbol.
    """

    bandwidth: float | None = None
    """
    The width of every kernel as a fraction of its variable's standard deviation, and
    for a symbol the fraction of the way from the row's symbol to all symbols being
    equally likely; ``None`` for Scott's rule, the number of rows to the power of ``-1 /
    (d + 4)`` for ``d`` variables.
    """

    @called_positionally
    def fit(
        self,
        data: pd.DataFrame,
        variables: Iterable[AnnotatedVariable] | None = None,
    ) -> ProbabilisticCircuit:
        """
        Unlike :meth:`LearningMethod.fit`, the variables may be annotated from other
        rows than those fitted on, such as every trial of a campaign when fitting on the
        successful ones, so that the kernels are as wide as the data varies.

        :raises NoRowsToFitError: If the dataframe has no rows.
        """
        if data.empty:
            raise NoRowsToFitError()
        annotated = (
            list(variables)
            if variables is not None
            else infer_variables_from_dataframe(data)
        )
        bandwidth = (
            self.bandwidth
            if self.bandwidth is not None
            else len(data) ** (-1 / (len(annotated) + 4))
        )
        circuit = ProbabilisticCircuit()
        mixture = SumUnit(probabilistic_circuit=circuit)
        for _, row in data.iterrows():
            kernel = ProductUnit(probabilistic_circuit=circuit)
            for annotated_variable in annotated:
                kernel.add_subcircuit(
                    leaf(
                        self.kernel(
                            annotated_variable=annotated_variable,
                            value=row[annotated_variable.variable.name],
                            bandwidth=bandwidth,
                        ),
                        circuit,
                    )
                )
            mixture.add_subcircuit(kernel, -math.log(len(data)))
        return circuit

    @staticmethod
    def kernel(
        *, annotated_variable: AnnotatedVariable, value: Any, bandwidth: float
    ) -> UnivariateDistribution:
        """
        :param annotated_variable: The variable of the kernel, with its standard
            deviation.
        :param value: The row's value of the variable.
        :param bandwidth: The width of the kernel, as in :attr:`bandwidth`.
        :return: The kernel of the variable centred on the value.
        """
        variable = annotated_variable.variable
        if isinstance(variable, Continuous):
            return GaussianDistribution(
                variable=variable,
                location=float(value),
                scale=bandwidth * annotated_variable.standard_deviation,
            )
        if isinstance(variable, Integer):
            return BinomialDistribution.from_location(
                variable=variable,
                location=int(value),
                standard_deviation=bandwidth * annotated_variable.standard_deviation,
            )
        symbols = list(variable.domain.simple_sets)
        spread = bandwidth * (len(symbols) - 1) / len(symbols)
        return SymbolicDistribution(
            variable=variable,
            probabilities=MissingDict(
                float,
                {
                    hash(symbol): (
                        1 - spread
                        if symbol.element == value
                        else spread / (len(symbols) - 1)
                    )
                    for symbol in symbols
                },
            ),
        )
