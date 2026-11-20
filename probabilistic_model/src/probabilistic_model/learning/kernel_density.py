"""
Fitting a circuit that puts a Gaussian kernel on every row, so that samples of it lie
near the rows it was fitted on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
from random_events.variable import Continuous
from typing_extensions import Iterable

from krrood.patterns.called_positionally import called_positionally

from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.learning.exceptions import (
    KernelOfNonContinuousVariableError,
    NoRowsToFitError,
)
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

# %% kernel density estimation


@dataclass(kw_only=True)
class GaussianKernelDensity(LearningMethod):
    """
    Fits an equal mixture of one product of Gaussians per row, centred on the row.

    Along each variable, the standard deviation of a kernel is :attr:`bandwidth` times
    the standard deviation the variable is annotated with.
    """

    bandwidth: float | None = None
    """
    The standard deviation of every kernel as a fraction of its variable's; ``None`` for
    Scott's rule, the number of rows to the power of ``-1 / (d + 4)`` for ``d``
    variables.
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
        :raises KernelOfNonContinuousVariableError: If a variable is not continuous.
        """
        if data.empty:
            raise NoRowsToFitError()
        annotated = (
            list(variables)
            if variables is not None
            else infer_variables_from_dataframe(data)
        )
        for annotated_variable in annotated:
            if not isinstance(annotated_variable.variable, Continuous):
                raise KernelOfNonContinuousVariableError(
                    variable=annotated_variable.variable
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
                        GaussianDistribution(
                            variable=annotated_variable.variable,
                            location=float(row[annotated_variable.variable.name]),
                            scale=bandwidth * annotated_variable.standard_deviation,
                        ),
                        circuit,
                    )
                )
            mixture.add_subcircuit(kernel, -math.log(len(data)))
        return circuit
