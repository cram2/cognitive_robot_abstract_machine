"""
The ways a probabilistic circuit is fitted on a dataframe, so that a model owning a
circuit can be told how to fit it instead of fitting it itself.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from typing_extensions import Iterable, Optional, Sequence

from probabilistic_model.learning.jpt.variables import (
    AnnotatedVariable,
    infer_variables_from_dataframe,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
)


@dataclass
class LearningMethod(ABC):
    """
    A way of fitting a probabilistic circuit on a dataframe.
    """

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        variables: Optional[Iterable[AnnotatedVariable]] = None,
    ) -> ProbabilisticCircuit:
        """
        Fit a circuit on the rows of a dataframe.

        :param data: The training rows, one column per variable.
        :param variables: The variables inferred over the rows, one per column,
            carrying the annotation (mean, standard deviation, split thresholds) the
            fit is guided by. ``None`` infers them from the data.
        :return: The fitted circuit.
        """


@dataclass
class StratifiedLearning(LearningMethod):
    """
    Fits one circuit per distinct joint value of some variables with another learning
    method, and combines them under a sum weighted by each value's relative frequency.

    Every row of one partition shares the same value of those variables, so within its
    circuit their distribution is a single point by construction, however the wrapped
    method splits on the remaining variables: the fitted circuit is
    support-deterministic over them, the precondition a causal query on them needs. A
    single unconstrained fit could instead spread rows sharing a value across sibling
    leaves.
    """

    variables: Sequence[str]
    """
    The names of the variables whose joint value the rows are partitioned by; a single
    variable is a one-element sequence.
    """

    method: LearningMethod
    """
    What every partition is fitted with.
    """

    def fit(
        self,
        data: pd.DataFrame,
        variables: Optional[Iterable[AnnotatedVariable]] = None,
    ) -> ProbabilisticCircuit:
        if variables is None:
            variables = infer_variables_from_dataframe(data)
        result = ProbabilisticCircuit()
        root = SumUnit(probabilistic_circuit=result)
        total_row_count = len(data)
        for _, partition in data.groupby(list(self.variables), sort=False):
            partition_circuit = self.method.fit(
                partition.reset_index(drop=True), variables
            )
            node_index_map = result.mount(partition_circuit.root)
            root.add_subcircuit(
                node_index_map[partition_circuit.root.index],
                math.log(len(partition) / total_row_count),
            )
        return result
