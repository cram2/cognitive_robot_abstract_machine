from __future__ import annotations

from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.parametrization.feature_extractor import FeatureExtractor
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from collections import deque
from typing import Union, Optional, List, Iterable
import numpy as np
import pandas as pd
from jpt.learning.impurity import Impurity
from random_events.product_algebra import VariableMap
from random_events.variable import Variable
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    SumUnit,
    ProbabilisticCircuit,
)


def learn_probabilistic_circuit(
    instances: List[DataAccessObject],
    targets: Optional[Iterable[Variable]] = None,
    features: Optional[Iterable[Variable]] = None,
    min_samples_per_leaf: Union[int, float] = 1,
    min_impurity_improvement: float = 0.0,
    max_leaves: Union[int, float] = float("inf"),
    max_depth: Union[int, float] = float("inf"),
    dependencies: Optional[VariableMap] = None,
    total_samples: int = 1,
    indices: Optional[np.ndarray] = None,
    impurity: Optional[Impurity] = None,
    c45queue: Optional[deque] = None,
    keep_sample_indices: bool = False,
    root: Optional[SumUnit] = None,
) -> ProbabilisticCircuit:
    """
    Learn a ProbabilisticCircuit from a class and a list of instances.
    :param instances: The instances to learn from.
    :param targets: The variables to optimize for.
    :param features: The variables that are used to craft criteria.
    :param min_samples_per_leaf: The minimum number of samples to create another sum node. If this is smaller than one, it will be reinterpreted as fraction w. r. t. the number of samples total.
    :param min_impurity_improvement: The minimum impurity improvement to create another sum node.
    :param max_leaves: The maximum number of leaves in the tree.
    :param max_depth: The maximum depth of the tree.
    :param dependencies: The dependencies between variables.
    :param total_samples: The total number of samples.
    :param indices: The indices of the samples.
    :param impurity: The impurity object to use.
    :param c45queue: The queue to use for C4.5.
    :param keep_sample_indices: Whether to keep the sample indices.
    :param root: The root of the tree.
    :return: The learned ProbabilisticCircuit.
    """

    if c45queue is None:
        c45queue = deque()

    extractor = FeatureExtractor.from_instances(instances)

    if not instances:
        raise ValueError("No instances provided")

    df: pd.DataFrame = extractor.create_dataframe(instances)
    df = extractor.preprocess_dataframe(df)
    df = df.sort_index(axis=1)
    variables = infer_variables_from_dataframe(df)

    jpt = JointProbabilityTree(
        annotated_variables=variables,
        targets=targets,
        features=features,
        min_samples_per_leaf=min_samples_per_leaf,
        min_impurity_improvement=min_impurity_improvement,
        max_leaves=max_leaves,
        max_depth=max_depth,
        dependencies=dependencies,
        total_samples=total_samples,
        indices=indices,
        impurity=impurity,
        c45queue=c45queue,
        keep_sample_indices=keep_sample_indices,
        root=root,
    )
    jpt = jpt.fit(df)
    return jpt
