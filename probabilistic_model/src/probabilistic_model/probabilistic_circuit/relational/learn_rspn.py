from __future__ import annotations

import enum
from datetime import datetime
from types import NoneType

import pandas as pd
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type

from krrood.ormatic.data_access_objects.helper import get_alternative_mapping
from probabilistic_model.probabilistic_circuit.relational.rspns import (
    RSPNTemplate,
    RSPNSpecification,
)
from random_events.variable import variable_from_name_and_type
from ...learning.jpt.jpt import JointProbabilityTree
from ...learning.jpt.variables import infer_variables_from_dataframe


def get_aggregate_statistics(instance: Any) -> List[Tuple[Any, str]]:
    statistics = []
    for name in dir(instance):
        if name.startswith("__"):
            continue

        attr = getattr(instance, name)

        if not callable(attr):
            continue

        if not hasattr(attr, "_statistic_name"):
            continue

        statistics.append((attr(), attr._statistic_name))

    return statistics


def fill_dataframe_with_parts(
    df_data: Dict[str, List[float]], instances: List[Any], cls: Type, path: str = ""
) -> Dict[str, List[float]]:
    # if cls has an alternative mapping, use that instead
    alternative_mapping = get_alternative_mapping(cls)
    if alternative_mapping:
        cls = alternative_mapping
        new_instances = []
        for instance in instances:
            if instance is None:
                new_instances.append(None)
                continue
            if not isinstance(instance, alternative_mapping):
                instance = alternative_mapping.from_domain_object(instance)
            new_instances.append(instance)
        instances = new_instances

    specification = RSPNSpecification(cls)
    if issubclass(cls, enum.Enum):
        for instance in instances:
            print(instance.value)
            df_data.setdefault(cls.__name__, []).append(instance.value)

    for attribute in specification.attributes:
        column_name = f"{path}.{attribute.name}" if path else attribute.name
        # safe check
        resolved_type = attribute.type_endpoint
        if not issubclass(resolved_type, (float, int, enum.Enum, bool)):
            continue
        for instance in instances:
            value = getattr(instance, attribute.name)
            # if isinstance(value, bool):
            #     value = int(value)
            df_data.setdefault(column_name, []).append(value)

    for part in specification.unique_parts:
        new_instances = []
        for instance in instances:
            if instance is None:
                return df_data
            new_instances.append(getattr(instance, part.public_name))
        new_path = f"{path}.{part.public_name}" if path else part.public_name
        df_data = fill_dataframe_with_parts(
            df_data, new_instances, part.type_endpoint, new_path
        )

    return df_data


def LearnRSPN(cls: Any, instances: List[Any]) -> RSPNTemplate:
    """
    Learn an RSPN for class C.

    - Attributes become univariate leaves (Gaussian for numeric, Bernoulli for boolean)
    - Relation aggregates become Bernoulli leaves over presence (1 if present, else 0)
    - Parts recurse into their class (unique part: map one-to-one; exchangeable part: flatten list)
    - Independent partitions become product nodes; clustering on instances becomes sum nodes with weights

    Returns the root node (ProductUnit or SumUnit) within a ProbabilisticCircuit.
    """
    df_data: Dict[str, List[float]] = {}
    df_data = fill_dataframe_with_parts(df_data, instances, cls)
    # print("----------------------")
    # print("final df_data", df_data)
    copy = df_data.copy()
    for col, val in df_data.items():
        if not isinstance(val[0], (float, int) or enum.Enum):
            del copy[col]

    df = pd.DataFrame(df_data)
    variables = infer_variables_from_dataframe(df)
    # enums = variable_from_name_and_type()
    jpt = JointProbabilityTree(variables, min_samples_per_leaf=15)
    jpt = jpt.fit(df)
    rspn = RSPNTemplate(RSPNSpecification(cls), jpt)
    return rspn
