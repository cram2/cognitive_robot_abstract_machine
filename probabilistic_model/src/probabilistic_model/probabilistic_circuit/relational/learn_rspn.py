from __future__ import annotations

import enum
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Type,
    Set,
    Union,
)

from krrood.entity_query_language.core.mapped_variable import MappedVariable, Attribute
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable
from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.data_access_objects.helper import (
    get_alternative_mapping,
    get_dao_class,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.relational.rspns import (
    RSPNTemplate,
    RSPNSpecification,
)
from random_events.variable import variable_from_name_and_type, compatible_types


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


def get_features_of_class(
    example_instance: DataAccessObject,
    symbolic_attribute_access: Variable,
    result=[],
    seen: Set = set(),
):
    """
    Get all the class attributes of the given instance that are compatible with the supported types and return them as a list of symbolic attribute accesses.
    :param example_instance: The instance to extract features from.
    :param symbolic_attribute_access: The symbolic variable representing the instance, used to create symbolic accesses to its attributes.
    :param result: The list to append the extracted features to.
    :param seen: A set to keep track of already seen instances to avoid infinite recursion.
    :return: A list of symbolic attribute accesses for the compatible attributes of the instance.
    """
    if id(example_instance) in seen:
        return result
    seen.add(id(example_instance))

    specification = RSPNSpecification(type(example_instance))
    for attribute in specification.attributes:

        value = getattr(example_instance, attribute.key)
        if not isinstance(value, compatible_types):
            continue

        current_symbolic_attribute_access = getattr(
            symbolic_attribute_access, attribute.name
        )
        result.append(current_symbolic_attribute_access)

    for part in specification.unique_parts:
        value = getattr(example_instance, part)
        if value is None:
            continue
        result = get_features_of_class(
            value, getattr(symbolic_attribute_access, part), result, seen
        )
    return result


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
    # if issubclass(cls, enum.Enum):
    #     for instance in instances:
    #         df_data.setdefault(cls.__name__, []).append(instance.key)

    if issubclass(cls, enum.Enum):
        for instance in instances:
            column_name = (
                f"{path}.{specification.spec.clazz.__name__}" if path else instance.name
            )
            # value = create_enum_mapping(cls)[instance.name]
            df_data.setdefault(column_name, []).append(hash(instance.name))

    for attribute in specification.attributes:
        # create Attribute
        column_name = f"{path}.{attribute.name}" if path else attribute.name
        resolved_type = attribute.type_endpoint
        if not issubclass(resolved_type, (float, int, enum.Enum, bool)):
            continue
        for instance in instances:
            value = getattr(instance, attribute.name)

            if value is None:
                value = np.nan

            elif isinstance(value, bool):
                value = int(value)

            elif isinstance(value, enum.Enum):
                value = value.value

            elif isinstance(value, (int, float, np.number)):
                value = float(value)

            else:
                raise TypeError(
                    f"Unsupported value type {type(value)} in column {column_name}: {value}"
                )

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


def LearnRSPN(cls: Any, instances: List[DataAccessObject]) -> RSPNTemplate:
    """
    Learn an RSPN for class C.

    - Attributes become univariate leaves (Gaussian for numeric, Bernoulli for boolean)
    - Relation aggregates become Bernoulli leaves over presence (1 if present, else 0)
    - Parts recurse into their class (unique part: map one-to-one; exchangeable part: flatten list)
    - Independent partitions become product nodes; clustering on instances becomes sum nodes with weights

    Returns the root node (ProductUnit or SumUnit) within a ProbabilisticCircuit.
    """
    features = get_features_of_class(instances[0], variable(cls, []), [], set())
    if not features:
        raise ValueError(f"No features found for class {cls}")

    feature_extractor = FeatureExtractor(features)

    df: pd.DataFrame = feature_extractor.create_dataframe(instances)
    variables = infer_variables_from_dataframe(df)

    jpt = JointProbabilityTree(variables, min_samples_per_leaf=15)
    jpt = jpt.fit(df)
    rspn = RSPNTemplate(RSPNSpecification(get_dao_class(cls)), jpt)
    return rspn


@dataclass
class FeatureExtractor:
    features: List[MappedVariable]

    def apply_mapping(self, instance: Any) -> List[Union[*compatible_types]]:
        return [
            feature.apply_mapping_on_external_root(instance)
            for feature in self.features
        ]

    def create_dataframe(self, instances: List[DataAccessObject]) -> pd.DataFrame:
        result = []
        for instance in instances:
            result.append(self.apply_mapping(instance))
        features_names = [f._name_ for f in self.features]
        return pd.DataFrame(columns=features_names, data=result)


def preprocess_dataframe(
    features: List[MappedVariable], df: pd.DataFrame
) -> pd.DataFrame:
    feature_map = dict(zip(df.columns, features))
    for column in df.columns:
        feature = feature_map[column]
        if feature._type_ is bool:
            df[column] = df[column].astype(int)
        elif isinstance(feature._type_, enum.EnumType):
            df[column] = df[column].apply(
                lambda x: hash(x.value) if isinstance(x, enum.Enum) else x
            )
        elif feature._type_ not in compatible_types and feature._type_ is not None:
            raise TypeError(f"Unsupported type {feature._type_} for column {column}")
    return df
