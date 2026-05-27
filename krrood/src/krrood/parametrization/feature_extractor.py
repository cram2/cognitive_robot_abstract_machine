from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from typing import List, Type

import pandas as pd
import sqlalchemy
from sqlalchemy.orm import MANYTOONE, ONETOMANY
from typing_extensions import TYPE_CHECKING, Any

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable
from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.utils import get_python_type_from_sqlalchemy_column, is_data_column
from random_events.variable import compatible_types

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import DataAccessObject


@dataclass
class FeatureExtractor:
    """
    A class to extract features from a given class. Features are all attributes of the class, propagating custom types/objects down. The features are represented as symbolic variables.
    A feature extractor provides additional knowledge about the class.
    """

    features: List[SymbolicExpression]
    """
    The features extracted from the class/instances.
    """

    def __post_init__(self):
        if not self.features:
            raise ValueError(
                "No features provided. If list of instances available, use `FeatureExtractor.from_instances` for instantiation."
            )

    @classmethod
    def from_instances(cls, instances: List[DataAccessObject]) -> FeatureExtractor:
        """
        Create a new feature extractor from the given instances.
        :param instances: The instances to create the feature extractor from.
        :return: A new feature extractor.
        """
        if not instances:
            raise ValueError("No instances provided")

        dao_state = FromDataAccessObjectState()
        root = variable(type(instances[0].from_dao(dao_state)), [])
        features = cls._extract_features(instances[0], root)
        return FeatureExtractor(features)

    @classmethod
    def _extract_features(
        cls, example_instance: DataAccessObject, symbolic_root: Variable
    ) -> List[MappedVariable]:
        result = []
        seen = set()
        queue = deque()
        queue.append((example_instance, symbolic_root))

        while queue:
            current_instance, current_symbolic = queue.popleft()

            if id(current_instance) in seen:
                continue
            seen.add(id(current_instance))

            specification = RelationalSumProductNetworkSpecification(
                type(current_instance)
            )

            for attribute in specification.attributes:
                value = getattr(current_instance, attribute.key)

                if not isinstance(value, compatible_types):
                    continue

                symbolic_attribute = getattr(current_symbolic, attribute.name)
                symbolic_attribute._type_ = get_python_type_from_sqlalchemy_column(
                    attribute
                )
                result.append(symbolic_attribute)

            for part in specification.unique_parts:
                value = getattr(current_instance, part)

                if value is None:
                    continue

                queue.append((value, getattr(current_symbolic, part)))

        return result

    def apply_mapping(self, instance: DataAccessObject) -> List[Any]:
        """
        Extracts the mapped values for each feature from the given instance.
        :param instance: The instance to extract features from.
        :return: A list of mapped values.
        """
        return [
            feature.apply_mapping_on_external_root(instance)
            for feature in self.features
        ]

    def create_dataframe(self, instances: List[DataAccessObject]) -> pd.DataFrame:
        """
        Create a dataframe from the given instances.
        :param instances: The instances to create the dataframe from.
        :return: A dataframe containing the mapped values for each feature.
        """
        result = []
        for instance in instances:
            result.append(self.apply_mapping(instance))
        features_names = [feature._name_ for feature in self.features]
        return pd.DataFrame(columns=features_names, data=result)

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataframe for JointProbabilityTrees by converting boolean columns to integers and enum columns to hashes.
        :param df: The dataframe to preprocess.
        :return: The dataframe in a JPT compatible format.
        """
        feature_map = dict(zip(df.columns, self.features))
        for column in df.columns:
            feature = feature_map[column]
            if feature._type_ is bool:
                df[column] = df[column].astype(int)
            elif isinstance(feature._type_, enum.EnumType):
                df[column] = df[column].apply(lambda x: hash(x))
            elif feature._type_ not in compatible_types and feature._type_ is not None:
                raise TypeError(
                    f"Unsupported type {feature._type_} for column {column}"
                )
        return df


@dataclass
class RelationalSumProductNetworkSpecification:
    """
    Specification used to learn a RelationalSumProductNetwork from a class.
    It contains information about the attributes, unique parts, exchangeable parts, and relations of the class.
    These are determined by the relationships and columns of the DAO class.
    """

    spec: Type[DataAccessObject] = field(init=True)
    """
    The wrapped class that is supposed to be an RSPN.
    """

    def __post_init__(self):
        self.attributes = []
        self.unique_parts = []
        self.exchangeable_parts = []
        self.relations = []

        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(self.spec)

        for relationship in mapper.relationships:
            if relationship.direction == MANYTOONE:
                self.unique_parts.append(relationship.key)
            # not many to many since we have the association table
            elif relationship.direction == ONETOMANY:
                self.exchangeable_parts.append(relationship.key)
        for column in mapper.columns:
            if is_data_column(column) and column not in mapper.relationships:
                self.attributes.append(column)
