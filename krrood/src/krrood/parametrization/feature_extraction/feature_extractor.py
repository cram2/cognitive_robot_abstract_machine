from __future__ import annotations

import enum
import itertools
from collections import defaultdict, deque
from dataclasses import dataclass

import pandas as pd
import sqlalchemy
from typing_extensions import TYPE_CHECKING, Any
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable
from krrood.ormatic.data_access_objects.dao import (
    CollectionRelationship,
    SingleRelationship,
    get_dao_schema,
)
from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.utils import get_python_type_from_sqlalchemy_column
from krrood.parametrization.feature_extraction.aggregations import get_aggregation_class
from krrood.parametrization.feature_extraction.exceptions import (
    NoInstancesProvidedError,
    UnsupportedFeatureTypeError,
)
from random_events.variable import compatible_types

if TYPE_CHECKING:
    from krrood.ormatic.data_access_objects.dao import DataAccessObject


@dataclass
class ExtractedFeatures:
    """
    The result of traversing a DAO object graph for features.
    """

    features: list[MappedVariable]
    """
    Symbolic variables for every extractable feature, in traversal order.
    """

    exchangeable_features: dict[str, list[MappedVariable]]
    """
    Mapping from each exchangeable-part field name to its aggregation variables.
    """


@dataclass
class FeatureExtractor:
    """
    Extracts symbolic features from DAO instances, including scalar attributes,
    unique-part sub-trees, and aggregation statistics over exchangeable parts.

    Prefer ``FeatureExtractor.from_instances`` for construction; the direct
    constructor receives an already-built :class:`ExtractedFeatures`.
    """

    extracted_features: ExtractedFeatures
    """
    The discovered features produced by traversing the DAO object graph.
    """

    @property
    def features(self) -> list[MappedVariable]:
        """
        Symbolic variables representing every extractable feature, in traversal order.
        """
        return self.extracted_features.features

    @property
    def exchangeable_features(self) -> dict[str, list[MappedVariable]]:
        """
        Mapping from each exchangeable-part field name to its aggregation variables.
        """
        return self.extracted_features.exchangeable_features

    @classmethod
    def from_instances(cls, instances: list[DataAccessObject]) -> FeatureExtractor:
        """
        Create a new feature extractor from the given instances.

        Exchangeable parts whose domain class has no :class:`~krrood.parametrization.feature_extraction.aggregations.AggregationStatistic`
        are silently skipped; the remaining scalar and unique-part features are still extracted.

        :param instances: The instances to create the feature extractor from.
        :return: A new feature extractor.
        :raises NoInstancesProvidedError: If ``instances`` is empty.
        """
        if not instances:
            raise NoInstancesProvidedError()

        dao_state = FromDataAccessObjectState()
        first_instance = instances[0]
        domain_object = first_instance.from_dao(dao_state)

        root = variable(type(domain_object), [])
        extracted = cls._extract_features(first_instance, root)
        return cls(extracted_features=extracted)

    @staticmethod
    def _extract_features(
        example_instance: DataAccessObject, symbolic_root: Variable
    ) -> ExtractedFeatures:
        """
        Traverses the DAO object graph breadth-first and collects all features.

        :param example_instance: A representative DAO instance that defines the schema.
        :param symbolic_root: The root symbolic variable for the traversal.
        :return: The discovered scalar features and per-relation aggregation features.
        """
        result = []
        seen = set()
        exchangeable_features = defaultdict(list)
        queue = deque()
        queue.append((example_instance, symbolic_root))

        while queue:
            current_instance, current_symbolic = queue.popleft()

            if id(current_instance) in seen:
                continue
            seen.add(id(current_instance))

            schema = get_dao_schema(type(current_instance))

            result.extend(
                FeatureExtractor._process_attributes(
                    current_instance, current_symbolic, schema.data_column_names
                )
            )

            exchangeable_features.update(
                FeatureExtractor._process_many_to_many(
                    current_instance, schema.collection_relationships
                )
            )
            queue.extend(
                FeatureExtractor._process_many_to_one(
                    current_instance,
                    current_symbolic,
                    schema.single_relationships,
                )
            )

        result.extend(itertools.chain.from_iterable(exchangeable_features.values()))
        return ExtractedFeatures(result, exchangeable_features)

    @staticmethod
    def _process_attributes(
        instance: DataAccessObject,
        symbolic_root: Variable,
        column_names: tuple[str, ...],
    ) -> list[MappedVariable]:
        """
        Collects symbolic variables for all scalar data columns of ``instance``.

        Columns whose value is not a compatible primitive type are skipped.
        :param instance: The DAO instance to inspect.
        :param symbolic_root: The symbolic variable rooted at ``instance``.
        :param column_names: Names of the scalar data columns of the instance's schema.
        :return: One typed ``MappedVariable`` per compatible scalar attribute.
        """
        mapper = sqlalchemy.inspection.inspect(type(instance))
        column_by_name = {column.name: column for column in mapper.columns}
        result = []
        for name in column_names:
            column = column_by_name[name]
            value = getattr(instance, column.key)

            if not isinstance(value, compatible_types):
                continue

            symbolic_attribute = getattr(symbolic_root, column.name)
            symbolic_attribute._type_ = get_python_type_from_sqlalchemy_column(column)
            result.append(symbolic_attribute)
        return result

    @staticmethod
    def _process_many_to_one(
        instance: DataAccessObject,
        symbolic_root: Variable,
        relationships: tuple[SingleRelationship, ...],
    ) -> deque[Any]:
        """
        Enqueues non-null single-valued relations for further BFS traversal.

        :param instance: The DAO instance to inspect.
        :param symbolic_root: The symbolic variable rooted at ``instance``.
        :param relationships: Single-valued relationships of the instance's schema.
        :return: ``(child_instance, child_symbolic)`` pairs ready for BFS expansion.
        """
        queue = deque()
        for relationship in relationships:
            value = getattr(instance, relationship.key)

            if value is None:
                continue

            queue.append((value, getattr(symbolic_root, relationship.key)))
        return queue

    @staticmethod
    def _process_many_to_many(
        current_instance: DataAccessObject,
        relationships: tuple[CollectionRelationship, ...],
    ) -> dict[str, list[MappedVariable]]:
        """
        Collects aggregation statistic variables for all collection-valued relations of ``current_instance``.

        :param current_instance: The DAO instance to inspect.
        :param relationships: Collection-valued relationships of the instance's schema.
        :return: A mapping from each collection field name to its aggregation variables.
        """
        result = defaultdict(list)
        dao_state = FromDataAccessObjectState()
        domain_object = current_instance.from_dao(dao_state)

        aggregation_cls = get_aggregation_class(type(domain_object))
        if aggregation_cls is None:
            return result

        for relationship in relationships:
            if not getattr(domain_object, relationship.key):
                continue
            aggregation_instance = aggregation_cls(
                instance=domain_object, field_name=relationship.key
            )
            for feature in aggregation_instance.symbolic_aggregation_features():
                result[relationship.key].append(feature)

        return result

    def apply_mapping(self, instance: DataAccessObject) -> list[Any]:
        """
        Extracts the mapped values for each feature from the given instance.
        :param instance: The instance to extract features from.
        :return: A list of mapped values.
        """
        aggregation_features = {
            aggregation
            for aggregations in self.exchangeable_features.values()
            for aggregation in aggregations
        }
        result = []
        dao_state = FromDataAccessObjectState()
        domain_object = instance.from_dao(dao_state)
        aggregation_cls = get_aggregation_class(type(domain_object))
        aggregation_instance = (
            aggregation_cls(instance=domain_object)
            if aggregation_cls is not None
            else None
        )
        for feature in self.features:
            if feature in aggregation_features:
                result.append(
                    feature.apply_mapping_on_external_root(aggregation_instance)
                )
            else:
                result.append(feature.apply_mapping_on_external_root(instance))
        return result

    def create_dataframe(self, instances: list[DataAccessObject]) -> pd.DataFrame:
        """
        Create a dataframe from the given instances.
        :param instances: The instances to create the dataframe from.
        :return: A dataframe containing the mapped values for each feature.
        """
        result = [self.apply_mapping(instance) for instance in instances]
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
                raise UnsupportedFeatureTypeError(
                    feature_type=feature._type_, column_name=column
                )
        return df
