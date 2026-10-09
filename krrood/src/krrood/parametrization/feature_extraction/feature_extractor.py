from __future__ import annotations

import enum
import itertools
from collections import defaultdict, deque
from dataclasses import dataclass

import pandas as pd
import sqlalchemy
from typing_extensions import Any, Type
from krrood.entity_query_language.core.mapped_variable import MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import variable
from krrood.ormatic.data_access_objects.dao import (
    CollectionRelationship,
    SingleRelationship,
    DataAccessObject,
    get_dao_schema,
)
from krrood.ormatic.data_access_objects.helper import get_data_access_object_class
from krrood.ormatic.exceptions import NoDAOFoundForTypeError
from krrood.ormatic.utils import get_python_type_from_sqlalchemy_column
from krrood.parametrization.feature_extraction.aggregations import get_aggregation_class
from krrood.parametrization.feature_extraction.exceptions import (
    NoInstancesProvidedError,
    UnsupportedFeatureTypeError,
)
from krrood.symbolic_math.symbolic_math import SymbolicMathType
from random_events.variable import compatible_types


@dataclass
class ExtractedFeatures:
    """
    The result of traversing an object graph for features.
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
    Extracts symbolic features from domain objects, including scalar attributes, unique-
    part sub-trees, and aggregation statistics over exchangeable parts.

    The objects are read as they are. Which of their attributes are scalar features,
    unique parts or exchangeable parts is taken from the data access object class of
    their type, so no object has to be converted.

    Prefer ``FeatureExtractor.from_instances`` for construction; the direct constructor
    receives an already-built :class:`ExtractedFeatures`.
    """

    extracted_features: ExtractedFeatures
    """
    The discovered features produced by traversing the object graph.
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
    def from_instances(cls, instances: list[Any]) -> FeatureExtractor:
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

        first_instance = instances[0]
        root = variable(type(first_instance), [])
        extracted = cls._extract_features(first_instance, root)
        return cls(extracted_features=extracted)

    @staticmethod
    def _data_access_object_class_of(instance: Any) -> Type[DataAccessObject]:
        """
        :param instance: A domain object.
        :return: The data access object class that describes the attributes of its type.
        :raises NoDAOFoundForTypeError: If the type of the instance is not mapped.
        """
        data_access_object_class = get_data_access_object_class(type(instance))
        if data_access_object_class is None:
            raise NoDAOFoundForTypeError(type(instance))
        return data_access_object_class

    @staticmethod
    def _extract_features(
        example_instance: Any, symbolic_root: Variable
    ) -> ExtractedFeatures:
        """
        Traverses the object graph breadth-first and collects all features.

        :param example_instance: A representative instance that defines the schema.
        :param symbolic_root: The root symbolic variable for the traversal.
        :return: The discovered scalar features and per-relation aggregation features.
        """
        result = []
        # A property may build the object it returns, which would be freed after its
        # visit and hand its id to a later object. Holding the visited objects keeps
        # their ids theirs.
        seen = {}
        exchangeable_features = defaultdict(list)
        queue = deque()
        queue.append((example_instance, symbolic_root))

        while queue:
            current_instance, current_symbolic = queue.popleft()

            if id(current_instance) in seen:
                continue
            seen[id(current_instance)] = current_instance

            data_access_object_class = FeatureExtractor._data_access_object_class_of(
                current_instance
            )
            schema = get_dao_schema(data_access_object_class)

            result.extend(
                FeatureExtractor._process_attributes(
                    current_instance,
                    current_symbolic,
                    data_access_object_class,
                    schema.data_column_names,
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
        instance: Any,
        symbolic_root: Variable,
        data_access_object_class: Type[DataAccessObject],
        column_names: tuple[str, ...],
    ) -> list[MappedVariable]:
        """
        Collects symbolic variables for all scalar data columns of ``instance``.

        Columns whose value is not a compatible primitive type are skipped.
        :param instance: The domain object to inspect.
        :param symbolic_root: The symbolic variable rooted at ``instance``.
        :param data_access_object_class: The data access object class of the instance's
            type.
        :param column_names: Names of the scalar data columns of the instance's schema.
        :return: One typed ``MappedVariable`` per compatible scalar attribute.
        """
        mapper = sqlalchemy.inspection.inspect(data_access_object_class)
        column_by_name = {column.name: column for column in mapper.columns}
        result = []
        for name in column_names:
            column = column_by_name[name]
            value = FeatureExtractor._as_feature_value(
                getattr(instance, column.key),
                get_python_type_from_sqlalchemy_column(column),
            )

            if not isinstance(value, compatible_types):
                continue

            symbolic_attribute = getattr(symbolic_root, column.name)
            symbolic_attribute._type_ = FeatureExtractor._type_of_column_value(
                column, value
            )
            result.append(symbolic_attribute)
        return result

    @staticmethod
    def _as_feature_value(value: Any, type_: type) -> Any:
        """
        The value as a feature holds it.

        A domain object may hold a number as a constant symbolic expression where its
        column holds the number itself, so such an expression is evaluated.

        :param value: The value read from a domain object.
        :param type_: The type the feature of that value ranges over.
        :return: The value as ``type_`` if it is a constant symbolic expression of a
            number, otherwise the value unchanged.
        """
        if (
            isinstance(value, SymbolicMathType)
            and type_ in (int, float, bool)
            and value.is_constant()
        ):
            return type_(value)
        return value

    @staticmethod
    def _type_of_column_value(column: sqlalchemy.Column, value: Any) -> type:
        """
        The python type of what a column holds.

        A column storing enum members of any enum says only :class:`enum.Enum`, which
        has no members of its own to build a domain from, so the value standing in it
        says which enum it is.

        :param column: The column the value was read from.
        :param value: The value read from it.
        :return: The type a variable over this column ranges over.
        """
        column_type = get_python_type_from_sqlalchemy_column(column)
        if column_type is enum.Enum:
            return type(value)
        return column_type

    @staticmethod
    def _process_many_to_one(
        instance: Any,
        symbolic_root: Variable,
        relationships: tuple[SingleRelationship, ...],
    ) -> deque[Any]:
        """
        Enqueues non-null single-valued relations for further BFS traversal.

        :param instance: The domain object to inspect.
        :param symbolic_root: The symbolic variable rooted at ``instance``.
        :param relationships: Single-valued relationships of the instance's schema.
        :return:``(child_instance, child_symbolic)`` pairs ready for BFS expansion.
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
        current_instance: Any,
        relationships: tuple[CollectionRelationship, ...],
    ) -> dict[str, list[MappedVariable]]:
        """
        Collects aggregation statistic variables for all collection-valued relations of
        ``current_instance``.

        :param current_instance: The domain object to inspect.
        :param relationships: Collection-valued relationships of the instance's schema.
        :return: A mapping from each collection field name to its aggregation variables.
        """
        result = defaultdict(list)
        aggregation_cls = get_aggregation_class(type(current_instance))
        if aggregation_cls is None:
            return result

        for relationship in relationships:
            if not getattr(current_instance, relationship.key):
                continue
            aggregation_instance = aggregation_cls(
                instance=current_instance, field_name=relationship.key
            )
            for feature in aggregation_instance.symbolic_aggregation_features():
                result[relationship.key].append(feature)

        return result

    def apply_mapping(self, instance: Any) -> list[Any]:
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
        aggregation_cls = get_aggregation_class(type(instance))
        aggregation_instance = (
            aggregation_cls(instance=instance) if aggregation_cls is not None else None
        )
        for feature in self.features:
            if feature in aggregation_features:
                result.append(
                    feature.apply_mapping_on_external_root(aggregation_instance)
                )
            else:
                result.append(
                    self._as_feature_value(
                        feature.apply_mapping_on_external_root(instance),
                        feature._type_,
                    )
                )
        return result

    def create_dataframe(self, instances: list[Any]) -> pd.DataFrame:
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
        Check the dataframe's columns are of types a JointProbabilityTree can be fitted
        on.

        Enum and boolean columns are left as they are: ``infer_variables_from_dataframe``
        types either as a ``Symbolic`` variable over the values present, which is what
        JPT needs to split on it by category rather than by a numeric threshold. An enum
        member itself is also what a query later conditions on, so its leaf keeps the
        member's own hash; storing the hash as a number instead would round it through
        a float and never match the member again.

        :param df: The dataframe to preprocess.
        :return: The dataframe in a JPT compatible format.
        :raises UnsupportedFeatureTypeError: If a column's type cannot be fitted on.
        """
        feature_map = dict(zip(df.columns, self.features))
        for column in df.columns:
            feature = feature_map[column]
            if isinstance(feature._type_, enum.EnumType):
                continue
            if feature._type_ not in compatible_types and feature._type_ is not None:
                raise UnsupportedFeatureTypeError(
                    feature_type=feature._type_, column_name=column
                )
        return df
