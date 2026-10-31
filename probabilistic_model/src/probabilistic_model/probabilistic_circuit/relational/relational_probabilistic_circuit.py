"""
Relational probabilistic circuits.

.. note::
    This module deliberately bridges ``probabilistic_model`` and ``krrood``: it
    imports krrood feature extraction here, while ``krrood.parametrization.model_registries``
    imports :class:`RelationalProbabilisticCircuit` back. This bidirectional coupling
    predates the relational refactor and is kept intentionally; it is the seam where
    krrood's symbolic feature extraction meets probabilistic_model's circuits.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Optional,
    Type,
)

from krrood.class_diagrams.utils import common_base_class
from krrood.ormatic.data_access_objects.dao import (
    DataAccessObjectSchema,
    get_dao_schema,
)
from krrood.ormatic.data_access_objects.helper import get_data_access_object_class
from krrood.parametrization.feature_extraction.aggregations import (
    compute_aggregation_statistics,
)
from krrood.parametrization.feature_extraction.feature_extractor import FeatureExtractor
from krrood.patterns.field_metadata import JSONMetadata

if TYPE_CHECKING:
    from krrood.entity_query_language.query.match import Match
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import LearningMethod
from probabilistic_model.learning.jpt.variables import (
    infer_variables_from_dataframe,
)
from probabilistic_model.probabilistic_model import PartialPointType
from probabilistic_model.probabilistic_circuit.groundable_circuit import (
    GroundableCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    InstanceNotOfClassError,
    MixedCircuitTypesError,
    NoCommonSuperclassError,
    UnknownExchangeablePartsError,
)
from probabilistic_model.probabilistic_circuit.relational.exchangeable_grounding import (
    CircuitType,
    GroundedPartTemplate,
    GroundingMode,
    RelationalGrounding,
)
from probabilistic_model.probabilistic_circuit.relational.part_learning_methods import (
    PartLearningMethods,
)
from probabilistic_model.probabilistic_circuit.relational.template import (
    RelationalDistributionTemplate,
)
from random_events.variable import Variable


@dataclass
class ExchangeableDistributionTemplate(RelationalDistributionTemplate):
    """
    A fitted distribution template for one exchangeable (many-to-many) relation.

    Wraps a ``RelationalProbabilisticCircuit`` that was trained on the child objects of
    the relation together with the parent's aggregation statistics as latent context
    variables.
    """

    latent_variables: list[Variable] = field(default_factory=list)
    """
    Variables shared between the parent and child circuits that are used for
    conditioning but are not part of the final grounded distribution.
    """

    def grounded_templates_of_parts(
        self, parts: list[Match], ground: Callable[[Match], CircuitType]
    ) -> list[GroundedPartTemplate[CircuitType]]:
        """
        :param parts: The query parts, one per child object of the relation.
        :param ground: Grounds the template for one query part.
        :return: The grounded template of every part. Parts whose queries have the same
            shape share one grounding.
        """
        groundings: dict[Hashable, tuple[CircuitType, str]] = {}
        result = []
        for index, part in enumerate(parts):
            prefix = self._prefix_for_part(part, index)
            shape = self.template_distribution.query_shape(part)
            if shape not in groundings:
                groundings[shape] = (ground(part), prefix)
            circuit, grounded_prefix = groundings[shape]
            result.append(GroundedPartTemplate(circuit, grounded_prefix, prefix))
        return result

    def variable_of_part(
        self, variable: Variable, part: GroundedPartTemplate[Any]
    ) -> Variable:
        """
        :param variable: A variable of the grounded template of a part.
        :param part: The grounded template of the part.
        :return: The variable under the namespace of the part.
        """
        if variable in self.template_distribution.class_probabilistic_circuit.variables:
            return type(variable)(
                f"{part.prefix}.{variable.name}", domain=variable.domain
            )
        if part.prefix == part.grounded_prefix:
            return variable
        return type(variable)(
            part.prefix + variable.name[len(part.grounded_prefix) :],
            domain=variable.domain,
        )


@dataclass
class RelationalProbabilisticCircuit:
    """
    A probabilistic circuit that jointly models a class and its relational structure.
    """

    class_: Optional[Type] = field(
        init=False, default=None, metadata=JSONMetadata(serialize=True).as_dict()
    )
    """
    The domain class whose instances this distribution models, set by ``fit``.
    """

    class_probabilistic_circuit: Optional[GroundableCircuit] = None
    """
    The fitted joint distribution over the class's scalar attributes and aggregation
    statistics, populated by ``fit``.

    Its type is the one :attr:`learning_method` fits, and the class circuits of all
    exchangeable templates have the same type.
    """

    exchangeable_distribution_templates: dict[str, ExchangeableDistributionTemplate] = (
        field(default_factory=dict)
    )
    """
    Mapping from each exchangeable-part field name to its fitted
    ``ExchangeableDistributionTemplate``.
    """

    monte_carlo_sample_count: int = 10
    """
    Number of Monte-Carlo samples drawn per exchangeable part to integrate out
    aggregation statistics that cannot be determined from the grounding query.

    Must be a positive integer.
    """

    learning_method: LearningMethod = field(default_factory=JointProbabilityTree)
    """
    What the class-level circuit is fitted with.
    """

    part_learning_methods: PartLearningMethods = field(
        default_factory=PartLearningMethods
    )
    """
    What the template distribution of every exchangeable part, nested ones included, is
    fitted with, keyed by attribute accesses on a variable of :attr:`class_`.

    Its default must fit the same circuit type as :attr:`learning_method`.
    """

    schema_information: Optional[DataAccessObjectSchema] = field(
        init=False, default=None
    )
    """
    The :class:`~krrood.ormatic.data_access_objects.dao.DataAccessObjectSchema`
    describing the DAO class's columns and relationships.
    """

    feature_extractor: Optional[FeatureExtractor] = field(init=False, default=None)
    """
    Feature extractor built from the training instances.

    ..note::
        Only needed while fitting; grounding derives its aggregation statistics from the
        queried domain object instead, so this stays ``None`` on a deserialized circuit.
    """

    @staticmethod
    def _build_class_dataframe(
        feature_extractor: FeatureExtractor,
        instances: list[Any],
        dataframe_from_parent: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Build the preprocessed dataframe used to fit the class-level JPT.

        :param feature_extractor: The extractor used to create and preprocess the
            dataframe.
        :param instances: Training instances to extract features from.
        :param dataframe_from_parent: Pre-built dataframe from a parent fit call, or
            ``None``.
        :return: A preprocessed, column-sorted dataframe ready for JPT training.
        """
        if dataframe_from_parent is not None:
            return dataframe_from_parent
        dataframe = feature_extractor.create_dataframe(instances)
        dataframe = feature_extractor.preprocess_dataframe(dataframe)
        return dataframe.sort_index(axis=1)

    def _build_child_joint_dataframe(
        self,
        exchangeable_part: str,
        instances: list[Any],
        aggregation_indices: list[int],
        aggregation_names: list[str],
        child_feature_extractor: FeatureExtractor,
    ) -> pd.DataFrame:
        """
        Build a dataframe combining aggregation statistics with per-child-object
        attributes.

        Each row corresponds to one child object and contains the parent instance's
        aggregation values followed by all child features (including nested unique-part
        attributes). Attribute columns are named by the access-path names produced by
        :meth:`~krrood.entity_query_language.core.mapped_variable.MappedVariable.get_clean_name_from_mapped_variable`
        so that, after part-prefix renaming, they align with the krrood access-path convention.
        The child's own aggregation statistics are named by their ``_name_`` instead, the
        name the template of the part nested below the child uses for them as latent
        variables.

        :param exchangeable_part: Field name of the one-to-many relation on each instance.
        :param instances: Training instances from which rows are generated.
        :param aggregation_indices: Positions of aggregation features in the feature vector.
        :param aggregation_names: Column names for the aggregation portion of each row.
        :param child_feature_extractor: Feature extractor built from the child instances.
        :return: A dataframe with one row per child object across all instances.
        """
        rows = []
        for instance in instances:
            feature_vector = self.feature_extractor.apply_mapping(instance)
            aggregation_row = [feature_vector[index] for index in aggregation_indices]
            for child in getattr(instance, exchangeable_part):
                child_features = child_feature_extractor.apply_mapping(child)
                rows.append(aggregation_row + child_features)
        child_aggregation_features = child_feature_extractor.aggregation_features
        child_column_names = [
            (
                feature._name_
                if feature in child_aggregation_features
                else feature.get_clean_name_from_mapped_variable()
            )
            for feature in child_feature_extractor.features
        ]
        return pd.DataFrame(columns=aggregation_names + child_column_names, data=rows)

    def _fit_exchangeable_part(
        self,
        exchangeable_part: str,
        instances: list[Any],
    ) -> ExchangeableDistributionTemplate:
        """
        Fit an ``ExchangeableDistributionTemplate`` for one exchangeable part.

        Builds a joint dataframe that pairs each child object's attributes with the
        parent's aggregation statistics, infers which variables are latent (the
        aggregation columns), and recursively fits a ``RelationalProbabilisticCircuit``
        on the child instances using that dataframe.

        :param exchangeable_part: Field name of the one-to-many relation on each
            instance.
        :param instances: Training instances whose children are used to fit the
            template.
        :return: A fitted ``ExchangeableDistributionTemplate`` for the given part.
        """
        aggregation_functions = self.feature_extractor.exchangeable_features[
            exchangeable_part
        ]
        aggregation_indices = [
            next(
                index
                for index, feature in enumerate(self.feature_extractor.features)
                if feature is aggregation_function
            )
            for aggregation_function in aggregation_functions
        ]
        aggregation_names = [function._name_ for function in aggregation_functions]

        child_instances = list(
            itertools.chain.from_iterable(
                getattr(instance, exchangeable_part) for instance in instances
            )
        )
        child_feature_extractor = FeatureExtractor.from_instances(child_instances)
        child_dataframe = self._build_child_joint_dataframe(
            exchangeable_part,
            instances,
            aggregation_indices,
            aggregation_names,
            child_feature_extractor,
        )
        latent_variables = [
            inferred.variable
            for inferred in infer_variables_from_dataframe(child_dataframe)
            if inferred.variable.name in aggregation_names
        ]
        template = ExchangeableDistributionTemplate(
            RelationalProbabilisticCircuit(
                learning_method=self.part_learning_methods.method_of_path(
                    exchangeable_part
                ),
                part_learning_methods=self.part_learning_methods.of_part(
                    exchangeable_part
                ),
            ),
            latent_variables,
        )
        template.template_distribution.fit(
            child_instances, dataframe_from_parent=child_dataframe
        )
        part_circuit_type = type(
            template.template_distribution.class_probabilistic_circuit
        )
        if part_circuit_type is not type(self.class_probabilistic_circuit):
            raise MixedCircuitTypesError(
                self.class_,
                exchangeable_part,
                type(self.class_probabilistic_circuit),
                part_circuit_type,
            )
        return template

    @staticmethod
    def _class_of(instances: list[Any], class_: Optional[Type]) -> Type:
        """
        :param instances: Training instances.
        :param class_: The class the circuit models, or ``None`` for the lowest common
            superclass of the instances in their method resolution orders.
        :return: The class the circuit models.
        """
        if class_ is None:
            class_ = common_base_class([type(instance) for instance in instances])
            if class_ is None:
                raise NoCommonSuperclassError(
                    sorted({type(instance) for instance in instances}, key=str)
                )
        for instance in instances:
            if not isinstance(instance, class_):
                raise InstanceNotOfClassError(class_, type(instance))
        return class_

    def fit(
        self,
        instances: list[Any],
        class_: Optional[Type] = None,
        dataframe_from_parent: Optional[pd.DataFrame] = None,
    ):
        """
        Fit the relational probabilistic circuit from a list of domain objects.

        Builds a ``FeatureExtractor``, fits the class-level circuit on the class-level
        features with :attr:`learning_method`, and then recursively fits one
        ``ExchangeableDistributionTemplate`` per exchangeable part discovered in the
        schema.

        :param instances: Training instances.
        :param class_: The class the circuit models, or ``None`` for the lowest common
            superclass of the instances.
        :param dataframe_from_parent: Pre-built dataframe supplied by a parent
            ``_fit_exchangeable_part`` call. When provided, feature extraction and
            preprocessing are skipped.
        :return:``self``, to allow chaining.
        :raises NoCommonSuperclassError: If ``class_`` is not given and the instances
            share no superclass but ``object``.
        :raises InstanceNotOfClassError: If an instance is not of ``class_``.
        """
        self.class_ = self._class_of(instances, class_)
        self.feature_extractor = FeatureExtractor.from_instances(instances)
        class_dataframe = self._build_class_dataframe(
            self.feature_extractor, instances, dataframe_from_parent
        )
        variables = infer_variables_from_dataframe(class_dataframe)
        self.class_probabilistic_circuit = self.learning_method.fit(
            class_dataframe, variables
        )
        self.schema_information = get_dao_schema(
            get_data_access_object_class(type(instances[0]))
        )
        exchangeable_parts = [
            collection_relationship.key
            for collection_relationship in self.schema_information.collection_relationships
            if collection_relationship.key
            in self.feature_extractor.exchangeable_features
        ]
        unknown_parts = {
            path.split(".")[0] for path in self.part_learning_methods.methods_by_path
        } - set(exchangeable_parts)
        if unknown_parts:
            raise UnknownExchangeablePartsError(self.class_, sorted(unknown_parts))
        for exchangeable_part in exchangeable_parts:
            self.exchangeable_distribution_templates[exchangeable_part] = (
                self._fit_exchangeable_part(exchangeable_part, instances)
            )
        return self

    def query_shape(self, query: Match) -> Hashable:
        """
        :param query: A query this circuit is grounded for.
        :return: What the structure of the grounding depends on: the aggregation
            statistics and query parts of every exchangeable relation. Queries of the
            same shape are grounded into circuits that differ only in the names of their
            variables and, for sampled statistics, in the samples.
        """
        instance = query.construct_instance()
        return tuple(
            (
                exchangeable_part_name,
                frozenset(
                    compute_aggregation_statistics(
                        instance, exchangeable_part_name, template.latent_variables
                    ).items()
                ),
                tuple(
                    template.template_distribution.query_shape(part)
                    for part in query._kwargs_[exchangeable_part_name]
                ),
            )
            for exchangeable_part_name, template in self.exchangeable_distribution_templates.items()
        )

    def ground(
        self,
        query: Match,
        grounding_mode: GroundingMode = GroundingMode.SAMPLED,
    ) -> GroundableCircuit:
        """
        Ground the relational circuit for a specific query, in the circuit type it was
        fitted in.

        :param query: An underspecified, resolved query instance whose structure
            determines which parts are grounded and how many child objects each
            exchangeable relation contains.
        :param grounding_mode: How to treat aggregation latents the query leaves
            undetermined. See :class:`GroundingMode`.
        :return: A circuit over all variables implied by the query.
        :raises CircuitNotFittedError: If ``ground`` is called before ``fit``.
        """
        return self.grounding(grounding_mode).ground(query)

    def grounding(
        self, grounding_mode: GroundingMode = GroundingMode.SAMPLED
    ) -> RelationalGrounding:
        """
        :param grounding_mode: How to treat aggregation latents a query leaves
            undetermined.
        :return: What grounds this circuit in the circuit type it was fitted in.
        """
        return RelationalGrounding(self, grounding_mode)
