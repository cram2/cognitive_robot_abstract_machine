"""
Relational probabilistic circuits ("RSPNs").

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
from sortedcontainers import SortedSet
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Optional,
    Type,
    TypeVar,
)

from krrood.ormatic.data_access_objects.dao import (
    DataAccessObjectSchema,
    get_dao_schema,
)
from krrood.ormatic.data_access_objects.helper import get_data_access_object_class
from krrood.parametrization.feature_extraction.aggregations import (
    compute_aggregation_statistics,
)
from krrood.parametrization.feature_extraction.feature_extractor import FeatureExtractor

if TYPE_CHECKING:
    from krrood.entity_query_language.query.match import Match
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import LearningMethod
from probabilistic_model.learning.jpt.variables import (
    infer_variables_from_dataframe,
)
from probabilistic_model.probabilistic_model import PartialPointType
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    CircuitNotFittedError,
    ClassCircuitGroundingFailedError,
    PartCircuitGroundingFailedError,
)
from probabilistic_model.probabilistic_circuit.relational.exchangeable_grounding import (
    ExchangeablePartGrounder,
    GroundedCircuit,
    GroundedPartTemplate,
    GroundingMode,
)
from probabilistic_model.adapters.rustworkx_tensorized.rustworkx_to_tensorized import (
    RustworkxCircuitToLayeredCircuitConverter,
)
from probabilistic_model.probabilistic_circuit.relational.layered_grounding import (
    LayeredExchangeablePartGrounder,
)
from probabilistic_model.probabilistic_circuit.relational.rustworkx_grounding import (
    RustworkxExchangeablePartGrounder,
)
from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.helper import (
    find_lowest_product_nodes_that_model_variables,
)
from probabilistic_model.probabilistic_circuit.relational.template import (
    RelationalDistributionTemplate,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
)
from random_events.interval import Interval
from random_events.variable import Variable

Grounder = TypeVar("Grounder", bound=ExchangeablePartGrounder)


def _is_concrete_statistic(variable: Variable, value: Any) -> bool:
    """
    Decide whether an aggregation value pins its variable to a single point.

    :param variable: The latent variable the value belongs to.
    :param value: The observed aggregation value, either a concrete point or a range.
    :return:``True`` if the value designates exactly one element of the variable's
        domain.
    """
    composite = variable.make_value(value)
    if isinstance(composite, Interval):
        return composite.is_singleton()
    return len(composite.simple_sets) == 1


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
        self, parts: list[Match], ground: Callable[[Match], GroundedCircuit]
    ) -> list[GroundedPartTemplate[GroundedCircuit]]:
        """
        :param parts: The query parts, one per child object of the relation.
        :param ground: Grounds the template for one query part.
        :return: The grounded template of every part. Parts whose queries have the same
            shape share one grounding.
        """
        groundings: dict[Hashable, tuple[GroundedCircuit, str]] = {}
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
        :return: The variable of the part. A variable of the template's class circuit
            is put under the namespace of the part. A variable of an exchangeable
            relation of the template, which its grounding names by its full query path,
            is moved from the namespace the template was grounded for to that of the
            part.
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

    def ground(
        self, parts_to_ground: list[Match], aggregation_statistics: PartialPointType
    ) -> ProbabilisticCircuit:
        """
        Build a product circuit by grounding each exchangeable part independently.

        :param parts_to_ground: The query parts, one per child object in the relation.
        :param aggregation_statistics: Observed aggregation values shared across all
            parts.
        :return: A product circuit over the grounded distributions of all parts.
        """
        return self.ground_instance(
            self.grounded_templates_of_parts(
                parts_to_ground, self.template_distribution.ground
            ),
            aggregation_statistics,
        )

    def ground_instance(
        self,
        parts: list[GroundedPartTemplate[ProbabilisticCircuit]],
        aggregation_statistics: PartialPointType,
    ) -> ProbabilisticCircuit:
        """
        :param parts: The grounded template of every part.
        :param aggregation_statistics: Observed aggregation values shared across all
            parts.
        :return: A product circuit over the conditioned templates of all parts.
        """
        result = ProbabilisticCircuit()
        root = ProductUnit(probabilistic_circuit=result)
        for part in parts:
            root.add_subcircuit(
                self._mount_part(
                    result, self._conditioned_part(part, aggregation_statistics)
                )
            )
        return result

    def _conditioned_part(
        self,
        part: GroundedPartTemplate[ProbabilisticCircuit],
        aggregation_statistics: PartialPointType,
    ) -> ProbabilisticCircuit:
        """
        Condition a copy of the grounded template of a part on
        ``aggregation_statistics`` and remove the latent variables.

        :param part: The grounded template of the part.
        :param aggregation_statistics: Observed aggregation values to condition on.
        :return: A self-contained circuit over the variables of the part, ready to be
            mounted into the parent.
        """
        part_circuit = part.circuit.__deepcopy__()
        conditioning_result, _ = part_circuit.log_conditional_in_place(
            aggregation_statistics, preserve_structure=True
        )
        if conditioning_result is None:
            part_circuit = part.circuit.__deepcopy__()
        non_latent_variables = [
            variable
            for variable in part_circuit.variables
            if variable not in self.latent_variables
        ]
        part_circuit.restrict_to_variables_in_place(non_latent_variables)
        part_circuit.update_variables(
            {
                variable: self.variable_of_part(variable, part)
                for variable in part_circuit.variables
            }
        )
        if len(part_circuit.nodes()) == 0:
            raise PartCircuitGroundingFailedError(self.template_distribution.class_)
        return part_circuit


@dataclass
class RelationalProbabilisticCircuit:
    """
    A probabilistic circuit that jointly models a class and its relational structure.
    """

    class_: Type
    """
    The domain class whose instances this distribution models.
    """

    class_probabilistic_circuit: Optional[ProbabilisticCircuit] = None
    """
    The fitted joint distribution over the class's scalar attributes and aggregation
    statistics, populated by ``fit``.
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

    part_learning_methods: dict[str, LearningMethod] = field(default_factory=dict)
    """
    Per exchangeable-part field name, what that part's template distribution is fitted
    with.

    A part absent from the mapping is fitted with a plain
    :class:`~probabilistic_model.learning.jpt.jpt.JointProbabilityTree`.
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
        attributes). Column names are the access-path names produced by
        :meth:`~krrood.entity_query_language.core.mapped_variable.MappedVariable.get_clean_name_from_mapped_variable`
        so that, after part-prefix renaming, they align with the krrood access-path convention.

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
        child_aggregation_features = {
            id(feature)
            for features in child_feature_extractor.exchangeable_features.values()
            for feature in features
        }
        # the child's own aggregation statistics keep the name the latent variables of
        # its exchangeable templates are given, so grounding the child can find them
        child_column_names = [
            (
                feature._name_
                if id(feature) in child_aggregation_features
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
        child_type = type(child_instances[0])
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
                child_type,
                learning_method=self.part_learning_methods.get(
                    exchangeable_part, JointProbabilityTree()
                ),
            ),
            latent_variables,
        )
        template.template_distribution.fit(
            child_instances, dataframe_from_parent=child_dataframe
        )
        return template

    def fit(
        self,
        instances: list[Any],
        dataframe_from_parent: Optional[pd.DataFrame] = None,
    ):
        """
        Fit the relational probabilistic circuit from a list of domain objects.

        Builds a ``FeatureExtractor``, fits the class-level circuit on the class-level
        features with :attr:`learning_method`, and then recursively fits one
        ``ExchangeableDistributionTemplate`` per exchangeable part discovered in the
        schema.

        :param instances: Training instances; all must be of the same class.
        :param dataframe_from_parent: Pre-built dataframe supplied by a parent
            ``_fit_exchangeable_part`` call. When provided, feature extraction and
            preprocessing are skipped.
        :return:``self``, to allow chaining.
        """
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
        for collection_relationship in self.schema_information.collection_relationships:
            exchangeable_part = collection_relationship.key
            if exchangeable_part not in self.feature_extractor.exchangeable_features:
                continue
            self.exchangeable_distribution_templates[exchangeable_part] = (
                self._fit_exchangeable_part(exchangeable_part, instances)
            )
        return self

    def _condition_class_circuit(
        self,
        circuit: ProbabilisticCircuit,
        aggregation_statistics: PartialPointType,
        latent_variables: list[Variable],
    ) -> tuple[ProbabilisticCircuit, list[ProductUnit]]:
        """
        Condition the class circuit on aggregation statistics, keeping its structure so
        that its leaf products stay the mounting points.

        Statistics the circuit deems impossible leave it as it is, together with what
        the exchangeable parts grounded before have attached to it.

        :param circuit: The current working copy of the class circuit.
        :param aggregation_statistics: Observed aggregation values to condition on.
        :param latent_variables: Variables that link the class circuit to the
            exchangeable distribution template.
        :return: The conditioned circuit and the product nodes that will be extended
            with the grounded exchangeable distribution.
        """
        if self._can_condition_on(circuit, aggregation_statistics):
            circuit.log_conditional_in_place(
                aggregation_statistics, preserve_structure=True
            )
        if len(circuit.nodes()) == 0:
            raise ClassCircuitGroundingFailedError(self.class_)
        product_nodes_to_extend = find_lowest_product_nodes_that_model_variables(
            circuit, SortedSet(latent_variables)
        )
        return circuit, product_nodes_to_extend

    @staticmethod
    def _can_condition_on(
        circuit: ProbabilisticCircuit, aggregation_statistics: PartialPointType
    ) -> bool:
        """
        :param circuit: The current working copy of the class circuit.
        :param aggregation_statistics: Observed aggregation values.
        :return: Whether there are values and the circuit deems them possible, which
            conditioning the circuit itself only tells after destroying it.
        """
        if not aggregation_statistics:
            return False
        statistics_circuit = circuit.marginal(aggregation_statistics)
        if statistics_circuit is None:
            return True
        conditioned, _ = statistics_circuit.log_conditional_in_place(
            aggregation_statistics
        )
        return conditioned is not None

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
    ) -> ProbabilisticCircuit:
        """
        Ground the relational circuit for a specific query.

        Starting from a deep copy of ``class_probabilistic_circuit``, each exchangeable
        part's template is grounded for the objects specified in the query and attached
        to the conditioning product nodes of the class circuit.

        :param query: An underspecified, resolved query instance whose structure
            determines which parts are grounded and how many child objects each
            exchangeable relation contains.
        :param grounding_mode: How to treat aggregation latents the query leaves
            undetermined. See :class:`GroundingMode`.
        :return: A concrete ``ProbabilisticCircuit`` over all variables implied by the
            query.
        :raises CircuitNotFittedError: If ``ground`` is called before ``fit``.
        """
        if self.class_probabilistic_circuit is None:
            raise CircuitNotFittedError(self.class_)
        circuit = self.class_probabilistic_circuit.__deepcopy__()
        instance = query.construct_instance()
        for (
            exchangeable_part_name,
            template,
        ) in self.exchangeable_distribution_templates.items():
            grounder = self._exchangeable_part_grounder(
                RustworkxExchangeablePartGrounder,
                circuit,
                exchangeable_part_name,
                template,
                query,
                instance,
            )
            circuit = grounder.ground(grounding_mode)
        return circuit

    def ground_layered(
        self,
        query: Match,
        grounding_mode: GroundingMode = GroundingMode.SAMPLED,
    ) -> LayeredProbabilisticCircuit:
        """
        Ground the relational circuit for a specific query into a layered circuit.

        The grounded distribution is the one :meth:`ground` creates. The instances of an
        exchangeable relation are the fitted template conditioned on different
        aggregation statistics, which only changes its weights, so they are built as one
        stack of layers rather than one circuit per instance and child object.

        :param query: An underspecified, resolved query instance whose structure
            determines which parts are grounded and how many child objects each
            exchangeable relation contains.
        :param grounding_mode: How to treat aggregation latents the query leaves
            undetermined. See :class:`GroundingMode`.
        :return: A layered circuit over all variables implied by the query.
        :raises CircuitNotFittedError: If ``ground_layered`` is called before ``fit``.
        """
        if self.class_probabilistic_circuit is None:
            raise CircuitNotFittedError(self.class_)
        circuit = self.class_probabilistic_circuit.__deepcopy__()
        instance = query.construct_instance()
        parts = []
        for (
            exchangeable_part_name,
            template,
        ) in self.exchangeable_distribution_templates.items():
            grounder = self._exchangeable_part_grounder(
                LayeredExchangeablePartGrounder,
                circuit,
                exchangeable_part_name,
                template,
                query,
                instance,
            )
            circuit = grounder.circuit
            parts.append(grounder.ground(grounding_mode))
        parts = [part.without_removed_mounting_nodes(circuit) for part in parts]

        converted = RustworkxCircuitToLayeredCircuitConverter.convert_with_layers(
            circuit
        )
        variables = SortedSet(converted.circuit.variables)
        for part in parts:
            variables.update(part.instances.variables)
        converted.circuit.restore_variables(variables)
        for part in parts:
            part.attach_to(converted, variables)
        converted.circuit.reset_scopes()
        return converted.circuit

    def _exchangeable_part_grounder(
        self,
        grounder_type: Type[Grounder],
        circuit: ProbabilisticCircuit,
        exchangeable_part_name: str,
        template: ExchangeableDistributionTemplate,
        query: Match,
        instance: Any,
    ) -> Grounder:
        """
        Condition the class circuit on the aggregation statistics of one exchangeable
        part that the query determines, and collect what grounding the part needs.

        :param grounder_type: What grounds the part.
        :param circuit: The current working copy of the class circuit.
        :param exchangeable_part_name: Field name of the exchangeable relation.
        :param template: The fitted template for this relation.
        :param query: The grounding query.
        :param instance: The concrete instance constructed from the query.
        :return: The grounder of the part, holding the conditioned class circuit.
        """
        aggregation_statistics = compute_aggregation_statistics(
            instance, exchangeable_part_name, template.latent_variables
        )
        determined_statistics = {
            variable: value
            for variable, value in aggregation_statistics.items()
            if _is_concrete_statistic(variable, value)
        }
        undetermined_latents = SortedSet(
            variable
            for variable in template.latent_variables
            if variable not in determined_statistics
        )
        circuit, product_nodes_to_extend = self._condition_class_circuit(
            circuit, determined_statistics, template.latent_variables
        )
        return grounder_type(
            circuit=circuit,
            product_nodes_to_extend=product_nodes_to_extend,
            template=template,
            query_parts=query._kwargs_[exchangeable_part_name],
            determined_statistics=determined_statistics,
            undetermined_latents=undetermined_latents,
            monte_carlo_sample_count=self.monte_carlo_sample_count,
        )
