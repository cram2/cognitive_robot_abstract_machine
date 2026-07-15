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

import numpy as np
import pandas as pd
from sortedcontainers import SortedSet
from typing_extensions import TYPE_CHECKING, Any, Optional, Type

from krrood.entity_query_language.query.match import AbstractMatchExpression
from krrood.ormatic.data_access_objects.dao import (
    DataAccessObject,
    DataAccessObjectSchema,
    get_dao_schema,
)
from krrood.parametrization.feature_extraction.aggregations import (
    compute_aggregation_statistics,
)
from krrood.parametrization.feature_extraction.feature_extractor import FeatureExtractor

if TYPE_CHECKING:
    from krrood.entity_query_language.query.match import Match
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    CircuitNotFittedError,
    InvalidMonteCarloSampleCountError,
    UndeterminedLatentsNotModeledError,
)
from probabilistic_model.probabilistic_circuit.relational.helper import (
    find_lowest_product_nodes_that_model_variables,
)
from krrood.utils import get_class_and_attribute_name
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    Unit,
)
from random_events.interval import Interval
from random_events.variable import Variable


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


def _rename_variables_with_part_prefix(
    circuit: ProbabilisticCircuit,
    prefix: str,
    excluded_variables: list[Variable],
) -> None:
    """
    Rename each variable in the circuit to include ``prefix`` as a namespace.

    Produces names of the form ``"{prefix}.{variable.name}"``. Variables listed in
    ``excluded_variables`` are left unchanged.

    :param circuit: The circuit whose variables are renamed in-place.
    :param prefix: String prefix to prepend to every variable name.
    :param excluded_variables: Variables that should keep their current names.
    """
    variable_renames = {
        variable: type(variable)(
            get_class_and_attribute_name(prefix, variable.name), domain=variable.domain
        )
        for variable in circuit.variables
        if variable not in excluded_variables
    }
    circuit.update_variables(variable_renames)


@dataclass
class ExchangeableDistributionTemplate:
    """
    A fitted distribution template for one exchangeable (many-to-many) relation.

    Wraps a ``RelationalProbabilisticCircuit`` that was trained on the child objects of
    the relation together with the parent's aggregation statistics as latent context
    variables.
    """

    template_distribution: RelationalProbabilisticCircuit
    """
    The fitted ``RelationalProbabilisticCircuit`` representing the child distribution.
    """

    latent_variables: list[Variable] = field(default_factory=list)
    """
    Variables shared between the parent and child circuits that are used for
    conditioning but are not part of the final grounded distribution.
    """

    def _ground_part_circuit(
        self, part, aggregation_statistics: dict[Variable, Any], index: int = 0
    ) -> ProbabilisticCircuit:
        """
        Ground and prepare the circuit for a single exchangeable part.

        Conditions the template circuit on ``aggregation_statistics``, marginalizes away
        the latent variables, renames surviving variables with the part's prefix, and
        reindexes the graph for safe mounting.

        :param part: The query part (a ``Match`` or a concrete domain object).
        :param aggregation_statistics: Observed aggregation values to condition on.
        :param index: Position of this part in its parent list; used as fallback prefix
            when ``part`` does not carry a symbolic variable.
        :return: A self-contained circuit ready to be mounted into the parent.
        """
        part_circuit = self.template_distribution.ground(part)
        conditioning_result, _ = part_circuit.log_conditional_in_place(
            aggregation_statistics
        )
        if conditioning_result is None:
            part_circuit = self.template_distribution.ground(part)
        non_latent_variables = [
            variable
            for variable in part_circuit.variables
            if variable not in self.latent_variables
        ]
        part_circuit.marginal_in_place(non_latent_variables)
        prefix = (
            str(part.variable)
            if isinstance(part, AbstractMatchExpression)
            else str(index)
        )
        _rename_variables_with_part_prefix(part_circuit, prefix, self.latent_variables)
        if len(part_circuit.nodes()) == 0:
            raise ValueError("The grounding of the part failed.")
        return part_circuit

    def ground(
        self, parts_to_ground: list, aggregation_statistics: dict[Variable, Any]
    ) -> ProbabilisticCircuit:
        """
        Build a product circuit by grounding each exchangeable part independently.

        :param parts_to_ground: The query parts, one per child object in the relation.
        :param aggregation_statistics: Observed aggregation values shared across all
            parts.
        :return: A product circuit over the grounded distributions of all parts.
        """
        result = ProbabilisticCircuit()
        root = ProductUnit(probabilistic_circuit=result)
        for index, part in enumerate(parts_to_ground):
            part_circuit = self._ground_part_circuit(
                part, aggregation_statistics, index
            )
            part_root_index = part_circuit.root.index
            node_index_map = result.mount(part_circuit.root)
            root.add_subcircuit(node_index_map[part_root_index])
        return result


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
    """

    @staticmethod
    def _build_class_dataframe(
        feature_extractor: FeatureExtractor,
        instances: list[DataAccessObject],
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
        instances: list[DataAccessObject],
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
            for association in getattr(instance, exchangeable_part):
                child_features = child_feature_extractor.apply_mapping(
                    association.target
                )
                rows.append(aggregation_row + child_features)
        child_column_names = [
            f.get_clean_name_from_mapped_variable()
            for f in child_feature_extractor.features
        ]
        return pd.DataFrame(columns=aggregation_names + child_column_names, data=rows)

    def _fit_exchangeable_part(
        self,
        exchangeable_part: str,
        instances: list[DataAccessObject],
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

        child_instances = [
            association.target
            for association in itertools.chain.from_iterable(
                getattr(instance, exchangeable_part) for instance in instances
            )
        ]
        child_type = type(getattr(instances[0], exchangeable_part)[0].target)
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
            RelationalProbabilisticCircuit(child_type),
            latent_variables,
        )
        template.template_distribution.fit(
            child_instances, dataframe_from_parent=child_dataframe
        )
        return template

    def fit(
        self,
        instances: list[DataAccessObject],
        dataframe_from_parent: Optional[pd.DataFrame] = None,
    ):
        """
        Fit the relational probabilistic circuit from a list of DAO instances.

        Builds a ``FeatureExtractor``, trains a ``JointProbabilityTree`` on the class-
        level features, and then recursively fits one
        ``ExchangeableDistributionTemplate`` per exchangeable part discovered in the
        schema.

        :param instances: Training instances; all must share the same DAO class.
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
        self.class_probabilistic_circuit = JointProbabilityTree(
            annotated_variables=variables
        ).fit(class_dataframe)
        self.schema_information = get_dao_schema(type(instances[0]))
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
        aggregation_statistics: dict[Variable, Any],
        latent_variables: list[Variable],
    ) -> tuple[ProbabilisticCircuit, list[ProductUnit]]:
        """
        Condition the class circuit on aggregation statistics.

        :param circuit: The current working copy of the class circuit.
        :param aggregation_statistics: Observed aggregation values to condition on.
        :param latent_variables: Variables that link the class circuit to the
            exchangeable distribution template.
        :return: The conditioned circuit and the surviving product nodes that will be
            extended with the grounded exchangeable distribution.
        """
        product_nodes_to_extend = find_lowest_product_nodes_that_model_variables(
            circuit, SortedSet(latent_variables)
        )
        conditioning_result, _ = circuit.log_conditional_in_place(
            aggregation_statistics
        )
        if conditioning_result is None:
            circuit = self.class_probabilistic_circuit.__deepcopy__()
            product_nodes_to_extend = find_lowest_product_nodes_that_model_variables(
                circuit, SortedSet(latent_variables)
            )
        if len(circuit.nodes()) == 0:
            raise ValueError("The grounding of the class failed.")
        surviving_product_nodes = [
            node for node in product_nodes_to_extend if node.index is not None
        ]
        return circuit, surviving_product_nodes

    def ground(self, query: Match) -> ProbabilisticCircuit:
        """
        Ground the relational circuit for a specific query.

        Starting from a deep copy of ``class_probabilistic_circuit``, each exchangeable
        part's template is grounded for the objects specified in the query and attached
        to the conditioning product nodes of the class circuit.

        :param query: An underspecified, resolved query instance whose structure
            determines which parts are grounded and how many child objects each
            exchangeable relation contains.
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
            circuit = self._ground_exchangeable_part(
                circuit, exchangeable_part_name, template, query, instance
            )
        return circuit

    def _ground_exchangeable_part(
        self,
        circuit: ProbabilisticCircuit,
        exchangeable_part_name: str,
        template: ExchangeableDistributionTemplate,
        query: Match,
        instance: Any,
    ) -> ProbabilisticCircuit:
        """
        Ground one exchangeable part and attach it to the class circuit.

        Aggregation statistics determinable from the query condition the class circuit
        directly. Monte-Carlo integrates out undetermined statistics: they are sampled
        from the conditioned class circuit and each sampled value yields its own
        exchangeable distribution instance.

        :param circuit: The current working copy of the class circuit.
        :param exchangeable_part_name: Field name of the exchangeable relation.
        :param template: The fitted template for this relation.
        :param query: The grounding query.
        :param instance: The concrete instance constructed from the query.
        :return: The class circuit extended with the grounded exchangeable part.
        """
        aggregation_statistics = compute_aggregation_statistics(
            instance,
            self.feature_extractor.exchangeable_features[exchangeable_part_name],
            template.latent_variables,
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
        query_parts = query.kwargs[exchangeable_part_name]

        sampled_assignments = self._sample_undetermined_latents(
            circuit, undetermined_latents
        )
        if not sampled_assignments:
            self._attach_single_exchangeable_instance(
                circuit,
                product_nodes_to_extend,
                template,
                query_parts,
                determined_statistics,
            )
            return circuit

        self._attach_monte_carlo_mixture(
            circuit,
            product_nodes_to_extend,
            template,
            query_parts,
            determined_statistics,
            undetermined_latents,
            sampled_assignments,
        )
        return circuit

    def _sample_undetermined_latents(
        self,
        conditioned_circuit: ProbabilisticCircuit,
        undetermined_latents: SortedSet[Variable],
    ) -> list[dict[Variable, Any]]:
        """
        Draw the distinct values of the undetermined latents to integrate over.

        Samples ``monte_carlo_sample_count`` joint assignments of the undetermined
        latents from the conditioned class circuit and deduplicates them, so that each
        distinct value is grounded only once.

        :param conditioned_circuit: The class circuit conditioned on the determined
            statistics.
        :param undetermined_latents: The latent variables that could not be determined
            from the query.
        :return: One value assignment per distinct sampled point, empty when there are
            no undetermined latents to integrate out.
        :raises InvalidMonteCarloSampleCountError: If there are undetermined latents but
            the sample count is not positive.
        :raises UndeterminedLatentsNotModeledError: If the conditioned class circuit
            does not model the undetermined latents and thus cannot be sampled from.
        """
        if not undetermined_latents:
            return []
        if self.monte_carlo_sample_count < 1:
            raise InvalidMonteCarloSampleCountError(self.monte_carlo_sample_count)
        proposal = conditioned_circuit.marginal(undetermined_latents)
        if proposal is None:
            raise UndeterminedLatentsNotModeledError(list(undetermined_latents))
        samples = proposal.sample(self.monte_carlo_sample_count)
        index_of_variable = proposal.variable_to_index_map
        unique_rows = {tuple(row) for row in samples.tolist()}
        return [
            {
                variable: row[index_of_variable[variable]]
                for variable in undetermined_latents
            }
            for row in (np.array(unique_row) for unique_row in unique_rows)
        ]

    @staticmethod
    def _node_local_latent_log_likelihoods(
        product_node: ProductUnit,
        undetermined_latents: SortedSet[Variable],
        latent_assignments: list[dict[Variable, Any]],
    ) -> list[float]:
        """
        Log-likelihoods of latent assignments local to a mounting product node.

        Marginalizes the subcircuit rooted at ``product_node`` to the undetermined
        latents once, then evaluates every assignment in a single batched pass.

        :param product_node: The mounting product node.
        :param undetermined_latents: The latent variables sampled by Monte-Carlo.
        :param latent_assignments: The sampled assignments of those latents.
        :return: One log-likelihood per assignment, in input order.
        """
        subcircuit = ProbabilisticCircuit()
        subcircuit.mount(product_node)
        subcircuit.marginal_in_place(undetermined_latents)
        index_of_variable = subcircuit.variable_to_index_map
        events = np.full((len(latent_assignments), len(index_of_variable)), np.nan)
        for row, assignment in enumerate(latent_assignments):
            for variable, value in assignment.items():
                events[row, index_of_variable[variable]] = value
        return [
            float(log_likelihood)
            for log_likelihood in subcircuit.log_likelihood(events)
        ]

    @staticmethod
    def _mount_instance(
        circuit: ProbabilisticCircuit,
        template: ExchangeableDistributionTemplate,
        query_parts: list,
        aggregation_statistics: dict[Variable, Any],
    ) -> Unit:
        """
        Ground one exchangeable instance and mount it into the class circuit.

        :param circuit: The working class circuit to mount into.
        :param template: The fitted template for this relation.
        :param query_parts: The query parts, one per child object.
        :param aggregation_statistics: Statistics to condition the instance on.
        :return: The root of the mounted instance, owned by ``circuit``.
        """
        grounded = template.ground(query_parts, aggregation_statistics)
        node_index_map = circuit.mount(grounded.root)
        return node_index_map[grounded.root.index]

    @staticmethod
    def _attach_single_exchangeable_instance(
        circuit: ProbabilisticCircuit,
        product_nodes_to_extend: list[ProductUnit],
        template: ExchangeableDistributionTemplate,
        query_parts: list,
        aggregation_statistics: dict[Variable, Any],
    ) -> None:
        """
        Attach one grounded exchangeable instance to every mounting product node.

        The instance is mounted once and shared as a child of every node.

        :param circuit: The working class circuit.
        :param product_nodes_to_extend: The mounting product nodes.
        :param template: The fitted template for this relation.
        :param query_parts: The query parts, one per child object.
        :param aggregation_statistics: Statistics to condition the instance on.
        """
        instance_root = RelationalProbabilisticCircuit._mount_instance(
            circuit, template, query_parts, aggregation_statistics
        )
        for product_node in product_nodes_to_extend:
            product_node.add_subcircuit(instance_root)

    def _attach_monte_carlo_mixture(
        self,
        circuit: ProbabilisticCircuit,
        product_nodes_to_extend: list[ProductUnit],
        template: ExchangeableDistributionTemplate,
        query_parts: list,
        determined_statistics: dict[Variable, Any],
        undetermined_latents: SortedSet[Variable],
        sampled_assignments: list[dict[Variable, Any]],
    ) -> None:
        """
        Attach a Monte-Carlo mixture over undetermined aggregation statistics.

        For every sampled assignment one exchangeable instance is grounded on the
        determined plus sampled statistics. The undetermined latents are then
        marginalized out of the class circuit so their distribution is carried solely by
        the mixture weights, which are the node-local likelihoods of the sampled values.
        Each mounting product node receives its own normalized sum unit over the
        instances.

        :param circuit: The working class circuit.
        :param product_nodes_to_extend: The mounting product nodes.
        :param template: The fitted template for this relation.
        :param query_parts: The query parts, one per child object.
        :param determined_statistics: Statistics determinable from the query.
        :param undetermined_latents: The latents integrated out by Monte-Carlo.
        :param sampled_assignments: Distinct sampled values of the undetermined latents.
        """
        log_weights_per_node = [
            self._node_local_latent_log_likelihoods(
                product_node, undetermined_latents, sampled_assignments
            )
            for product_node in product_nodes_to_extend
        ]
        retained_variables = SortedSet(circuit.variables) - undetermined_latents
        circuit.marginal_in_place(retained_variables)
        mounted_roots = [
            self._mount_instance(
                circuit, template, query_parts, {**determined_statistics, **assignment}
            )
            for assignment in sampled_assignments
        ]
        for product_node, log_weights in zip(
            product_nodes_to_extend, log_weights_per_node
        ):
            self._attach_mixture_to_node(
                circuit, product_node, mounted_roots, log_weights
            )

    @staticmethod
    def _attach_mixture_to_node(
        circuit: ProbabilisticCircuit,
        product_node: ProductUnit,
        instance_roots: list[Unit],
        log_weights: list[float],
    ) -> None:
        """
        Attach a normalized sum unit over exchangeable instances to one node.

        Instances whose node-local likelihood is zero are skipped. The instances are
        already mounted in ``circuit`` and shared across all mounting nodes; only the
        weighted sum-unit edges differ per node.

        :param circuit: The working class circuit.
        :param product_node: The mounting product node to extend.
        :param instance_roots: The roots of the mounted exchangeable instances.
        :param log_weights: The node-local log-likelihood weight of each instance.
        """
        weighted_instances = [
            (instance_root, log_weight)
            for instance_root, log_weight in zip(instance_roots, log_weights)
            if log_weight > -np.inf
        ]
        if not weighted_instances:
            weighted_instances = [(instance_roots[0], 0.0)]
        sum_unit = SumUnit(probabilistic_circuit=circuit)
        product_node.add_subcircuit(sum_unit)
        for instance_root, log_weight in weighted_instances:
            sum_unit.add_subcircuit(instance_root, log_weight)
        sum_unit.normalize()
