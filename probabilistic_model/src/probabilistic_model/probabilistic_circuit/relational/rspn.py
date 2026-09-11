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

import enum
import itertools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sortedcontainers import SortedSet
from typing_extensions import TYPE_CHECKING, Any, Optional, Type

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
from probabilistic_model.distributions.helper import make_dirac
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import (
    AnnotatedVariable,
    infer_variables_from_dataframe,
)
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    CircuitNotFittedError,
    ClassCircuitGroundingFailedError,
    InvalidMonteCarloSampleCountError,
    PartCircuitGroundingFailedError,
    UndeterminedLatentsNotModeledError,
    UndeterminedLatentsNotPartitionedError,
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
    SumUnit,
    Unit,
    leaf,
)
from random_events.interval import Interval
from random_events.variable import Variable

logger = logging.getLogger(__name__)


class GroundingMode(enum.IntEnum):
    """
    Selects how ``RelationalProbabilisticCircuit.ground`` represents an exchangeable
    relation's aggregation latents that a query leaves undetermined.

    Undetermined latents are always retained as variables of the grounded circuit --
    never integrated out -- so any caller that only wants the query's own variables
    marginalizes the ones it does not need afterward, as a postprocessing step, rather
    than grounding deciding that for it.
    """

    SAMPLED = enum.auto()
    """
    Retain each undetermined latent as a point-valued variable at its Monte-Carlo
    sampled value.

    Default behaviour.
    """

    EXACT = enum.auto()
    """
    Retain undetermined latents by enumerating the fitted circuit's own exact, disjoint
    partition over them instead of sampling.

    Reproducible across calls and covers the whole domain the model learned about,
    unlike :attr:`SAMPLED`. Falls back to :attr:`SAMPLED`, with a logged warning, when
    the fitted circuit's partition over the undetermined latents is not itself disjoint.
    """


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

    def _ground_part_circuit(
        self, part: Match, aggregation_statistics: dict[Variable, Any], index: int = 0
    ) -> ProbabilisticCircuit:
        """
        Ground and prepare the circuit for a single exchangeable part.

        Conditions the template circuit on ``aggregation_statistics``, marginalizes away
        the latent variables, renames surviving variables with the part's prefix, and
        reindexes the graph for safe mounting.

        :param part: The query part being grounded.
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
        prefix = self._prefix_for_part(part, index)
        part_circuit.rename_variables_with_prefix(prefix, self.latent_variables)
        if len(part_circuit.nodes()) == 0:
            raise PartCircuitGroundingFailedError(self.template_distribution.class_)
        return part_circuit

    def ground(
        self, parts_to_ground: list[Match], aggregation_statistics: dict[Variable, Any]
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
            root.add_subcircuit(self._mount_part(result, part_circuit))
        return result


@dataclass
class ExchangeablePartGrounder:
    """
    Grounds one exchangeable part into the mounting product nodes of a class circuit.

    Bundles the circuit, template, and query context one exchangeable part's grounding
    needs, so the different ways of representing undetermined aggregation latents
    (:attr:`GroundingMode.SAMPLED`, :attr:`GroundingMode.EXACT`) read and mutate that
    context through fields instead of threading the same handful of arguments through a
    chain of static methods.
    """

    circuit: ProbabilisticCircuit
    """
    The working class circuit being extended.
    """

    product_nodes_to_extend: list[ProductUnit]
    """
    The class circuit's product nodes that mount the grounded instance(s).
    """

    template: RelationalDistributionTemplate
    """
    The fitted template whose exchangeable relation is being grounded.
    """

    query_parts: list[Match]
    """
    The query parts, one per child object in the relation.
    """

    determined_statistics: dict[Variable, Any]
    """
    Aggregation statistics determinable from the query.
    """

    undetermined_latents: SortedSet[Variable] = field(default_factory=SortedSet)
    """
    Latent variables the query leaves undetermined.
    """

    monte_carlo_sample_count: int = 10
    """
    Number of Monte-Carlo samples drawn to integrate out ``undetermined_latents``.
    """

    def attach_single_instance(self) -> None:
        """
        Attach one grounded exchangeable instance to every mounting product node.

        The instance is mounted once and shared as a child of every node.
        """
        instance_root = self._mount_instance(self.determined_statistics)
        for product_node in self.product_nodes_to_extend:
            product_node.add_subcircuit(instance_root)

    def attach_monte_carlo_mixture(self) -> None:
        """
        Attach a Monte-Carlo mixture over undetermined aggregation statistics.

        For every sampled assignment one exchangeable instance is grounded on the
        determined plus sampled statistics, and additionally carries its sampled latent
        values back as point-valued variables instead of leaving them discarded. Each
        mounting product node receives its own normalized sum unit over the instances,
        weighted by the node-local likelihoods of the sampled values.
        """
        sampled_assignments = self._sample_undetermined_latents()
        log_weights_per_node = [
            self._node_local_latent_log_likelihoods(
                product_node, self.undetermined_latents, sampled_assignments
            )
            for product_node in self.product_nodes_to_extend
        ]
        retained_variables = (
            SortedSet(self.circuit.variables) - self.undetermined_latents
        )
        self.circuit.restrict_to_variables_in_place(retained_variables)
        mounted_roots = [
            self._mount_instance_with_retained_latents(assignment)
            for assignment in sampled_assignments
        ]
        for product_node, log_weights in zip(
            self.product_nodes_to_extend, log_weights_per_node
        ):
            self._attach_mixture_to_node(product_node, mounted_roots, log_weights)

    def attach_exact_partition_mixture(self) -> None:
        """
        Attach a mixture over the undetermined latents' own exact partition.

        Unlike Monte-Carlo sampling, this enumerates ``circuit``'s already-fitted,
        exact partition over ``undetermined_latents`` (the branches of
        ``circuit.marginal(undetermined_latents)``'s root) instead of drawing samples
        from it: reproducible across calls, and covers every value the model learned
        about rather than only whichever points got sampled. One exchangeable instance
        is grounded per branch and retains that branch's full region -- not narrowed to
        a point -- by mounting the branch itself alongside the grounded instance.

        :raises UndeterminedLatentsNotModeledError: If ``circuit`` does not model
            ``undetermined_latents`` and thus has no partition over them.
        :raises UndeterminedLatentsNotPartitionedError: If that partition's branches
            are not pairwise disjoint, so exact-partition grounding would not be
            support-deterministic.
        """
        proposal = self.circuit.marginal(self.undetermined_latents)
        if proposal is None:
            raise UndeterminedLatentsNotModeledError(list(self.undetermined_latents))
        if not self._undetermined_latents_partition_disjointly(proposal):
            raise UndeterminedLatentsNotPartitionedError(
                list(self.undetermined_latents)
            )

        # the precondition just verified guarantees proposal.root is a SumUnit with at
        # least two branches
        branches = proposal.root.log_weighted_subcircuits
        # _undetermined_latents_partition_disjointly already called proposal.support
        # above, caching each branch's region on it as result_of_current_query
        branch_regions = [branch.result_of_current_query for _, branch in branches]

        # each node's weights must be read off circuit before undetermined_latents are
        # stripped from it below -- product_node stops modeling them afterward
        log_weights_per_node = [
            self._node_local_branch_log_probabilities(
                product_node, self.undetermined_latents, branch_regions
            )
            for product_node in self.product_nodes_to_extend
        ]

        retained_variables = (
            SortedSet(self.circuit.variables) - self.undetermined_latents
        )
        self.circuit.restrict_to_variables_in_place(retained_variables)

        mounted_roots = []
        for _, latent_branch in branches:
            representative_value = self._representative_value(
                latent_branch, self.undetermined_latents
            )
            instance_root = self._mount_instance(
                {**self.determined_statistics, **representative_value}
            )
            branch_root = ProductUnit(probabilistic_circuit=self.circuit)
            branch_root.add_subcircuit(instance_root)
            mounted_branch_nodes = self.circuit.mount(latent_branch)
            branch_root.add_subcircuit(mounted_branch_nodes[latent_branch.index])
            mounted_roots.append(branch_root)

        for product_node, log_weights in zip(
            self.product_nodes_to_extend, log_weights_per_node
        ):
            self._attach_mixture_to_node(product_node, mounted_roots, log_weights)

    def _mount_instance(self, aggregation_statistics: dict[Variable, Any]) -> Unit:
        """
        Ground one exchangeable instance and mount it into the class circuit.

        :param aggregation_statistics: Statistics to condition the instance on.
        :return: The root of the mounted instance, owned by ``circuit``.
        """
        grounded = self.template.ground(self.query_parts, aggregation_statistics)
        node_index_map = self.circuit.mount(grounded.root)
        return node_index_map[grounded.root.index]

    def _mount_instance_with_retained_latents(
        self, assignment: dict[Variable, Any]
    ) -> Unit:
        """
        Ground one exchangeable instance and retain its sampled latents as variables.

        Same grounding as :meth:`_mount_instance`, but each variable in
        ``undetermined_latents`` is additionally mounted as a point-valued sibling leaf
        at its sampled value, instead of leaving it to be marginalized away. Distinct
        sampled values produce disjoint singleton supports by construction, so the
        resulting mixture stays support-deterministic on the retained latents.

        :param assignment: The sampled values of ``undetermined_latents`` for this
            instance.
        :return: The root of a product uniting the mounted instance with a point leaf
            per retained latent, owned by ``circuit``.
        """
        instance_root = self._mount_instance(
            {**self.determined_statistics, **assignment}
        )
        wrapper = ProductUnit(probabilistic_circuit=self.circuit)
        wrapper.add_subcircuit(instance_root)
        for variable in self.undetermined_latents:
            wrapper.add_subcircuit(
                leaf(make_dirac(variable, assignment[variable]), self.circuit)
            )
        return wrapper

    def _sample_undetermined_latents(self) -> list[dict[Variable, Any]]:
        """
        Draw the distinct values of the undetermined latents to integrate over.

        Samples ``monte_carlo_sample_count`` joint assignments of the undetermined
        latents from the conditioned class circuit and deduplicates them, so that each
        distinct value is grounded only once.

        :return: One value assignment per distinct sampled point.
        :raises InvalidMonteCarloSampleCountError: If the sample count is not positive.
        :raises UndeterminedLatentsNotModeledError: If the conditioned class circuit
            does not model the undetermined latents and thus cannot be sampled from.
        """
        if self.monte_carlo_sample_count < 1:
            raise InvalidMonteCarloSampleCountError(self.monte_carlo_sample_count)
        proposal = self.circuit.marginal(self.undetermined_latents)
        if proposal is None:
            raise UndeterminedLatentsNotModeledError(list(self.undetermined_latents))
        samples = proposal.sample(self.monte_carlo_sample_count)
        index_of_variable = proposal.variable_to_index_map
        unique_rows = {tuple(row) for row in samples.tolist()}
        return [
            {
                variable: row[index_of_variable[variable]]
                for variable in self.undetermined_latents
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
    def _node_local_branch_log_probabilities(
        product_node: ProductUnit,
        undetermined_latents: SortedSet[Variable],
        branch_regions: list,
    ) -> list[float]:
        """
        Log-probability of each partition branch's region, local to a mounting product
        node.

        Different product nodes can correlate ``undetermined_latents`` with the
        variables that distinguish them, so each node's weights over the same global
        partition must be computed from its own local marginal, mirroring
        :meth:`_node_local_latent_log_likelihoods`'s per-node handling for the Monte-
        Carlo mixture.

        :param product_node: The mounting product node.
        :param undetermined_latents: The latent variables the partition covers.
        :param branch_regions: Each partition branch's own support region, in the same
            order as the branches being weighted.
        :return: One log-probability per region, in input order.
        """
        subcircuit = ProbabilisticCircuit()
        subcircuit.mount(product_node)
        subcircuit.marginal_in_place(undetermined_latents)
        with np.errstate(divide="ignore"):
            return [
                float(np.log(subcircuit.probability(region)))
                for region in branch_regions
            ]

    @staticmethod
    def _undetermined_latents_partition_disjointly(
        proposal: ProbabilisticCircuit,
    ) -> bool:
        """
        Check whether ``proposal`` is a genuine, pairwise-disjoint partition over the
        undetermined latents: a mixture of at least two branches, no two of which
        overlap.

        A single, undifferentiated branch fails this precondition rather than
        trivially passing it: grounding would still retain the latents as a real
        distribution, but every exchangeable instance would be grounded from the same
        representative point regardless of which latent value the region actually
        corresponds to, silently discarding any correlation between the latents and
        the rest of the circuit.

        :param proposal: ``circuit`` marginalized down to exactly the undetermined
            latents.
        :return:``True`` only if ``proposal``'s root is a mixture of at least two
            branches, every pair of which has non-overlapping support.
        """
        root = proposal.root
        if not isinstance(root, SumUnit) or len(root.subcircuits) < 2:
            return False
        _ = proposal.support
        branch_supports = [child.result_of_current_query for child in root.subcircuits]
        return all(
            left.intersection_with(right).is_empty()
            for left, right in itertools.combinations(branch_supports, 2)
        )

    @staticmethod
    def _representative_value(
        latent_branch: Unit, undetermined_latents: SortedSet[Variable]
    ) -> dict[Variable, Any]:
        """
        Extract one concrete point per undetermined latent from a partition branch.

        Uses each leaf's own mode, collapsed to a single point of that mode region.
        Any point within the branch's support would do for grounding purposes here,
        since the branch's actual probability mass is retained separately by mounting
        the branch itself, not narrowed by this choice. A single point is required
        because conditioning a leaf's distribution on a whole region -- rather than one
        point -- is not supported by :meth:`~probabilistic_model.distributions.distributions.ContinuousDistribution.log_conditional`,
        which every leaf here resolves to (including :class:`IntegerDistribution`,
        through its continuous base).

        :param latent_branch: One branch of ``undetermined_latents``' exact partition.
        :param undetermined_latents: The latent variables to extract a value for.
        :return: One conditioning point per variable in ``undetermined_latents``.
        """
        values = {}
        for leaf_node in latent_branch.leaves:
            if leaf_node.variable not in undetermined_latents:
                continue
            mode, _ = leaf_node.distribution.univariate_log_mode()
            values[leaf_node.variable] = (
                mode.simple_sets[0].lower
                if isinstance(mode, Interval)
                else next(iter(mode))
            )
        return values

    def _attach_mixture_to_node(
        self,
        product_node: ProductUnit,
        instance_roots: list[Unit],
        log_weights: list[float],
    ) -> None:
        """
        Attach a normalized sum unit over exchangeable instances to one node.

        Instances whose node-local likelihood is zero are skipped. The instances are
        already mounted in ``circuit`` and shared across all mounting nodes; only the
        weighted sum-unit edges differ per node.

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
        sum_unit = SumUnit(probabilistic_circuit=self.circuit)
        product_node.add_subcircuit(sum_unit)
        for instance_root, log_weight in weighted_instances:
            sum_unit.add_subcircuit(instance_root, log_weight)
        sum_unit.normalize()


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

    class_circuit_builder: Optional[
        Callable[[pd.DataFrame, list[AnnotatedVariable]], ProbabilisticCircuit]
    ] = None
    """
    Builds the class-level circuit from the class dataframe and its inferred variables,
    in place of ``fit``'s plain, unconstrained ``JointProbabilityTree`` fit.

    Set this before calling ``fit`` when a specific circuit shape is needed -- for
    instance one that is support-deterministic over a chosen variable -- rather than
    ``fit`` knowing about that requirement itself. Leave ``None`` for the plain,
    unconstrained fit.
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
        level features (or runs :attr:`class_circuit_builder` instead, if set), and then
        recursively fits one ``ExchangeableDistributionTemplate`` per exchangeable part
        discovered in the schema.

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
        if self.class_circuit_builder is None:
            self.class_probabilistic_circuit = JointProbabilityTree(
                annotated_variables=variables
            ).fit(class_dataframe)
        else:
            self.class_probabilistic_circuit = self.class_circuit_builder(
                class_dataframe, variables
            )
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
        :return: The conditioned circuit and the product nodes that will be extended
            with the grounded exchangeable distribution.
        """
        if aggregation_statistics:
            conditioning_result, _ = circuit.log_conditional_in_place(
                aggregation_statistics
            )
            if conditioning_result is None:
                circuit = self.class_probabilistic_circuit.__deepcopy__()
        if len(circuit.nodes()) == 0:
            raise ClassCircuitGroundingFailedError(self.class_)
        product_nodes_to_extend = find_lowest_product_nodes_that_model_variables(
            circuit, SortedSet(latent_variables)
        )
        return circuit, product_nodes_to_extend

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
            circuit = self._ground_exchangeable_part(
                circuit,
                exchangeable_part_name,
                template,
                query,
                instance,
                grounding_mode,
            )
        return circuit

    def _ground_exchangeable_part(
        self,
        circuit: ProbabilisticCircuit,
        exchangeable_part_name: str,
        template: ExchangeableDistributionTemplate,
        query: Match,
        instance: Any,
        grounding_mode: GroundingMode,
    ) -> ProbabilisticCircuit:
        """
        Ground one exchangeable part and attach it to the class circuit.

        Aggregation statistics determinable from the query condition the class circuit
        directly. Undetermined statistics are retained as variables, represented either
        by Monte-Carlo sampling (:attr:`GroundingMode.SAMPLED`) or by enumerating the
        fitted circuit's own exact partition over them (:attr:`GroundingMode.EXACT`,
        falling back to :attr:`GroundingMode.SAMPLED` if that partition is not
        disjoint).

        :param circuit: The current working copy of the class circuit.
        :param exchangeable_part_name: Field name of the exchangeable relation.
        :param template: The fitted template for this relation.
        :param query: The grounding query.
        :param instance: The concrete instance constructed from the query.
        :param grounding_mode: How to treat aggregation latents the query leaves
            undetermined. See :class:`GroundingMode`.
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

        grounder = ExchangeablePartGrounder(
            circuit=circuit,
            product_nodes_to_extend=product_nodes_to_extend,
            template=template,
            query_parts=query_parts,
            determined_statistics=determined_statistics,
            undetermined_latents=undetermined_latents,
            monte_carlo_sample_count=self.monte_carlo_sample_count,
        )

        if not undetermined_latents:
            grounder.attach_single_instance()
            return circuit

        if grounding_mode is GroundingMode.EXACT:
            return self._ground_exact(grounder, circuit)
        return self._ground_monte_carlo(grounder, circuit)

    def _ground_exact(
        self, grounder: ExchangeablePartGrounder, circuit: ProbabilisticCircuit
    ) -> ProbabilisticCircuit:
        """
        Ground via :attr:`GroundingMode.EXACT`, falling back to Monte-Carlo sampling if
        the undetermined latents' fitted partition is not support-deterministic.

        :param grounder: The grounder holding this exchangeable part's grounding
            context.
        :param circuit: The class circuit being extended; ``grounder`` mutates it in
            place, so it is returned unchanged once grounding succeeds.
        :return: The class circuit extended with the grounded exchangeable part.
        """
        try:
            grounder.attach_exact_partition_mixture()
            return circuit
        except UndeterminedLatentsNotPartitionedError:
            logger.warning(
                "Exact-partition grounding for latents [%s] is not support-"
                "deterministic; falling back to GroundingMode.SAMPLED.",
                ", ".join(variable.name for variable in grounder.undetermined_latents),
            )
            return self._ground_monte_carlo(grounder, circuit)

    def _ground_monte_carlo(
        self, grounder: ExchangeablePartGrounder, circuit: ProbabilisticCircuit
    ) -> ProbabilisticCircuit:
        """
        Ground via :attr:`GroundingMode.SAMPLED`, attaching a Monte-Carlo mixture over
        the undetermined latents.

        :param grounder: The grounder holding this exchangeable part's grounding
            context.
        :param circuit: The class circuit being extended; ``grounder`` mutates it in
            place, so it is returned unchanged.
        :return: The class circuit extended with the grounded exchangeable part.
        """
        grounder.attach_monte_carlo_mixture()
        return circuit
