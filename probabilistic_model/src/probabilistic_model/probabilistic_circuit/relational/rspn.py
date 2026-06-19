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
from typing_extensions import TYPE_CHECKING, Any, Optional, Type

from krrood.entity_query_language.query.match import AbstractMatchExpression
from krrood.ormatic.data_access_objects.dao import (
    DataAccessObject,
    DataAccessObjectSchema,
    get_dao_schema,
)
from krrood.ormatic.data_access_objects.from_dao import FromDataAccessObjectState
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extraction.aggregations import get_aggregation_class
from krrood.parametrization.feature_extraction.feature_extractor import FeatureExtractor

if TYPE_CHECKING:
    from krrood.entity_query_language.query.match import Match
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    CircuitNotFittedError,
)
from probabilistic_model.probabilistic_circuit.relational.helper import (
    find_lowest_product_nodes_that_model_variables,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
)
from random_events.variable import Variable


def _rename_variables_with_part_prefix(
    circuit: ProbabilisticCircuit,
    prefix: str,
    excluded_variables: list[Variable],
) -> None:
    """
    Rename each variable in the circuit to include ``prefix`` as a namespace.

    Produces names of the form ``"{prefix}.{variable.name}"``.
    Variables listed in ``excluded_variables`` are left unchanged.

    :param circuit: The circuit whose variables are renamed in-place.
    :param prefix: String prefix to prepend to every variable name.
    :param excluded_variables: Variables that should keep their current names.
    """
    variable_renames = {
        variable: type(variable)(f"{prefix}.{variable.name}", domain=variable.domain)
        for variable in circuit.variables
        if variable not in excluded_variables
    }
    circuit.update_variables(variable_renames)


@dataclass
class ExchangeableDistributionTemplate:
    """
    A fitted distribution template for one exchangeable (many-to-many) relation.

    Wraps a ``RelationalProbabilisticCircuit`` that was trained on the child
    objects of the relation together with the parent's aggregation statistics as
    latent context variables.
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

        Conditions the template circuit on ``aggregation_statistics``, marginalizes
        away the latent variables, renames surviving variables with the part's
        prefix, and reindexes the graph for safe mounting.

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
        :param aggregation_statistics: Observed aggregation values shared across all parts.
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

    schema_information: Optional[DataAccessObjectSchema] = field(init=False, default=None)
    """
    The :class:`~krrood.ormatic.data_access_objects.dao.DataAccessObjectSchema` describing
    the DAO class's columns and relationships.
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

        :param feature_extractor: The extractor used to create and preprocess the dataframe.
        :param instances: Training instances to extract features from.
        :param dataframe_from_parent: Pre-built dataframe from a parent fit call, or ``None``.
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
        child_class_prefix: str,
    ) -> pd.DataFrame:
        """
        Build a dataframe combining aggregation statistics with per-child-object attributes.

        Each row corresponds to one child object and contains the parent instance's
        aggregation values followed by all child features (including nested unique-part
        attributes). Column names strip the child class prefix so that, after variable
        renaming, they align with the krrood access-path convention.

        :param exchangeable_part: Field name of the one-to-many relation on each instance.
        :param instances: Training instances from which rows are generated.
        :param aggregation_indices: Positions of aggregation features in the feature vector.
        :param aggregation_names: Column names for the aggregation portion of each row.
        :param child_feature_extractor: Feature extractor built from the child instances.
        :param child_class_prefix: Class name prefix to strip from child feature names
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
        full_names = [f._name_ for f in child_feature_extractor.features]
        short_names = [
            (
                name[len(child_class_prefix) :]
                if name.startswith(child_class_prefix)
                else name
            )
            for name in full_names
        ]
        return pd.DataFrame(columns=aggregation_names + short_names, data=rows)

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

        :param exchangeable_part: Field name of the one-to-many relation on each instance.
        :param instances: Training instances whose children are used to fit the template.
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
        child_class_name = type(
            child_instances[0].from_dao(FromDataAccessObjectState())
        ).__name__
        child_class_prefix = f"{child_class_name}."
        child_dataframe = self._build_child_joint_dataframe(
            exchangeable_part,
            instances,
            aggregation_indices,
            aggregation_names,
            child_feature_extractor,
            child_class_prefix,
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

        Builds a ``FeatureExtractor``, trains a ``JointProbabilityTree`` on the
        class-level features, and then recursively fits one
        ``ExchangeableDistributionTemplate`` per exchangeable part discovered in
        the schema.

        :param instances: Training instances; all must share the same DAO class.
        :param dataframe_from_parent: Pre-built dataframe supplied by a parent
            ``_fit_exchangeable_part`` call.  When provided, feature extraction
            and preprocessing are skipped.
        :return: ``self``, to allow chaining.
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

    def _compute_aggregation_statistics(
        self,
        queryable_object,
        exchangeable_part_name: str,
        template: ExchangeableDistributionTemplate,
    ) -> dict[Variable, Any]:
        """
        Compute aggregation statistics from the query for conditioning.

        Evaluates each aggregation feature function against the query's constructed
        instance and maps the result to its corresponding latent variable.  Values
        outside the training domain are silently skipped; conditioning on them would
        produce an impossible event and collapse the circuit.

        :param queryable_object: The DAO representation of the query's constructed instance.
        :param exchangeable_part_name: Field name identifying which relation's statistics
            are being computed.
        :param template: The ``ExchangeableDistributionTemplate`` whose latent variables
            define which statistics are relevant.
        :return: A mapping from latent variables to their observed values in the query.
        """
        latent_variable_by_name = {
            variable.name: variable for variable in template.latent_variables
        }
        aggregation_statistics = {}
        for feature_function in self.feature_extractor.exchangeable_features[
            exchangeable_part_name
        ]:
            feature_name = feature_function._name_
            if feature_name not in latent_variable_by_name:
                continue
            domain_object = queryable_object.from_dao()
            aggregation_cls = get_aggregation_class(type(domain_object))
            if aggregation_cls is None:
                continue
            aggregation_instance = aggregation_cls(instance=domain_object)
            value = feature_function.apply_mapping_on_external_root(
                aggregation_instance
            )
            latent_variable = latent_variable_by_name[feature_name]
            # ``make_value`` is the random_events API that validates domain membership;
            # it signals an out-of-domain value by raising. There is no non-throwing
            # membership predicate for a raw value against a symbolic domain, so this
            # boundary adapts that exception into a skip.
            try:
                latent_variable.make_value(value)
                aggregation_statistics[latent_variable] = value
            except (ValueError, TypeError):
                pass
        return aggregation_statistics

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
        :return: The conditioned circuit and the surviving product
            nodes that will be extended with the grounded exchangeable distribution.
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
        """Ground the relational circuit for a specific query.

        Starting from a deep copy of ``class_probabilistic_circuit``, each
        exchangeable part's template is grounded for the objects specified in the
        query and attached to the conditioning product nodes of the class circuit.

        :param query: An underspecified, resolved query instance whose structure
            determines which parts are grounded and how many child objects each
            exchangeable relation contains.
        :return: A concrete ``ProbabilisticCircuit`` over all variables implied
            by the query.
        :raises CircuitNotFittedError: If ``ground`` is called before ``fit``.
        """
        if self.class_probabilistic_circuit is None:
            raise CircuitNotFittedError(self.class_)
        circuit = self.class_probabilistic_circuit.__deepcopy__()
        instance = query.construct_instance()
        queryable_object = to_dao(instance)
        for (
            exchangeable_part_name,
            template,
        ) in self.exchangeable_distribution_templates.items():
            aggregation_statistics = self._compute_aggregation_statistics(
                queryable_object, exchangeable_part_name, template
            )
            circuit, product_nodes_to_extend = self._condition_class_circuit(
                circuit, aggregation_statistics, template.latent_variables
            )
            grounded_exchangeable_circuit = template.ground(
                query.kwargs[exchangeable_part_name], aggregation_statistics
            )
            exchangeable_root = grounded_exchangeable_circuit.root
            node_index_map = circuit.mount(exchangeable_root)
            for product_node in product_nodes_to_extend:
                product_node.add_subcircuit(node_index_map[exchangeable_root.index])
        return circuit
