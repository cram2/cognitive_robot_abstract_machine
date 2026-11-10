"""
Learning a causal graph from the rows a model is fitted on, with the Peter-Clark
algorithm of causal-learn (Zheng et al., 2024), from a flat table or from the tables a
relational circuit is fitted on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum

import numpy as np
import pandas as pd
from causallearn.graph.GraphNode import GraphNode as CausalLearnNode
from causallearn.search.ConstraintBased.PC import pc as peter_clark
from causallearn.utils.PCUtils.BackgroundKnowledge import (
    BackgroundKnowledge as CausalLearnKnowledge,
)
from typing_extensions import Any

from krrood.class_diagrams.causal_relation import (
    AttributeEndpoint,
    CausalEndpoint,
    CausalRelation,
    CausalRelationsOfClass,
    StatisticEndpoint,
)
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.parametrization.feature_extraction.feature_extractor import (
    FeatureExtractor,
)

from probabilistic_model.causal_structure.domain_causal_knowledge import (
    DomainCausalKnowledge,
)
from probabilistic_model.causal_structure.discretization import Discretizer
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    EdgeMark,
    GraphEdge,
    GraphNode,
    NodeLevel,
)
from probabilistic_model.causal_structure.identification import AdjustmentSelector
from probabilistic_model.causal_structure.relational_checks import (
    ExampleToPartCheck,
    PartAttributeAdjacency,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
    RelationalTrainingTables,
)

# %% learning from one table


class ConditionalIndependenceTest(StrEnum):
    """
    The conditional independence tests the Peter-Clark algorithm can run on discretized
    columns, by the name causal-learn gives them.
    """

    G_SQUARED = "gsq"
    """
    The likelihood-ratio G² test.
    """

    CHI_SQUARED = "chisq"
    """
    Pearson's χ² test.
    """


class AdjacencyMatrixMark(IntEnum):
    """
    How causal-learn's adjacency matrix marks the end of an edge at a node: entry ``[i,
    j]`` is the mark the edge between nodes ``i`` and ``j`` shows at ``i``.
    """

    TAIL = -1
    """
    A tail.
    """

    ARROW = 1
    """
    An arrow head.
    """

    @property
    def mark(self) -> EdgeMark:
        """
        :return: The same mark on a :class:`GraphEdge`.
        """
        return EdgeMark[self.name]


class CausalStructureLearner(ABC):
    """
    Learns a causal graph from the rows of one table.
    """

    @abstractmethod
    def learn(
        self,
        rows: pd.DataFrame,
        nodes: list[GraphNode],
        domain_knowledge: DomainCausalKnowledge,
    ) -> CausalGraph:
        """
        :param rows: The table, one column per node, named by the node.
        :param nodes: The nodes to learn the graph over.
        :param domain_knowledge: What is known about the structure in advance.
        :return: The learned graph over the nodes.
        """


@dataclass
class PeterClarkStructureLearner(CausalStructureLearner):
    """
    The Peter-Clark algorithm in its order-independent form, on the table's columns
    discretized into a few levels each.
    """

    significance_level: float = 0.05
    """
    The level below which a test's p-value rejects independence.
    """

    independence_test: ConditionalIndependenceTest = (
        ConditionalIndependenceTest.G_SQUARED
    )
    """
    The conditional independence test.
    """

    discretizer: Discretizer = field(default_factory=Discretizer)
    """
    How the columns are turned into levels before testing.
    """

    def learn(
        self,
        rows: pd.DataFrame,
        nodes: list[GraphNode],
        domain_knowledge: DomainCausalKnowledge,
    ) -> CausalGraph:
        """
        Deterministic copies and columns that never vary are left out of the tests; a
        copy is restored as a child of its source, and a constant column stays without
        edges.
        """
        node_of = {node.name: node for node in nodes}
        domain_knowledge.check_names(list(node_of))
        copies = {copy.copy for copy in domain_knowledge.deterministic_copies}
        tested = [
            node
            for node in nodes
            if node.name not in copies and rows[node.name].nunique() > 1
        ]
        tested_names = [node.name for node in tested]
        codes = self.discretizer.fit_transform(rows[tested_names])
        graph = peter_clark(
            codes.to_numpy(dtype=np.int64),
            self.significance_level,
            self.independence_test,
            stable=True,
            show_progress=False,
            node_names=tested_names,
            background_knowledge=self.to_causal_learn_knowledge(
                domain_knowledge.restricted_to(tested_names)
            ),
        ).G.graph
        return CausalGraph.from_edges(
            nodes,
            self.read_adjacency_matrix(graph, tested)
            + [
                GraphEdge.directed_edge(node_of[copy.source], node_of[copy.copy])
                for copy in domain_knowledge.deterministic_copies
            ],
        )

    @staticmethod
    def to_causal_learn_knowledge(
        domain_knowledge: DomainCausalKnowledge,
    ) -> CausalLearnKnowledge:
        """
        :param domain_knowledge: Domain knowledge about the tested variables.
        :return: The same knowledge as causal-learn takes it.
        """
        translated = CausalLearnKnowledge()
        for tier, names in enumerate(domain_knowledge.tiers):
            for name in names:
                translated.add_node_to_tier(CausalLearnNode(name), tier)
        for edge in domain_knowledge.required_edges:
            translated.add_required_by_node(
                CausalLearnNode(edge.parent), CausalLearnNode(edge.child)
            )
        for edge in domain_knowledge.forbidden_edges:
            translated.add_forbidden_by_node(
                CausalLearnNode(edge.parent), CausalLearnNode(edge.child)
            )
        return translated

    @staticmethod
    def read_adjacency_matrix(
        graph: np.ndarray, nodes: list[GraphNode]
    ) -> list[GraphEdge]:
        """
        :param graph: causal-learn's adjacency matrix over the nodes.
        :param nodes: Its nodes, in its order.
        :return: Its edges.
        """
        return [
            GraphEdge(
                first=nodes[first],
                second=nodes[second],
                mark_at_first=AdjacencyMatrixMark(graph[first, second]).mark,
                mark_at_second=AdjacencyMatrixMark(graph[second, first]).mark,
            )
            for first, second in np.argwhere(np.triu(graph != 0, k=1))
        ]


# %% what is learned


@dataclass
class LearnedCausalStructure:
    """
    The causal structure learned for one class, ready to choose the variables a causal
    query about the class adjusts for. Its edges are the causal relations of the class
    in a class diagram, read as they are when asked.
    """

    relations: CausalRelationsOfClass
    """
    The causal relations of the class.
    """

    nodes: list[GraphNode]
    """
    A node for each variable of the class's model.
    """

    training_rows: pd.DataFrame
    """
    The model's training rows, one column per example-level node, named as the graph's
    nodes are, which rank the valid adjustment sets by their support.
    """

    @property
    def selector(self) -> AdjustmentSelector:
        """
        :return: What chooses an adjustment set from the class's causal relations.
        """
        return AdjustmentSelector(self.relations.causal_graph(self.nodes))


# %% learning from a relational circuit's tables


@dataclass
class RelationalCausalStructureLearner:
    """
    Learns a causal graph over a relational circuit's variables: over the instances'
    attributes and aggregation statistics from the class table, and over each
    exchangeable part's attributes, with the statistics, from that part's table.

    A part attribute is one template node for every part of its collection, named
    ``Class.collection.attribute``. The circuit lets an instance's attributes reach a
    part's attributes only through the aggregation statistics, so without an
    :attr:`example_to_part_check` no edge joins an instance attribute and a part
    attribute.

    The learned graph is kept in a class diagram, as causal relations of the circuit's
    class.
    """

    class_diagram: ClassDiagram
    """
    The diagram that keeps the learned relations, holding every class an attribute of
    the circuit's variables belongs to.
    """

    learner: CausalStructureLearner = field(default_factory=PeterClarkStructureLearner)
    """
    What learns the graph of each table.
    """

    example_to_part_check: ExampleToPartCheck | None = None
    """
    What finds the instance attributes that depend on a part attribute even given the
    aggregation statistics and any one other attribute the dependence could run
    through. Each such pair is joined by an edge from the instance attribute to the part
    attribute, as the circuit draws an instance before its parts, unless the domain
    causal knowledge's tiers place the part attribute first.
    """

    def learn(
        self,
        circuit: RelationalProbabilisticCircuit,
        instances: list[Any],
        domain_knowledge: DomainCausalKnowledge | None = None,
    ) -> LearnedCausalStructure:
        """
        :param circuit: A fitted relational circuit.
        :param instances: The instances it was fitted on.
        :param domain_knowledge: What is known about the structure in advance, naming part
            attributes by their template node.
        :return: The structure learned from the circuit's training tables.
        """
        domain_knowledge = domain_knowledge or DomainCausalKnowledge()
        tables = circuit.training_tables(instances)
        statistic_names = {
            variable.name
            for template in circuit.exchangeable_distribution_templates.values()
            for variable in template.latent_variables
        }
        example_nodes = [
            GraphNode(
                variable=variable,
                level=(
                    NodeLevel.AGGREGATION_STATISTIC
                    if variable.name in statistic_names
                    else NodeLevel.EXAMPLE_ATTRIBUTE
                ),
            )
            for variable in circuit.class_probabilistic_circuit.variables
        ]
        example_names = [node.name for node in example_nodes]
        example_graph = self.learner.learn(
            tables.example_rows,
            example_nodes,
            domain_knowledge.restricted_to(example_names),
        )
        nodes, edges = list(example_nodes), example_graph.edges()
        for part, table in tables.part_tables.items():
            part_node_of_column = self.part_node_of_column(
                part, self.create_part_nodes(circuit, part)
            )
            part_edges = self.learn_part_edges(
                table.rows, part_node_of_column, example_nodes, domain_knowledge
            )
            nodes += part_node_of_column.values()
            edges += part_edges + self.learn_example_to_part_edges(
                part,
                tables,
                part_node_of_column,
                part_edges,
                example_nodes,
                domain_knowledge,
            )
        return LearnedCausalStructure(
            relations=self.record_in_class_diagram(circuit, edges),
            nodes=nodes,
            training_rows=tables.example_rows,
        )

    def record_in_class_diagram(
        self, circuit: RelationalProbabilisticCircuit, edges: list[GraphEdge]
    ) -> CausalRelationsOfClass:
        """
        Replace the causal relations of the circuit's class in :attr:`class_diagram`
        with the learned edges.

        :param circuit: A fitted relational circuit.
        :param edges: The learned edges.
        :return: The causal relations of the class.
        """
        relations = CausalRelationsOfClass(
            self.class_diagram, self.class_diagram.get_wrapped_class(circuit.class_)
        )
        endpoints = self.create_endpoints(circuit.feature_extractor, None)
        for part, template in circuit.exchangeable_distribution_templates.items():
            endpoints += self.create_endpoints(
                template.template_distribution.feature_extractor,
                relations.owner.get_field(part),
            )
        endpoint_of = {endpoint.name: endpoint for endpoint in endpoints}
        relations.replace_relations(
            [
                CausalRelation(
                    source=relations.owner,
                    target=relations.owner,
                    first=endpoint_of[edge.first.name],
                    second=endpoint_of[edge.second.name],
                    mark_at_first=edge.mark_at_first,
                    mark_at_second=edge.mark_at_second,
                )
                for edge in edges
            ]
        )
        return relations

    def create_endpoints(
        self, extractor: FeatureExtractor, collection: WrappedField | None
    ) -> list[CausalEndpoint]:
        """
        :param extractor: The feature extractor of the circuit's class, or of the parts
            in one of its collections.
        :param collection: The field of the class holding those parts, or ``None`` for
            the class itself.
        :return: The variable of the class each feature is.
        """
        prefix = () if collection is None else (collection,)
        return [
            (
                StatisticEndpoint(feature, collection)
                if feature in extractor.aggregation_features
                else AttributeEndpoint(
                    prefix
                    + AttributeEndpoint.along(
                        self.class_diagram, feature._access_path_
                    ).fields
                )
            )
            for feature in extractor.features
        ]

    @staticmethod
    def create_part_nodes(
        circuit: RelationalProbabilisticCircuit, part: str
    ) -> list[GraphNode]:
        """
        :param circuit: A fitted relational circuit.
        :param part: One of its exchangeable parts.
        :return: One template node per attribute of the part.
        """
        template = circuit.exchangeable_distribution_templates[part]
        statistic_names = {variable.name for variable in template.latent_variables}
        return [
            GraphNode(
                variable=replace(
                    variable,
                    name=f"{circuit.class_.__name__}.{part}.{variable.name}",
                ),
                level=NodeLevel.PART_ATTRIBUTE,
                collection=part,
            )
            for variable in template.template_distribution.class_probabilistic_circuit.variables
            if variable.name not in statistic_names
        ]

    @staticmethod
    def part_node_of_column(
        part: str, part_nodes: list[GraphNode]
    ) -> dict[str, GraphNode]:
        """
        :param part: An exchangeable part.
        :param part_nodes: The template nodes of its attributes.
        :return: Each template node by the name the part's training table gives its
            column.
        """
        return {node.name.split(f".{part}.", 1)[1]: node for node in part_nodes}

    def learn_part_edges(
        self,
        rows: pd.DataFrame,
        part_node_of_column: dict[str, GraphNode],
        example_nodes: list[GraphNode],
        domain_knowledge: DomainCausalKnowledge,
    ) -> list[GraphEdge]:
        """
        :param rows: The training table of an exchangeable part.
        :param part_node_of_column: The template nodes of its attributes, by column.
        :param example_nodes: The nodes of the class table.
        :param domain_knowledge: What is known about the structure in advance.
        :return: The edges learned from the part's table that touch a part attribute.
        """
        part_nodes = list(part_node_of_column.values())
        statistic_nodes = [node for node in example_nodes if node.name in rows.columns]
        renamed = rows.rename(
            columns={column: node.name for column, node in part_node_of_column.items()}
        )
        names = [node.name for node in statistic_nodes + part_nodes]
        graph = self.learner.learn(
            renamed, statistic_nodes + part_nodes, domain_knowledge.restricted_to(names)
        )
        return [
            edge
            for edge in graph.edges()
            if edge.first in part_nodes or edge.second in part_nodes
        ]

    def learn_example_to_part_edges(
        self,
        part: str,
        tables: RelationalTrainingTables,
        part_node_of_column: dict[str, GraphNode],
        part_edges: list[GraphEdge],
        example_nodes: list[GraphNode],
        domain_knowledge: DomainCausalKnowledge,
    ) -> list[GraphEdge]:
        """
        :param part: An exchangeable part.
        :param tables: The circuit's training tables.
        :param part_node_of_column: The template nodes of the part's attributes, by
            column.
        :param part_edges: The edges learned from the part's table.
        :param example_nodes: The nodes of the class table.
        :param domain_knowledge: What is known about the structure in advance.
        :return: An edge for every dependence :attr:`example_to_part_check` keeps
            between an instance attribute and an attribute of the part; none without a
            check.
        """
        if self.example_to_part_check is None:
            return []
        example_node_of = {node.name: node for node in example_nodes}
        return [
            domain_knowledge.orient_edge(
                example_node_of[dependence.example_attribute],
                part_node_of_column[dependence.part_attribute],
            )
            for dependence in self.example_to_part_check.dependences_given_another_attribute(
                tables.example_rows,
                part,
                tables.part_tables[part],
                PartAttributeAdjacency.from_part_edges(part_edges, part_node_of_column),
            )
        ]
