"""
Learning a causal graph from the rows a model is fitted on, with the PC algorithm of
causal-learn (Zheng et al., 2024), from a flat table or from the tables a relational
circuit is fitted on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum

import numpy as np
import pandas as pd
from causallearn.graph.GraphNode import GraphNode as CausalLearnNode
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import (
    BackgroundKnowledge as CausalLearnKnowledge,
)
from random_events.variable import Variable
from typing_extensions import Any

from probabilistic_model.causal_structure.background_knowledge import (
    BackgroundKnowledge,
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
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
    TrainingTableColumn,
)

# %% learning from one table


class IndependenceTest(StrEnum):
    """
    The conditional independence tests the PC algorithm can run on discretized columns,
    by the name causal-learn gives them.
    """

    G_SQUARED = "gsq"
    """
    The likelihood-ratio G² test.
    """

    CHI_SQUARED = "chisq"
    """
    Pearson's χ² test.
    """


class CausalLearnEndpoint(IntEnum):
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


class StructureLearner(ABC):
    """
    Learns a causal graph from the rows of one table.
    """

    @abstractmethod
    def learn(
        self,
        rows: pd.DataFrame,
        nodes: list[GraphNode],
        knowledge: BackgroundKnowledge,
    ) -> CausalGraph:
        """
        :param rows: The table, one column per node, named by the node.
        :param nodes: The nodes to learn the graph over.
        :param knowledge: What is known about the structure in advance.
        :return: The learned graph over the nodes.
        """


@dataclass
class PCStructureLearner(StructureLearner):
    """
    The PC algorithm in its order-independent form, on the table's columns discretized
    into a few levels each.
    """

    significance_level: float = 0.05
    """
    The level below which a test's p-value rejects independence.
    """

    independence_test: IndependenceTest = IndependenceTest.G_SQUARED
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
        knowledge: BackgroundKnowledge,
    ) -> CausalGraph:
        """
        Deterministic copies and columns that never vary are left out of the tests; a
        copy is restored as a child of its source, and a constant column stays without
        edges.
        """
        names = [node.name for node in nodes]
        knowledge.check_names(names)
        copies = {copy.copy for copy in knowledge.deterministic_copies}
        tested = [
            name for name in names if name not in copies and rows[name].nunique() > 1
        ]
        codes = self.discretizer.fit_transform(rows[tested])
        graph = pc(
            codes.to_numpy(dtype=np.int64),
            self.significance_level,
            self.independence_test,
            stable=True,
            show_progress=False,
            node_names=tested,
            background_knowledge=self.causal_learn_knowledge(
                knowledge.restricted_to(tested)
            ),
        ).G.graph
        return CausalGraph.from_edges(
            nodes,
            self.edges_of(graph, tested)
            + [
                GraphEdge.directed(copy.source, copy.copy)
                for copy in knowledge.deterministic_copies
            ],
        )

    @staticmethod
    def causal_learn_knowledge(knowledge: BackgroundKnowledge) -> CausalLearnKnowledge:
        """
        :param knowledge: Knowledge about the tested variables.
        :return: The same knowledge as causal-learn takes it.
        """
        translated = CausalLearnKnowledge()
        for tier, names in enumerate(knowledge.tiers):
            for name in names:
                translated.add_node_to_tier(CausalLearnNode(name), tier)
        for edge in knowledge.required_edges:
            translated.add_required_by_node(
                CausalLearnNode(edge.parent), CausalLearnNode(edge.child)
            )
        for edge in knowledge.forbidden_edges:
            translated.add_forbidden_by_node(
                CausalLearnNode(edge.parent), CausalLearnNode(edge.child)
            )
        return translated

    @staticmethod
    def edges_of(graph: np.ndarray, names: list[str]) -> list[GraphEdge]:
        """
        :param graph: causal-learn's adjacency matrix over the named nodes.
        :param names: The names of its nodes, in its order.
        :return: Its edges.
        """
        return [
            GraphEdge(
                first=names[first],
                second=names[second],
                mark_at_first=CausalLearnEndpoint(graph[first, second]).mark,
                mark_at_second=CausalLearnEndpoint(graph[second, first]).mark,
            )
            for first in range(len(names))
            for second in range(first + 1, len(names))
            if graph[first, second] != 0
        ]


# %% what is learned


@dataclass
class LearnedCausalStructure:
    """
    A causal graph learned from a model's training rows, ready to choose the variables a
    causal query over that model adjusts for.
    """

    selector: AdjustmentSelector
    """
    What chooses an adjustment set from the learned graph.
    """

    training_rows: pd.DataFrame
    """
    The model's training rows, one column per example-level node, named as the graph's
    nodes are, which rank the valid adjustment sets by their support.
    """

    @classmethod
    def from_rows(
        cls,
        rows: pd.DataFrame,
        variables: list[Variable],
        learner: StructureLearner | None = None,
        knowledge: BackgroundKnowledge | None = None,
    ) -> LearnedCausalStructure:
        """
        :param rows: The table a flat model was fitted on, one column per variable.
        :param variables: The model's variables.
        :param learner: What learns the graph; the PC algorithm if not given.
        :param knowledge: What is known about the structure in advance.
        :return: The structure learned from the table.
        """
        learner = learner or PCStructureLearner()
        nodes = [
            GraphNode(variable=variable, level=NodeLevel.EXAMPLE_ATTRIBUTE)
            for variable in variables
        ]
        return cls(
            selector=AdjustmentSelector(
                learner.learn(rows, nodes, knowledge or BackgroundKnowledge())
            ),
            training_rows=rows,
        )


# %% learning from a relational circuit's tables


@dataclass
class RelationalStructureLearner:
    """
    Learns a causal graph over a relational circuit's variables: over the instances'
    attributes and aggregation statistics from the class table, and over each
    exchangeable part's attributes, with the statistics, from that part's table.

    A part attribute is one template node for every part of its collection, named
    ``Class.collection.attribute``. The circuit lets an instance's attributes reach a
    part's attributes only through the aggregation statistics, so no edge joins an
    instance attribute and a part attribute.
    """

    learner: StructureLearner = field(default_factory=PCStructureLearner)
    """
    What learns the graph of each table.
    """

    def learn(
        self,
        circuit: RelationalProbabilisticCircuit,
        instances: list[Any],
        knowledge: BackgroundKnowledge | None = None,
    ) -> LearnedCausalStructure:
        """
        :param circuit: A fitted relational circuit.
        :param instances: The instances it was fitted on.
        :param knowledge: What is known about the structure in advance, naming part
            attributes by their template node.
        :return: The structure learned from the circuit's training tables.
        """
        knowledge = knowledge or BackgroundKnowledge()
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
            knowledge.restricted_to(example_names),
        )
        nodes, edges = list(example_nodes), example_graph.edges()
        for part, rows in tables.part_rows.items():
            part_nodes = self.part_nodes_of(circuit, part)
            nodes += part_nodes
            edges += self.part_edges(part, rows, part_nodes, example_nodes, knowledge)
        return LearnedCausalStructure(
            selector=AdjustmentSelector(CausalGraph.from_edges(nodes, edges)),
            training_rows=tables.example_rows,
        )

    @staticmethod
    def part_nodes_of(
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

    def part_edges(
        self,
        part: str,
        rows: pd.DataFrame,
        part_nodes: list[GraphNode],
        example_nodes: list[GraphNode],
        knowledge: BackgroundKnowledge,
    ) -> list[GraphEdge]:
        """
        :param part: An exchangeable part.
        :param rows: Its training table.
        :param part_nodes: The template nodes of its attributes.
        :param example_nodes: The nodes of the class table.
        :param knowledge: What is known about the structure in advance.
        :return: The edges learned from the part's table that touch a part attribute.
        """
        statistic_nodes = [node for node in example_nodes if node.name in rows.columns]
        renamed = rows.drop(columns=[TrainingTableColumn.EXAMPLE_POSITION]).rename(
            columns={
                node.name.split(f".{part}.", 1)[1]: node.name for node in part_nodes
            }
        )
        names = [node.name for node in statistic_nodes + part_nodes]
        graph = self.learner.learn(
            renamed, statistic_nodes + part_nodes, knowledge.restricted_to(names)
        )
        part_names = {node.name for node in part_nodes}
        return [
            edge
            for edge in graph.edges()
            if edge.first in part_names or edge.second in part_names
        ]
