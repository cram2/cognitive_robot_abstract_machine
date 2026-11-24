"""
Learning a causal graph from the rows a model is fitted on, with the Peter-Clark
algorithm, greedy equivalence search or fast causal inference of causal-learn
:cite:p:`zheng2024causal`, from a flat table or from the tables a relational circuit is
fitted on.
"""

from __future__ import annotations

import contextlib
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import IntEnum, StrEnum

import numpy as np
import numpy.typing as npt
import pandas as pd
from causallearn.graph.GraphClass import CausalGraph as CausalLearnGraph
from causallearn.graph.GraphNode import GraphNode as CausalLearnNode
from causallearn.search.ConstraintBased.FCI import fci as fast_causal_inference
from causallearn.search.ConstraintBased.PC import pc as peter_clark
from causallearn.search.ScoreBased.GES import ges as greedy_equivalence_search
from causallearn.utils.PCUtils.BackgroundKnowledge import (
    BackgroundKnowledge as CausalLearnKnowledge,
)
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import (
    orient_by_background_knowledge,
)
from causallearn.utils.PCUtils.Meek import meek
from typing_extensions import Any, Generic

from krrood.parametrization.causal_relation import (
    AttributeEndpoint,
    CausalEndpoint,
    CausalRelation,
    CausalRelationsOfClass,
    StatisticEndpoint,
)
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.patterns.classproperty import classproperty
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.parametrization.feature_extraction.feature_extractor import (
    FeatureExtractor,
)

from probabilistic_model.causal_structure.ancestral_graph import AncestralGraph
from probabilistic_model.causal_structure.domain_causal_knowledge import (
    DomainCausalKnowledge,
)
from probabilistic_model.causal_structure.discretization import Discretizer
from probabilistic_model.causal_structure.exceptions import (
    QuerySpansSeveralParts,
    TooFewVariablesToScore,
    UnsupportedKnowledgeUnderLatentConfounding,
)
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    EdgeMark,
    EdgeMarkedGraph,
    GraphEdge,
    GraphNode,
    MarkedGraphType,
    NodeLevel,
)
from probabilistic_model.causal_structure.identification import AdjustmentSelector
from probabilistic_model.causal_structure.relational_checks import (
    ExampleToPartCheck,
    PartAttributeAdjacency,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    PartTrainingTable,
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


class GraphScore(StrEnum):
    """
    The scores greedy equivalence search can rate a graph of discretized columns by, by
    the name causal-learn gives them.
    """

    BAYESIAN_DIRICHLET_EQUIVALENT_UNIFORM = "local_score_BDeu"
    """
    The Bayesian Dirichlet equivalent uniform score :cite:p:`heckerman1995learning`.
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

    CIRCLE = 2
    """
    A circle.
    """

    @property
    def mark(self) -> EdgeMark:
        """
        :return: The same mark on a :class:`GraphEdge`.
        """
        return EdgeMark[self.name]


class KnowledgeKind(StrEnum):
    """
    The kinds of knowledge about a structure that are given before it is learned.
    """

    TIERS = "tiers"
    """
    Tiers that order the variables, none caused by a later one.
    """

    FORBIDDEN_EDGES = "forbidden edges"
    """
    Pairs of variables of which the first does not directly cause the second.
    """

    EXAMPLE_TO_PART_CHECK = "an example-to-part check"
    """
    A check that joins an instance's attribute to its parts' attributes.
    """

    @property
    def correction_under_latent_confounding(self) -> str:
        """
        :return: What to do instead when a learner that allows hidden confounders is
            given this knowledge.
        """
        match self:
            case KnowledgeKind.EXAMPLE_TO_PART_CHECK:
                return (
                    "Leave the check out: the learner learns the edges between an "
                    "instance and its parts from a table of instances and one part each."
                )
            case _:
                return (
                    "Give only required edges, or learn with the Peter-Clark algorithm, "
                    "which assumes no hidden confounders."
                )


class CausalStructureLearner(ABC):
    """
    Learns a causal graph from the rows of one table.
    """

    @classproperty
    def graph_class(cls) -> type[EdgeMarkedGraph]:
        """
        :return: The kind of graph the learner learns.
        """
        return CausalGraph

    @abstractmethod
    def learn(
        self,
        rows: pd.DataFrame,
        nodes: list[GraphNode],
        domain_knowledge: DomainCausalKnowledge,
    ) -> EdgeMarkedGraph:
        """
        :param rows: The table, one column per node, named by the node.
        :param nodes: The nodes to learn the graph over.
        :param domain_knowledge: What is known about the structure in advance.
        :return: The learned graph over the nodes.
        """


@dataclass
class DiscretizedTableStructureLearner(
    CausalStructureLearner, Generic[MarkedGraphType], SubClassSafeGeneric, ABC
):
    """
    Learns a causal graph from the table's columns discretized into a few levels each.

    A subclass binds the kind of graph it learns.
    """

    @classproperty
    def graph_class(cls) -> type[EdgeMarkedGraph]:
        """
        :return: The kind of graph the learner learns, as the subclass binds it.
        """
        return cls.get_type_of_generic_parameter(parameter=MarkedGraphType)

    discretizer: Discretizer = field(default_factory=Discretizer)
    """
    How the columns are turned into levels before learning.
    """

    def learn(
        self,
        rows: pd.DataFrame,
        nodes: list[GraphNode],
        domain_knowledge: DomainCausalKnowledge,
    ) -> MarkedGraphType:
        """
        Deterministic copies and columns that never vary are left out of the learning; a
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
        levels = self.discretizer.fit_transform(rows[tested_names])
        return self.graph_class.from_edges(
            nodes=nodes,
            edges=self.learn_edges(
                levels=levels.to_numpy(dtype=np.int64),
                nodes=tested,
                domain_knowledge=self.to_causal_learn_knowledge(
                    domain_knowledge=domain_knowledge.restricted_to(tested_names)
                ),
            )
            + [
                GraphEdge.directed_edge(node_of[copy.source], node_of[copy.copy])
                for copy in domain_knowledge.deterministic_copies
            ],
        )

    @abstractmethod
    def learn_edges(
        self,
        *,
        levels: npt.NDArray[np.integer],
        nodes: list[GraphNode],
        domain_knowledge: CausalLearnKnowledge,
    ) -> list[GraphEdge]:
        """
        :param levels: The discretized columns, one per node, in the order of the nodes.
        :param nodes: The nodes to learn the edges between.
        :param domain_knowledge: What is known about them in advance.
        :return: The learned edges between them.
        """

    @staticmethod
    def to_causal_learn_knowledge(
        *,
        domain_knowledge: DomainCausalKnowledge,
    ) -> CausalLearnKnowledge:
        """
        :param domain_knowledge: Domain knowledge about the learned variables.
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
        graph: npt.NDArray[np.integer], nodes: list[GraphNode]
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


@dataclass
class PeterClarkStructureLearner(DiscretizedTableStructureLearner[CausalGraph]):
    """
    The Peter-Clark algorithm in its order-independent form.
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

    def learn_edges(
        self,
        *,
        levels: npt.NDArray[np.integer],
        nodes: list[GraphNode],
        domain_knowledge: CausalLearnKnowledge,
    ) -> list[GraphEdge]:
        graph = peter_clark(
            levels,
            self.significance_level,
            self.independence_test,
            stable=True,
            show_progress=False,
            node_names=[node.name for node in nodes],
            background_knowledge=domain_knowledge,
        ).G.graph
        return self.read_adjacency_matrix(graph, nodes)


@dataclass
class GreedyEquivalenceSearchStructureLearner(
    DiscretizedTableStructureLearner[CausalGraph]
):
    """
    Greedy equivalence search :cite:p:`chickering2002optimal`, which moves between
    equivalence classes while that raises their score.

    .. note::
        The domain causal knowledge does not restrict the search. It orients the edges
        the search leaves undirected, followed by the orientations they imply
        :cite:p:`meek1995causal`.
    """

    score: GraphScore = GraphScore.BAYESIAN_DIRICHLET_EQUIVALENT_UNIFORM
    """
    The score of a graph.
    """

    @classproperty
    def minimum_variable_count(cls) -> int:
        """
        :return: The fewest variables causal-learn's score is defined for.
        """
        return 3

    def learn_edges(
        self,
        *,
        levels: npt.NDArray[np.integer],
        nodes: list[GraphNode],
        domain_knowledge: CausalLearnKnowledge,
    ) -> list[GraphEdge]:
        """
        :raises TooFewVariablesToScore: If there are fewer than
            :attr:`minimum_variable_count` nodes.
        """
        if len(nodes) < self.minimum_variable_count:
            raise TooFewVariablesToScore(len(nodes), self.minimum_variable_count)
        names = [node.name for node in nodes]
        graph = CausalLearnGraph(len(names), names)
        graph.G = greedy_equivalence_search(levels, self.score, node_names=names)["G"]
        orient_by_background_knowledge(graph, domain_knowledge)
        return self.read_adjacency_matrix(meek(graph, domain_knowledge).G.graph, nodes)


@dataclass(kw_only=True)
class FastCausalInferenceStructureLearner(
    DiscretizedTableStructureLearner[AncestralGraph]
):
    """
    Fast causal inference :cite:p:`spirtes2000causation`, which learns a partial
    ancestral graph and so allows hidden variables that cause two or more of the table's
    columns.

    .. note::
        Tiers and forbidden edges are rejected: causal-learn turns each into a tail as
        well as an arrow, which claims a cause the data does not show once a hidden
        variable may cause both ends. Required edges are kept.
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

    maximum_conditioning_set_size: int | None = None
    """
    How many variables a test may condition on; ``None`` for no limit.
    """

    maximum_discriminating_path_length: int | None = None
    """
    How long a discriminating path may be that orients an edge; ``None`` for no limit.
    """

    @classproperty
    def no_limit(cls) -> int:
        """
        :return: The value causal-learn reads as no limit on a size or a length.
        """
        return -1

    def learn_edges(
        self,
        *,
        levels: npt.NDArray[np.integer],
        nodes: list[GraphNode],
        domain_knowledge: CausalLearnKnowledge,
    ) -> list[GraphEdge]:
        with contextlib.redirect_stdout(io.StringIO()):
            graph, _ = fast_causal_inference(
                levels,
                independence_test_method=self.independence_test,
                alpha=self.significance_level,
                depth=self.limit_of(maximum=self.maximum_conditioning_set_size),
                max_path_length=self.limit_of(
                    maximum=self.maximum_discriminating_path_length
                ),
                show_progress=False,
                node_names=[node.name for node in nodes],
                background_knowledge=domain_knowledge,
            )
        return self.read_adjacency_matrix(graph.graph, nodes)

    def limit_of(self, *, maximum: int | None) -> int:
        """
        :param maximum: A limit, or ``None`` for none.
        :return: The limit as causal-learn takes it.
        """
        return self.no_limit if maximum is None else maximum

    @staticmethod
    def to_causal_learn_knowledge(
        *,
        domain_knowledge: DomainCausalKnowledge,
    ) -> CausalLearnKnowledge:
        """
        :raises UnsupportedKnowledgeUnderLatentConfounding: If the knowledge has tiers
            or forbidden edges.
        """
        if any(domain_knowledge.tiers):
            raise UnsupportedKnowledgeUnderLatentConfounding(kind=KnowledgeKind.TIERS)
        if domain_knowledge.forbidden_edges:
            raise UnsupportedKnowledgeUnderLatentConfounding(
                kind=KnowledgeKind.FORBIDDEN_EDGES
            )
        return DiscretizedTableStructureLearner.to_causal_learn_knowledge(
            domain_knowledge=domain_knowledge
        )


# %% what is learned


@dataclass(frozen=True, kw_only=True)
class PartOfCollection:
    """
    One part of a collection, as a query's variable names it.
    """

    collection: str
    """
    The name of the collection.
    """

    index: int
    """
    The position of the part in it.
    """

    def __str__(self) -> str:
        """
        :return: The part as ``collection[index]``.
        """
        return f"{self.collection}[{self.index}]"


@dataclass
class LearnedCausalStructure:
    """
    The causal structure learned for one class, ready to choose the variables a causal
    query about the class adjusts for.

    Its edges are the causal relations of the class in a class diagram, read as they are
    when asked.
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

    graph_class: type[EdgeMarkedGraph] = CausalGraph
    """
    The kind of graph the causal relations are read as; one not closed under union is
    learned once per scope, for the class table and for each collection.
    """

    @property
    def selector(self) -> AdjustmentSelector:
        """
        :return: What chooses an adjustment set from the class's causal relations, of
            the class table's graph for a graph learned once per scope.
        """
        return self.selector_of(scope=None)

    def selector_covering(self, *, variable_names: list[str]) -> AdjustmentSelector:
        """
        :param variable_names: The names of a query's cause, effect and declared
            confounders.
        :return: What chooses an adjustment set among them: :attr:`selector`, or for a
            graph learned once per scope, that of the collection whose part they reach.
        :raises QuerySpansSeveralParts: If, for a graph learned once per scope, they
            reach two collections or two parts of one.
        """
        if self.graph_class.is_closed_under_union:
            return self.selector
        return self.selector_of(scope=self.scope_of(variable_names=variable_names))

    def selector_of(self, *, scope: str | None) -> AdjustmentSelector:
        """
        :param scope: The collection whose table a graph was learned from, or ``None``
            for the class table.
        :return: What chooses an adjustment set from the graph of that scope, over its
            nodes: every node for a graph that is closed under union, else the class
            table's nodes and those of the collection's parts.
        """
        nodes = (
            self.nodes
            if self.graph_class.is_closed_under_union
            else [node for node in self.nodes if node.collection in (None, scope)]
        )
        return AdjustmentSelector(
            self.relations.causal_graph(
                nodes=nodes, graph_class=self.graph_class, scope=scope
            )
        )

    def scope_of(self, *, variable_names: list[str]) -> str | None:
        """
        :param variable_names: The names of a query's variables.
        :return: The collection whose one part the variables reach; ``None`` if they
            reach no part.
        :raises QuerySpansSeveralParts: If they reach two collections or two parts of
            one.
        """
        parts = {
            PartOfCollection(collection=node.collection, index=index)
            for name in variable_names
            for node in self.nodes
            if (index := node.part_index_of(name)) is not None
        }
        if not parts:
            return None
        if len(parts) > 1:
            raise QuerySpansSeveralParts(
                variable_names=variable_names,
                parts=sorted(str(part) for part in parts),
            )
        [part] = parts
        return part.collection


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
    aggregation statistics and any one other attribute the dependence could run through.

    Each such pair is joined by an edge from the instance attribute to the part
    attribute, as the circuit draws an instance before its parts, unless the domain
    causal knowledge's tiers place the part attribute first.
    """

    part_draw_seed: int = 0
    """
    The seed of the draw of one part per instance, for a learner that learns one graph
    per scope.
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
        if not self.learner.graph_class.is_closed_under_union:
            return self.learn_per_scope(
                circuit=circuit,
                tables=tables,
                example_nodes=example_nodes,
                domain_knowledge=domain_knowledge,
            )
        example_names = [node.name for node in example_nodes]
        example_graph = self.learner.learn(
            rows=tables.example_rows,
            nodes=example_nodes,
            domain_knowledge=domain_knowledge.restricted_to(example_names),
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
            relations=self.record_in_class_diagram(circuit, {None: edges}),
            nodes=nodes,
            training_rows=tables.example_rows,
        )

    def learn_per_scope(
        self,
        *,
        circuit: RelationalProbabilisticCircuit,
        tables: RelationalTrainingTables,
        example_nodes: list[GraphNode],
        domain_knowledge: DomainCausalKnowledge,
    ) -> LearnedCausalStructure:
        """
        :param circuit: A fitted relational circuit.
        :param tables: Its training tables.
        :param example_nodes: The nodes of the class table.
        :param domain_knowledge: What is known about the structure in advance.
        :return: The structure of one graph learned from the class table, and one per
            collection from a table that pairs every instance with one of its parts,
            drawn at random, as a learner whose graphs are not closed under union
            needs.
        :raises UnsupportedKnowledgeUnderLatentConfounding: If an
            :attr:`example_to_part_check` is given.
        """
        if self.example_to_part_check is not None:
            raise UnsupportedKnowledgeUnderLatentConfounding(
                kind=KnowledgeKind.EXAMPLE_TO_PART_CHECK
            )
        example_names = [node.name for node in example_nodes]
        edges_by_scope = {
            None: self.learner.learn(
                rows=tables.example_rows,
                nodes=example_nodes,
                domain_knowledge=domain_knowledge.restricted_to(example_names),
            ).edges()
        }
        nodes = list(example_nodes)
        for part, table in tables.part_tables.items():
            part_node_of_column = self.part_node_of_column(
                part, self.create_part_nodes(circuit, part)
            )
            part_nodes = list(part_node_of_column.values())
            scope_nodes = example_nodes + part_nodes
            edges_by_scope[part] = self.learner.learn(
                rows=self.collection_rows(
                    example_rows=tables.example_rows,
                    table=table.one_part_per_example(seed=self.part_draw_seed),
                    part_node_of_column=part_node_of_column,
                ),
                nodes=scope_nodes,
                domain_knowledge=domain_knowledge.restricted_to(
                    [node.name for node in scope_nodes]
                ),
            ).edges()
            nodes += part_nodes
        return LearnedCausalStructure(
            relations=self.record_in_class_diagram(circuit, edges_by_scope),
            nodes=nodes,
            training_rows=tables.example_rows,
            graph_class=self.learner.graph_class,
        )

    @staticmethod
    def collection_rows(
        *,
        example_rows: pd.DataFrame,
        table: PartTrainingTable,
        part_node_of_column: dict[str, GraphNode],
    ) -> pd.DataFrame:
        """
        :param example_rows: The class table.
        :param table: A part table with one row per instance.
        :param part_node_of_column: The template nodes of the part's attributes, by
            column.
        :return: Each instance's row of the class table, followed by its part's
            attributes, named by their template nodes.
        """
        parts = table.rows.drop(columns=table.aggregation_names).rename(
            columns={column: node.name for column, node in part_node_of_column.items()}
        )
        return pd.concat(
            [
                example_rows.iloc[table.example_positions].reset_index(drop=True),
                parts.reset_index(drop=True),
            ],
            axis=1,
        )

    def record_in_class_diagram(
        self,
        circuit: RelationalProbabilisticCircuit,
        edges_by_scope: dict[str | None, list[GraphEdge]],
    ) -> CausalRelationsOfClass:
        """
        Replace the causal relations of the circuit's class in :attr:`class_diagram`
        with the learned edges.

        :param circuit: A fitted relational circuit.
        :param edges_by_scope: The learned edges of each scope: ``None`` for the class
            table or for a graph not learned per scope, else the collection's name.
        :return: The causal relations of the class.
        """
        relations = CausalRelationsOfClass(
            class_diagram=self.class_diagram,
            owner=self.class_diagram.get_wrapped_class(clazz=circuit.class_),
        )
        endpoints = self.create_endpoints(circuit.feature_extractor, None)
        for part, template in circuit.exchangeable_distribution_templates.items():
            endpoints += self.create_endpoints(
                template.template_distribution.feature_extractor,
                relations.owner.get_field(name=part),
            )
        endpoint_of = {endpoint.name: endpoint for endpoint in endpoints}
        relations.replace_relations(
            relations=[
                CausalRelation(
                    source=relations.owner,
                    target=relations.owner,
                    first=endpoint_of[edge.first.name],
                    second=endpoint_of[edge.second.name],
                    mark_at_first=edge.mark_at_first,
                    mark_at_second=edge.mark_at_second,
                    scope=scope,
                )
                for scope, edges in edges_by_scope.items()
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
                StatisticEndpoint(statistic=feature, collection=collection)
                if feature in extractor.aggregation_features
                else AttributeEndpoint(
                    fields=prefix
                    + AttributeEndpoint.along(
                        class_diagram=self.class_diagram,
                        access_path=feature._access_path_,
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
            rows=renamed,
            nodes=statistic_nodes + part_nodes,
            domain_knowledge=domain_knowledge.restricted_to(names),
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
