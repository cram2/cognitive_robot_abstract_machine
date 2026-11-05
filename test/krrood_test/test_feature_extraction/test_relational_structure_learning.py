import itertools
from dataclasses import dataclass

import pandas as pd
import pytest

from krrood.class_diagrams.causal_relation import (
    AttributeEndpoint,
    CausalRelationsOfClass,
    StatisticEndpoint,
)
from krrood.class_diagrams.class_diagram import ClassDiagram
from probabilistic_model.causal_structure.domain_causal_knowledge import (
    DomainCausalKnowledge,
)
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    GraphEdge,
    GraphNode,
    NodeLevel,
)
from probabilistic_model.causal_structure.learning import (
    CausalStructureLearner,
    RelationalCausalStructureLearner,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    RelationalProbabilisticCircuit,
)

from .test_relational_probabilistic_circuit import (  # noqa: F401  # fixtures
    layered_relational_probabilistic_circuit,
    relational_probabilistic_circuit,
    room_query_4,
    scenario,
)
from .test_nested_exchangeable_parts import (  # noqa: F401  # fixtures
    _nested_scenario,
    nested_relational_probabilistic_circuit,
)
from ..dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneRoom,
    SceneWithExchangeableParts,
)

# %% a class diagram and a learner that joins every pair of nodes


@pytest.fixture
def class_diagram() -> ClassDiagram:
    return ClassDiagram(
        classes=[
            SceneWithExchangeableParts,
            SceneRoom,
            SceneObject,
            KRROODPosition,
            KRROODOrientation,
        ]
    )


@dataclass(kw_only=True)
class EveryPairJoinedLearner(CausalStructureLearner):
    """
    Joins every pair of nodes by an undirected edge, whatever the rows say.
    """

    def learn(
        self,
        *, rows: pd.DataFrame,
        nodes: list[GraphNode],
        domain_knowledge: DomainCausalKnowledge,
    ) -> CausalGraph:
        return CausalGraph.from_edges(
            nodes,
            [
                GraphEdge.undirected_edge(first, second)
                for first, second in itertools.combinations(nodes, 2)
            ],
        )


# %% the tables a relational circuit is fitted on


def test_training_tables_match_the_fitted_tables(
    relational_probabilistic_circuit, scenario
):
    tables = relational_probabilistic_circuit.training_tables(list(scenario))
    template = relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]

    assert list(tables.example_rows.columns) == [
        variable.name
        for variable in relational_probabilistic_circuit.class_probabilistic_circuit.variables
    ]
    assert set(tables.part_rows["objects"].columns) == {
        variable.name
        for variable in template.template_distribution.class_probabilistic_circuit.variables
    }


# %% the graph learned from them


@pytest.mark.parametrize(
    "fitted_circuit",
    ["relational_probabilistic_circuit", "layered_relational_probabilistic_circuit"],
)
def test_learned_graph_variables_match_query_variables(
    fitted_circuit, scenario, room_query_4, class_diagram, request
):
    circuit: RelationalProbabilisticCircuit = request.getfixturevalue(fitted_circuit)
    graph = (
        RelationalCausalStructureLearner(class_diagram)
        .learn(circuit, list(scenario))
        .selector.graph
    )
    grounded_names = {
        variable.name for variable in circuit.ground(room_query_4).variables
    }

    part_names = {
        node.variable_of_part(0).name
        for node in graph.nodes()
        if node.level is NodeLevel.PART_ATTRIBUTE
    }
    example_names = {
        node.name
        for node in graph.nodes()
        if node.level is not NodeLevel.PART_ATTRIBUTE
    }

    assert part_names <= grounded_names
    assert example_names <= grounded_names | {
        variable.name for variable in circuit.class_probabilistic_circuit.variables
    }


def test_statistics_are_aggregation_nodes(
    relational_probabilistic_circuit, scenario, class_diagram
):
    graph = (
        RelationalCausalStructureLearner(class_diagram)
        .learn(relational_probabilistic_circuit, list(scenario))
        .selector.graph
    )
    template = relational_probabilistic_circuit.exchangeable_distribution_templates[
        "objects"
    ]

    assert {
        node.name
        for node in graph.nodes()
        if node.level is NodeLevel.AGGREGATION_STATISTIC
    } == {variable.name for variable in template.latent_variables}


# %% the learned graph kept in the class diagram


def test_every_learned_edge_is_a_causal_relation_of_the_class(
    relational_probabilistic_circuit, scenario, class_diagram
):
    graph = (
        RelationalCausalStructureLearner(class_diagram, EveryPairJoinedLearner())
        .learn(relational_probabilistic_circuit, list(scenario))
        .selector.graph
    )
    relations = CausalRelationsOfClass(
        class_diagram=class_diagram, owner=class_diagram.get_wrapped_class(clazz=SceneRoom)
    )

    node_of = {node.name: node for node in graph.nodes()}
    assert {relation.edge_among(node_of=node_of) for relation in relations.relations} == set(
        graph.edges()
    )


def test_relearning_replaces_the_causal_relations_of_the_class(
    relational_probabilistic_circuit, scenario, class_diagram
):
    learner = RelationalCausalStructureLearner(class_diagram, EveryPairJoinedLearner())
    graph = learner.learn(
        relational_probabilistic_circuit, list(scenario)
    ).selector.graph

    learner.learn(relational_probabilistic_circuit, list(scenario))

    assert len(
        CausalRelationsOfClass(
            class_diagram=class_diagram, owner=class_diagram.get_wrapped_class(SceneRoom)
        ).relations
    ) == len(graph.edges())


def test_a_part_attribute_is_reached_through_its_collection(
    relational_probabilistic_circuit, scenario, class_diagram
):
    RelationalCausalStructureLearner(class_diagram, EveryPairJoinedLearner()).learn(
        relational_probabilistic_circuit, list(scenario)
    )
    room = class_diagram.get_wrapped_class(clazz=SceneRoom)

    assert {
        endpoint.fields
        for relation in CausalRelationsOfClass(class_diagram=class_diagram, owner=room).relations
        for endpoint in (relation.first, relation.second)
        if isinstance(endpoint, AttributeEndpoint)
        and endpoint.fields[0] == room.get_field("objects")
    } == {
        (
            room.get_field("objects"),
            class_diagram.get_wrapped_class(SceneObject).get_field("type"),
        )
    }


def test_a_statistic_of_the_parts_is_reached_through_their_collection(
    nested_relational_probabilistic_circuit, class_diagram
):
    RelationalCausalStructureLearner(class_diagram, EveryPairJoinedLearner()).learn(
        nested_relational_probabilistic_circuit, _nested_scenario([2, 3])
    )
    room_extractor = (
        nested_relational_probabilistic_circuit.exchangeable_distribution_templates[
            "rooms"
        ].template_distribution.feature_extractor
    )
    scene = class_diagram.get_wrapped_class(clazz=SceneWithExchangeableParts)

    assert {
        endpoint.name
        for relation in CausalRelationsOfClass(class_diagram=class_diagram, owner=scene).relations
        for endpoint in (relation.first, relation.second)
        if isinstance(endpoint, StatisticEndpoint)
        and endpoint.collection == scene.get_field("rooms")
    } == {
        f"{SceneWithExchangeableParts.__name__}.rooms.{statistic._name_}"
        for statistic in room_extractor.aggregation_features
    }


def test_the_learned_structure_reads_the_relations_the_diagram_holds_now(
    relational_probabilistic_circuit, scenario, class_diagram
):
    structure = RelationalCausalStructureLearner(
        class_diagram, EveryPairJoinedLearner()
    ).learn(relational_probabilistic_circuit, list(scenario))

    structure.relations.replace_relations(relations=[])

    assert structure.selector.graph.edges() == []
