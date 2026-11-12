import itertools
from dataclasses import dataclass, field

import numpy as np
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
from probabilistic_model.causal_structure.relational_checks import (
    ExampleToPartCheck,
    ExampleToPartDependence,
    PartAttributeAdjacency,
)
from probabilistic_model.probabilistic_circuit.relational.relational_probabilistic_circuit import (
    PartTrainingTable,
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
    GraspAttempt,
    PickingRobot,
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
        *,
        rows: pd.DataFrame,
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
    assert set(tables.part_tables["objects"].rows.columns) == {
        variable.name
        for variable in template.template_distribution.class_probabilistic_circuit.variables
    }


def test_training_tables_place_each_part_with_its_instance(
    relational_probabilistic_circuit, scenario
):
    tables = relational_probabilistic_circuit.training_tables(list(scenario))

    assert tables.part_tables["objects"].example_positions == [
        position for position, room in enumerate(scenario) for _ in room.objects
    ]


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
        class_diagram=class_diagram,
        owner=class_diagram.get_wrapped_class(clazz=SceneRoom),
    )

    node_of = {node.name: node for node in graph.nodes()}
    assert {
        relation.edge_among(node_of=node_of) for relation in relations.relations
    } == set(graph.edges())


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
            class_diagram=class_diagram,
            owner=class_diagram.get_wrapped_class(clazz=SceneRoom),
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
        for relation in CausalRelationsOfClass(
            class_diagram=class_diagram, owner=room
        ).relations
        for endpoint in (relation.first, relation.second)
        if isinstance(endpoint, AttributeEndpoint)
        and endpoint.fields[0] == room.get_field(name="objects")
    } == {
        (
            room.get_field(name="objects"),
            class_diagram.get_wrapped_class(clazz=SceneObject).get_field(name="type"),
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
        for relation in CausalRelationsOfClass(
            class_diagram=class_diagram, owner=scene
        ).relations
        for endpoint in (relation.first, relation.second)
        if isinstance(endpoint, StatisticEndpoint)
        and endpoint.collection == scene.get_field(name="rooms")
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


# %% edges between an instance and its parts


@dataclass(kw_only=True)
class FixedDependenceCheck(ExampleToPartCheck):
    """
    Keeps one fixed dependence between an instance and a part attribute, whatever the
    rows say.
    """

    dependence: ExampleToPartDependence = field(kw_only=True)
    """
    The dependence it keeps.
    """

    def dependences_given_another_attribute(
        self,
        example_rows: pd.DataFrame,
        part_collection: str,
        table: PartTrainingTable,
        adjacency: PartAttributeAdjacency,
    ) -> list[ExampleToPartDependence]:
        return [self.dependence]


@pytest.fixture
def room_type_dependence(relational_probabilistic_circuit) -> ExampleToPartDependence:
    """
    :return: A dependence of the rooms' first attribute on their objects' type.
    """
    return ExampleToPartDependence(
        example_attribute=relational_probabilistic_circuit.class_probabilistic_circuit.variables[
            0
        ].name,
        part_attribute="type",
        part_collection="objects",
        probability=0.0,
    )


def edges_between_an_instance_and_a_part_attribute(
    graph: CausalGraph,
) -> set[GraphEdge]:
    """
    :param graph: A learned graph.
    :return: Its edges that join an instance attribute to a part attribute.
    """
    return {
        edge
        for edge in graph.edges()
        if {edge.first.level, edge.second.level}
        == {NodeLevel.EXAMPLE_ATTRIBUTE, NodeLevel.PART_ATTRIBUTE}
    }


def test_without_a_check_no_edge_joins_an_instance_and_a_part_attribute(
    relational_probabilistic_circuit, scenario, class_diagram
):
    graph = (
        RelationalCausalStructureLearner(class_diagram, EveryPairJoinedLearner())
        .learn(relational_probabilistic_circuit, list(scenario))
        .selector.graph
    )

    assert edges_between_an_instance_and_a_part_attribute(graph) == set()


def test_a_kept_dependence_runs_from_the_instance_to_the_part_attribute(
    relational_probabilistic_circuit, scenario, class_diagram, room_type_dependence
):
    graph = (
        RelationalCausalStructureLearner(
            class_diagram,
            example_to_part_check=FixedDependenceCheck(dependence=room_type_dependence),
        )
        .learn(relational_probabilistic_circuit, list(scenario))
        .selector.graph
    )

    assert edges_between_an_instance_and_a_part_attribute(graph) == {
        GraphEdge.directed_edge(
            graph.node(room_type_dependence.example_attribute),
            graph.node(f"{SceneRoom.__name__}.objects.type"),
        )
    }


def test_a_kept_dependence_is_reversed_where_the_tiers_say_so(
    relational_probabilistic_circuit, scenario, class_diagram, room_type_dependence
):
    part_attribute = f"{SceneRoom.__name__}.objects.type"
    domain_knowledge = DomainCausalKnowledge(
        tiers=[[part_attribute], [room_type_dependence.example_attribute]]
    )
    graph = (
        RelationalCausalStructureLearner(
            class_diagram,
            example_to_part_check=FixedDependenceCheck(dependence=room_type_dependence),
        )
        .learn(relational_probabilistic_circuit, list(scenario), domain_knowledge)
        .selector.graph
    )

    assert edges_between_an_instance_and_a_part_attribute(graph) == {
        GraphEdge.directed_edge(
            graph.node(part_attribute),
            graph.node(room_type_dependence.example_attribute),
        )
    }


# %% the check on robots whose skill sets each attempt's arm


def robots_whose_skill_sets_the_arm(
    robot_count: int = 150, seed: int = 0
) -> list[PickingRobot]:
    """
    :param robot_count: How many robots to draw.
    :param seed: Seed of the draw.
    :return: Robots of skill 0 or 1 with two to four attempts each; each attempt's arm
        follows the robot's skill, and whether it grasps is a coin flip.
    """
    random_state = np.random.default_rng(seed)
    robots = []
    for _ in range(robot_count):
        skill = float(random_state.integers(0, 2))
        robots.append(
            PickingRobot(
                skill=skill,
                attempts=[
                    GraspAttempt(
                        arm=skill + random_state.normal(0.0, 0.3),
                        grasped=bool(random_state.random() < 0.5),
                    )
                    for _ in range(random_state.integers(2, 5))
                ],
            )
        )
    return robots


def test_the_check_joins_the_skill_to_the_arm_it_sets():
    robots = robots_whose_skill_sets_the_arm()
    circuit = RelationalProbabilisticCircuit().fit(robots)

    graph = (
        RelationalCausalStructureLearner(
            ClassDiagram(classes=[PickingRobot, GraspAttempt]),
            example_to_part_check=ExampleToPartCheck(),
        )
        .learn(circuit, robots)
        .selector.graph
    )

    assert edges_between_an_instance_and_a_part_attribute(graph) == {
        GraphEdge.directed_edge(
            graph.node(f"{PickingRobot.__name__}.skill"),
            graph.node(f"{PickingRobot.__name__}.attempts.arm"),
        )
    }
