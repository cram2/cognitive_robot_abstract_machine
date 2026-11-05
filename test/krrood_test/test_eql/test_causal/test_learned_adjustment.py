"""
Tests that a causal query takes its adjustment set from the causal structure learned for
its class: chosen when the query marks no ``confounder``, checked when it does.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import pytest
from probabilistic_model.causal_structure.domain_causal_knowledge import (
    DomainCausalKnowledge,
)
from probabilistic_model.causal_structure.exceptions import InvalidAdjustmentSet
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    EdgeMark,
    GraphEdge,
    GraphNode,
    NodeLevel,
)
from probabilistic_model.causal_structure.learning import (
    CausalStructureLearner,
    LearnedCausalStructure,
    RelationalCausalStructureLearner,
)
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.learning_method import StratifiedLearning
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    MarginalDeterminismTreeNode,
)
from random_events.variable import Integer

from krrood.class_diagrams.causal_relation import (
    AttributeEndpoint,
    CausalRelation,
    CausalRelationsOfClass,
)
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import EffectNotIdentified
from krrood.entity_query_language.factories import a, cause, confounder
from krrood.parametrization.model_registries import (
    CausalCircuitRegistry,
    RelationalCircuitRegistry,
)
from krrood.parametrization.parameterizer import UnderspecifiedParameters

from ...dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneRoom,
)
from ...test_feature_extraction.test_relational_probabilistic_circuit import (  # noqa: F401  # fixtures
    layered_relational_probabilistic_circuit,
    relational_probabilistic_circuit,
    scenario,
)

# %% a summer whose season drives both ice cream sales and drownings


@dataclass
class SummerDay:
    season: int
    ice_cream: int
    sunscreen: int
    heat: int
    drowning: int


def summer_rows(row_count: int = 2000, seed: int = 0) -> pd.DataFrame:
    """
    :return: Rows where the season drives the ice cream sales and the drownings, and
        neither ice cream nor sunscreen causes a drowning.
    """
    random_state = np.random.default_rng(seed)
    season = random_state.integers(0, 2, row_count)
    heat = random_state.integers(0, 2, row_count)

    def noisy(values: np.ndarray) -> np.ndarray:
        return values ^ (random_state.random(row_count) < 0.2)

    return pd.DataFrame(
        {
            "SummerDay.season": season,
            "SummerDay.ice_cream": noisy(season),
            "SummerDay.sunscreen": noisy(heat),
            "SummerDay.heat": heat,
            "SummerDay.drowning": noisy(season | heat),
        }
    )


@dataclass
class FieldEdge:
    """
    An edge between two :class:`SummerDay` fields.
    """

    first: str
    """
    The name of the field at the tail.
    """

    second: str
    """
    The name of the field at the other end.
    """

    directed: bool = True
    """
    Whether the edge points from :attr:`first` to :attr:`second`; else it is undirected.
    """


def summer_structures(edges: list[FieldEdge]) -> dict[type, LearnedCausalStructure]:
    """
    :param edges: The edges between ``SummerDay`` fields.
    :return: The structure of ``SummerDay`` with these causal relations in a class
        diagram, and the summer's rows to rank sets by.
    """
    class_diagram = ClassDiagram([SummerDay])
    day = class_diagram.get_wrapped_class(SummerDay)
    relations = CausalRelationsOfClass(class_diagram, day)
    relations.replace_relations(
        CausalRelation(
            source=day,
            target=day,
            first=AttributeEndpoint((day.get_field(edge.first),)),
            second=AttributeEndpoint((day.get_field(edge.second),)),
            mark_at_first=EdgeMark.TAIL,
            mark_at_second=EdgeMark.ARROW if edge.directed else EdgeMark.TAIL,
        )
        for edge in edges
    )
    nodes = [
        GraphNode(
            variable=Integer(f"SummerDay.{summer_field.name}"),
            level=NodeLevel.EXAMPLE_ATTRIBUTE,
        )
        for summer_field in fields(SummerDay)
    ]
    return {SummerDay: LearnedCausalStructure(relations, nodes, summer_rows())}


def true_summer_edges() -> list[FieldEdge]:
    """
    :return: The edges the summer's rows are drawn from.
    """
    return [
        FieldEdge("season", "ice_cream"),
        FieldEdge("season", "drowning"),
        FieldEdge("heat", "sunscreen"),
        FieldEdge("heat", "drowning"),
    ]


@pytest.fixture
def summer_backend_registry() -> CausalCircuitRegistry:
    """
    :return: A causal circuit over the summer's rows, stratified by the season and the
        ice cream sales so that both can be intervened on and adjusted for.
    """
    rows = summer_rows()
    variables = infer_variables_from_dataframe(rows)
    by_name = {annotated.variable.name: annotated.variable for annotated in variables}
    circuit = StratifiedLearning(
        variables=["SummerDay.ice_cream", "SummerDay.season"],
        method=JointProbabilityTree(),
    ).fit(rows, variables)
    ice_cream, season = by_name["SummerDay.ice_cream"], by_name["SummerDay.season"]
    return CausalCircuitRegistry(
        {
            SummerDay: CausalCircuit.from_probabilistic_circuit(
                circuit,
                MarginalDeterminismTreeNode.from_variable_order([ice_cream, season]),
                [ice_cream, season],
                [by_name["SummerDay.drowning"]],
            )
        }
    )


def ice_cream_query(declares_the_season: bool):
    """
    :param declares_the_season: Whether the season is marked ``confounder``.
    :return: Does ice cream cause drowning?
    """
    query = a(SummerDay)(
        season=confounder if declares_the_season else ...,
        ice_cream=cause,
        sunscreen=...,
        heat=...,
        drowning=...,
    )
    query.causes_effect(query.drowning == 1)
    return query


# %% choosing the adjustment set


def test_learned_graph_supplies_the_confounder(summer_backend_registry):
    learned = ProbabilisticBackend(
        model_registry=summer_backend_registry,
        causal_structures=summer_structures(true_summer_edges()),
    ).rank_causes(ice_cream_query(declares_the_season=False))
    declared = ProbabilisticBackend(model_registry=summer_backend_registry).rank_causes(
        ice_cream_query(declares_the_season=True)
    )

    assert [candidate.effect_probability_given_region for candidate in learned] == [
        candidate.effect_probability_given_region for candidate in declared
    ]


def test_backend_without_a_structure_for_the_class_chooses_nothing():
    parameters = UnderspecifiedParameters(ice_cream_query(declares_the_season=False))

    ProbabilisticBackend().choose_adjustment_sets(
        parameters, ice_cream_query(declares_the_season=False)
    )

    assert parameters.derived_adjustment_variables == {}


def test_each_cause_candidate_gets_its_own_adjustment_set():
    query = a(SummerDay)(
        season=..., ice_cream=cause, sunscreen=cause, heat=..., drowning=...
    )
    query.causes_effect(query.drowning == 1)
    parameters = UnderspecifiedParameters(query)
    backend = ProbabilisticBackend(
        causal_structures=summer_structures(true_summer_edges())
    )

    backend.choose_adjustment_sets(parameters, query)

    assert {
        cause_variable.name: [
            variable.name
            for variable in parameters.adjustment_variables_of(cause_variable)
        ]
        for cause_variable in parameters.search_cause_variables
    } == {
        "SummerDay.ice_cream": ["SummerDay.season"],
        "SummerDay.sunscreen": ["SummerDay.heat"],
    }


# %% refusing


def test_declared_invalid_confounder_is_refused():
    mediated = summer_structures(
        [FieldEdge("ice_cream", "season"), FieldEdge("season", "drowning")]
    )
    query = ice_cream_query(declares_the_season=True)

    with pytest.raises(InvalidAdjustmentSet):
        ProbabilisticBackend(causal_structures=mediated).choose_adjustment_sets(
            UnderspecifiedParameters(query), query
        )


def test_partially_identified_effect_raises_effect_not_identified():
    undecided = summer_structures(
        [
            FieldEdge("season", "drowning"),
            FieldEdge("ice_cream", "drowning"),
            FieldEdge("season", "ice_cream", directed=False),
        ]
    )
    query = ice_cream_query(declares_the_season=False)

    with pytest.raises(EffectNotIdentified):
        ProbabilisticBackend(causal_structures=undecided).choose_adjustment_sets(
            UnderspecifiedParameters(query), query
        )


# %% the relational path


@dataclass
class FixedEdgesLearner(CausalStructureLearner):
    """
    Learns the given edges between the nodes it is asked about, whatever the rows say.
    """

    parent_names: dict[str, list[str]]
    """
    Per node name, the names of the nodes it is learned to be a child of.
    """

    def learn(
        self,
        rows: pd.DataFrame,
        nodes: list[GraphNode],
        domain_knowledge: DomainCausalKnowledge,
    ) -> CausalGraph:
        by_name = {node.name: node for node in nodes}
        return CausalGraph.from_edges(
            nodes,
            [
                GraphEdge.directed_edge(by_name[parent], by_name[child])
                for child, parents in self.parent_names.items()
                if child in by_name
                for parent in parents
                if parent in by_name
            ],
        )


def room_query():
    """
    :return: Does a room's x position cause its y position, among rooms of four
        objects?
    """
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=cause, y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    query.causes_effect(query.position.y == 1.0)
    return query


@pytest.mark.parametrize(
    "fitted_circuit",
    ["relational_probabilistic_circuit", "layered_relational_probabilistic_circuit"],
)
def test_derived_count_survives_the_relational_trim(fitted_circuit, scenario, request):
    circuit = request.getfixturevalue(fitted_circuit)
    count = circuit.exchangeable_distribution_templates["objects"].latent_variables[0]
    query = room_query()
    parameters = UnderspecifiedParameters(query)
    [x] = parameters.search_cause_variables
    [y] = parameters.effect_variables_from_causes_effect
    learner = RelationalCausalStructureLearner(
        ClassDiagram([SceneRoom, SceneObject, KRROODPosition, KRROODOrientation]),
        FixedEdgesLearner({x.name: [count.name], y.name: [count.name]}),
    )
    backend = ProbabilisticBackend(
        causal_structures={SceneRoom: learner.learn(circuit, list(scenario))}
    )

    backend.choose_adjustment_sets(parameters, query)
    np.random.seed(0)
    causal_circuit = RelationalCircuitRegistry(
        relational_probabilistic_circuit=circuit
    ).get_model(parameters)

    assert count.name in {
        variable.name for variable in causal_circuit.probabilistic_circuit.variables
    }
