"""
Tests that a causal query takes its adjustment set from the causal structure learned for
its class: chosen when the query marks no ``confounder``, checked when it does.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd
import pytest
from typing_extensions import Any
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
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
)
from random_events.product_algebra import Event
from random_events.variable import Integer

from krrood.class_diagrams.causal_relation import (
    AttributeEndpoint,
    CausalRelation,
    CausalRelationsOfClass,
)
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.exceptions import (
    EffectNotIdentified,
    NoCauseVariablesForRanking,
)
from krrood.entity_query_language.factories import a, cause, confounder
from krrood.entity_query_language.query.match import Match
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


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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
    class_diagram = ClassDiagram(classes=[SummerDay])
    day = class_diagram.get_wrapped_class(clazz=SummerDay)
    relations = CausalRelationsOfClass(class_diagram=class_diagram, owner=day)
    relations.replace_relations(
        relations=(
        CausalRelation(
            source=day,
            target=day,
            first=AttributeEndpoint(fields=(day.get_field(name=edge.first),)),
            second=AttributeEndpoint(fields=(day.get_field(name=edge.second),)),
            mark_at_first=EdgeMark.TAIL,
            mark_at_second=EdgeMark.ARROW if edge.directed else EdgeMark.TAIL,
        )
        for edge in edges
        )
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
        FieldEdge(first="season", second="ice_cream"),
        FieldEdge(first="season", second="drowning"),
        FieldEdge(first="heat", second="sunscreen"),
        FieldEdge(first="heat", second="drowning"),
    ]


def summer_circuit_registry(causes: list[str]) -> CausalCircuitRegistry:
    """
    :param causes: The ``SummerDay`` fields to stratify by and register as causes.
    :return: A causal circuit over the summer's rows with those causes.
    """
    rows = summer_rows()
    variables = infer_variables_from_dataframe(rows)
    by_name = {annotated.variable.name: annotated.variable for annotated in variables}
    cause_variables = [by_name[f"SummerDay.{name}"] for name in causes]
    effect_variable = by_name["SummerDay.drowning"]
    circuit = StratifiedLearning(
        variables=[variable.name for variable in cause_variables],
        method=JointProbabilityTree(),
    ).fit(rows, variables)
    return CausalCircuitRegistry(
        circuits={
            SummerDay: CausalCircuit.from_probabilistic_circuit(
                circuit,
                MarginalDeterminismTreeNode.from_variable_order(cause_variables),
                cause_variables,
                [effect_variable],
            )
        }
    )


@pytest.fixture
def summer_backend_registry() -> CausalCircuitRegistry:
    """
    :return: A causal circuit over the summer's rows with the ice cream sales and the
        season as causes, so that both can be intervened on and adjusted for.
    """
    return summer_circuit_registry(["ice_cream", "season"])


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
    ).rank_causes(expression=ice_cream_query(declares_the_season=False))
    declared = ProbabilisticBackend(model_registry=summer_backend_registry).rank_causes(
        expression=ice_cream_query(declares_the_season=True)
    )

    assert [candidate.effect_probability_given_region for candidate in learned] == [
        candidate.effect_probability_given_region for candidate in declared
    ]


def test_backend_without_a_structure_for_the_class_chooses_nothing():
    parameters = UnderspecifiedParameters(statement=ice_cream_query(declares_the_season=False))

    ProbabilisticBackend().choose_adjustment_sets(
        parameters=parameters, expression=ice_cream_query(declares_the_season=False)
    )

    assert parameters.derived_adjustment_variables == {}


def test_each_cause_candidate_gets_its_own_adjustment_set():
    query = a(SummerDay)(
        season=..., ice_cream=cause, sunscreen=cause, heat=..., drowning=...
    )
    query.causes_effect(query.drowning == 1)
    parameters = UnderspecifiedParameters(statement=query)
    backend = ProbabilisticBackend(
        causal_structures=summer_structures(true_summer_edges())
    )

    backend.choose_adjustment_sets(parameters=parameters, expression=query)

    assert {
        cause_variable.name: [
            variable.name
            for variable in parameters.adjustment_variables_of(cause_variable=cause_variable)
        ]
        for cause_variable in parameters.search_cause_variables
    } == {
        "SummerDay.ice_cream": ["SummerDay.season"],
        "SummerDay.sunscreen": ["SummerDay.heat"],
    }


# %% refusing


def test_declared_invalid_confounder_is_refused():
    mediated = summer_structures(
        [FieldEdge(first="ice_cream", second="season"), FieldEdge(first="season", second="drowning")]
    )
    query = ice_cream_query(declares_the_season=True)

    with pytest.raises(InvalidAdjustmentSet):
        ProbabilisticBackend(causal_structures=mediated).choose_adjustment_sets(
            parameters=UnderspecifiedParameters(statement=query), expression=query
        )


def test_partially_identified_effect_raises_effect_not_identified():
    undecided = summer_structures(
        [
            FieldEdge(first="season", second="drowning"),
            FieldEdge(first="ice_cream", second="drowning"),
            FieldEdge(first="season", second="ice_cream", directed=False),
        ]
    )
    query = ice_cream_query(declares_the_season=False)

    with pytest.raises(EffectNotIdentified):
        ProbabilisticBackend(causal_structures=undecided).choose_adjustment_sets(
            parameters=UnderspecifiedParameters(statement=query), expression=query
        )


# %% the relational path


@dataclass(kw_only=True)
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
        *, rows: pd.DataFrame,
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
    parameters = UnderspecifiedParameters(statement=query)
    [x] = parameters.search_cause_variables
    [y] = parameters.effect_variables_from_causes_effect
    learner = RelationalCausalStructureLearner(
        ClassDiagram(classes=[SceneRoom, SceneObject, KRROODPosition, KRROODOrientation]),
        FixedEdgesLearner(parent_names={x.name: [count.name], y.name: [count.name]}),
    )
    backend = ProbabilisticBackend(
        causal_structures={SceneRoom: learner.learn(circuit, list(scenario))}
    )

    backend.choose_adjustment_sets(parameters=parameters, expression=query)
    np.random.seed(0)
    causal_circuit = RelationalCircuitRegistry(
        relational_probabilistic_circuit=circuit
    ).get_model(parameters=parameters)

    assert count.name in {
        variable.name for variable in causal_circuit.probabilistic_circuit.variables
    }


# %% an effect identified only up to a few adjustment sets


def effect_probability_within(
    interventional: ProbabilisticCircuit, region: Event, effect_event: Event
) -> float:
    """
    :param interventional: A joint circuit over the cause and the effect.
    :param region: A region of the cause.
    :param effect_event: The effect's condition.
    :return: The effect's probability within one region of the cause.
    """
    truncated = interventional.truncated(
        region.fill_missing_variables_pure(interventional.variables)
    ).distribution
    return float(
        truncated.probability(
            effect_event.fill_missing_variables_pure(truncated.variables)
        )
    )


def test_range_spans_the_answers_of_each_candidate():
    registry = summer_circuit_registry(["ice_cream", "season"])
    undecided = summer_structures(
        [
            FieldEdge(first="season", second="drowning"),
            FieldEdge(first="ice_cream", second="drowning"),
            FieldEdge(first="season", second="ice_cream", directed=False),
        ]
    )
    query = ice_cream_query(declares_the_season=False)
    [answer] = ProbabilisticBackend(
        model_registry=registry, causal_structures=undecided
    ).interventional_range(expression=query)
    causal_circuit = registry.circuits[SummerDay]
    [ice_cream, season] = causal_circuit.causal_variables
    [drowning] = causal_circuit.effect_variables
    effect_event = UnderspecifiedParameters(
        statement=query
    ).truncation_assignments_from_where_conditions
    candidates = [
        causal_circuit.backdoor_adjustment(ice_cream, drowning, adjustment)
        for adjustment in ([], [season])
    ]

    assert [
        (region.lowest_effect_probability, region.highest_effect_probability)
        for region in answer.regions
    ] == [
        (
            min(probabilities),
            max(probabilities),
        )
        for probabilities in (
            [
                effect_probability_within(candidate, region.event, effect_event)
                for candidate in candidates
            ]
            for region in causal_circuit._extract_disjoint_regions_for_variable(
                ice_cream
            )
        )
    ]


# %% ranking every possible cause of an effect


def drowning_query(**causes: Any) -> Match[SummerDay]:
    """
    :param causes: The fields to mark, by name.
    :return: What causes drowning, with the given fields marked.
    """
    fields = dict(season=..., ice_cream=..., sunscreen=..., heat=..., drowning=...)
    fields.update(causes)
    query = a(SummerDay)(**fields)
    query.causes_effect(query.drowning == 1)
    return query


def test_effect_only_ranking_equals_marking_every_possible_ancestor():
    backend = ProbabilisticBackend(
        model_registry=summer_circuit_registry(["season", "heat"]),
        causal_structures=summer_structures(true_summer_edges()),
    )

    effect_only = backend.rank_causes(expression=drowning_query())
    marked = backend.rank_causes(expression=drowning_query(season=cause, heat=cause))

    assert [
        (candidate.cause_variable.name, candidate.effect_probability_given_region)
        for candidate in effect_only
    ] == [
        (candidate.cause_variable.name, candidate.effect_probability_given_region)
        for candidate in marked
    ]


def test_effect_only_ranking_without_a_learned_structure_raises():
    with pytest.raises(NoCauseVariablesForRanking):
        ProbabilisticBackend(
            model_registry=summer_circuit_registry(["season", "heat"])
        ).rank_causes(expression=drowning_query())
