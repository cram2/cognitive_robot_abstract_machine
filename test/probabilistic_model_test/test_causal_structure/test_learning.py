import pandas as pd
import pytest
from random_events.variable import Continuous, Integer

from probabilistic_model.causal_structure.background_knowledge import (
    BackgroundKnowledge,
    DeterministicCopy,
)
from probabilistic_model.causal_structure.exceptions import UnknownGraphNode
from probabilistic_model.causal_structure.graph import GraphEdge
from probabilistic_model.causal_structure.identification import IdentifiedEffect
from probabilistic_model.causal_structure.learning import (
    LearnedCausalStructure,
    PeterClarkStructureLearner,
)

from .structural_causal_models import chain_with_a_copy, collider, season_fork

# %% learning from one table


def structure_of(
    rows: pd.DataFrame, knowledge: BackgroundKnowledge | None = None
) -> LearnedCausalStructure:
    """
    :param rows: Integer columns.
    :param knowledge: What is known in advance.
    :return: The structure learned from them.
    """
    return LearnedCausalStructure.from_rows(
        rows, [Integer(name) for name in rows.columns], knowledge=knowledge
    )


def test_fork_with_tiers_is_recovered():
    knowledge = BackgroundKnowledge(
        tiers=[["season"], ["ice_cream_sales", "drowning_incidents"]]
    )

    graph = structure_of(season_fork(), knowledge).selector.graph

    assert set(graph.edges()) == {
        GraphEdge.directed("season", "ice_cream_sales"),
        GraphEdge.directed("season", "drowning_incidents"),
    }


def test_collider_is_oriented_without_tiers():
    graph = structure_of(collider()).selector.graph

    assert {
        (edge.parent, edge.child) for edge in graph.edges() if edge.is_directed
    } == {
        ("A", "C"),
        ("B", "C"),
    }


def test_deterministic_copy_is_restored_as_child():
    knowledge = BackgroundKnowledge(
        tiers=[["A"], ["B"]],
        deterministic_copies=[DeterministicCopy(copy="D", source="B")],
    )

    graph = structure_of(chain_with_a_copy(), knowledge).selector.graph

    assert set(graph.edges()) == {
        GraphEdge.directed("A", "B"),
        GraphEdge.directed("B", "D"),
    }


def test_knowledge_naming_an_unknown_variable_raises():
    knowledge = BackgroundKnowledge(tiers=[["winter"]])

    with pytest.raises(UnknownGraphNode):
        structure_of(season_fork(), knowledge)


def test_learned_fork_supplies_the_confounder():
    knowledge = BackgroundKnowledge(
        tiers=[["season"], ["ice_cream_sales", "drowning_incidents"]]
    )
    structure = structure_of(season_fork(), knowledge)

    identification = structure.selector.identify(
        "ice_cream_sales",
        "drowning_incidents",
        {"season"},
        structure.training_rows,
    )

    assert identification == IdentifiedEffect(frozenset({"season"}))


def test_continuous_columns_are_tested_after_discretization():
    rows = season_fork().astype(float)
    rows["season"] += pd.Series(range(len(rows))) / (10 * len(rows))
    knowledge = BackgroundKnowledge(
        tiers=[["season"], ["ice_cream_sales", "drowning_incidents"]]
    )

    graph = LearnedCausalStructure.from_rows(
        rows, [Continuous(name) for name in rows.columns], knowledge=knowledge
    ).selector.graph

    assert {(edge.parent, edge.child) for edge in graph.edges()} == {
        ("season", "ice_cream_sales"),
        ("season", "drowning_incidents"),
    }
