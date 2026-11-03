import pandas as pd
import pytest
from random_events.variable import Continuous, Integer

from probabilistic_model.causal_structure.domain_causal_knowledge import (
    DomainCausalKnowledge,
    DeterministicCopy,
)
from probabilistic_model.causal_structure.exceptions import UnknownGraphNode
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    GraphEdge,
    GraphNode,
    NodeLevel,
)
from probabilistic_model.causal_structure.identification import (
    AdjustmentSelector,
    IdentifiedEffect,
)
from probabilistic_model.causal_structure.learning import PeterClarkStructureLearner

from .structural_causal_models import chain_with_a_copy, collider, season_fork

# %% learning from one table


def graph_of(
    rows: pd.DataFrame,
    knowledge: DomainCausalKnowledge | None = None,
    variable_type: type[Integer | Continuous] = Integer,
) -> CausalGraph:
    """
    :param rows: One column per variable.
    :param knowledge: What is known in advance.
    :param variable_type: The type of every variable.
    :return: The graph learned from them.
    """
    nodes = [
        GraphNode(variable=variable_type(name), level=NodeLevel.EXAMPLE_ATTRIBUTE)
        for name in rows.columns
    ]
    return PeterClarkStructureLearner().learn(
        rows, nodes, knowledge or DomainCausalKnowledge()
    )


def test_fork_with_tiers_is_recovered():
    knowledge = DomainCausalKnowledge(
        tiers=[["season"], ["ice_cream_sales", "drowning_incidents"]]
    )

    graph = graph_of(season_fork(), knowledge)

    assert set(graph.edges()) == {
        GraphEdge.directed("season", "ice_cream_sales"),
        GraphEdge.directed("season", "drowning_incidents"),
    }


def test_collider_is_oriented_without_tiers():
    graph = graph_of(collider())

    assert {
        (edge.parent, edge.child) for edge in graph.edges() if edge.is_directed
    } == {
        ("A", "C"),
        ("B", "C"),
    }


def test_deterministic_copy_is_restored_as_child():
    knowledge = DomainCausalKnowledge(
        tiers=[["A"], ["B"]],
        deterministic_copies=[DeterministicCopy(copy="D", source="B")],
    )

    graph = graph_of(chain_with_a_copy(), knowledge)

    assert set(graph.edges()) == {
        GraphEdge.directed("A", "B"),
        GraphEdge.directed("B", "D"),
    }


def test_knowledge_naming_an_unknown_variable_raises():
    knowledge = DomainCausalKnowledge(tiers=[["winter"]])

    with pytest.raises(UnknownGraphNode):
        graph_of(season_fork(), knowledge)


def test_learned_fork_supplies_the_confounder():
    knowledge = DomainCausalKnowledge(
        tiers=[["season"], ["ice_cream_sales", "drowning_incidents"]]
    )
    rows = season_fork()

    identification = AdjustmentSelector(graph_of(rows, knowledge)).identify(
        "ice_cream_sales", "drowning_incidents", {"season"}, rows
    )

    assert identification == IdentifiedEffect(frozenset({"season"}))


def test_continuous_columns_are_tested_after_discretization():
    rows = season_fork().astype(float)
    rows["season"] += pd.Series(range(len(rows))) / (10 * len(rows))
    knowledge = DomainCausalKnowledge(
        tiers=[["season"], ["ice_cream_sales", "drowning_incidents"]]
    )

    graph = graph_of(rows, knowledge, Continuous)

    assert {(edge.parent, edge.child) for edge in graph.edges()} == {
        ("season", "ice_cream_sales"),
        ("season", "drowning_incidents"),
    }
