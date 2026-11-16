import pandas as pd
import pytest
from random_events.variable import Continuous, Integer

from probabilistic_model.causal_structure.domain_causal_knowledge import (
    DomainCausalKnowledge,
    DeterministicCopy,
)
from probabilistic_model.causal_structure.exceptions import (
    TooFewVariablesToScore,
    UnknownGraphNode,
)
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    GraphEdge,
    GraphNode,
    NodeLevel,
)
from probabilistic_model.causal_structure.identification import (
    AdjustmentSelector,
    AdjustmentSet,
    IdentifiedEffect,
)
from probabilistic_model.causal_structure.learning import (
    DiscretizedTableStructureLearner,
    GreedyEquivalenceSearchStructureLearner,
    PeterClarkStructureLearner,
)

from .structural_causal_models import chain_with_a_copy, collider, season_fork

# %% learning from one table


@pytest.fixture(
    params=[PeterClarkStructureLearner, GreedyEquivalenceSearchStructureLearner]
)
def learner(request: pytest.FixtureRequest) -> DiscretizedTableStructureLearner:
    """
    :return: Each learner of a graph from one table, with its defaults.
    """
    return request.param()


def graph_of(
    rows: pd.DataFrame,
    learner: DiscretizedTableStructureLearner,
    domain_knowledge: DomainCausalKnowledge | None = None,
    variable_type: type[Integer | Continuous] = Integer,
) -> CausalGraph:
    """
    :param rows: One column per variable.
    :param learner: What learns the graph.
    :param domain_knowledge: What is known in advance.
    :param variable_type: The type of every variable.
    :return: The graph learned from them.
    """
    nodes = [
        GraphNode(variable=variable_type(name), level=NodeLevel.EXAMPLE_ATTRIBUTE)
        for name in rows.columns
    ]
    return learner.learn(rows, nodes, domain_knowledge or DomainCausalKnowledge())


def test_fork_with_tiers_is_recovered(learner: DiscretizedTableStructureLearner):
    domain_knowledge = DomainCausalKnowledge(
        tiers=[["season"], ["ice_cream_sales", "drowning_incidents"]]
    )

    graph = graph_of(season_fork(), learner, domain_knowledge)

    season = graph.node("season")
    assert set(graph.edges()) == {
        GraphEdge.directed_edge(season, graph.node("ice_cream_sales")),
        GraphEdge.directed_edge(season, graph.node("drowning_incidents")),
    }


def test_collider_is_oriented_without_tiers(learner: DiscretizedTableStructureLearner):
    graph = graph_of(collider(), learner)

    assert {
        (edge.parent.name, edge.child.name)
        for edge in graph.edges()
        if edge.is_directed
    } == {
        ("A", "C"),
        ("B", "C"),
    }


def test_deterministic_copy_is_restored_as_child(
    learner: DiscretizedTableStructureLearner,
):
    domain_knowledge = DomainCausalKnowledge(
        tiers=[["A"], ["B"]],
        deterministic_copies=[DeterministicCopy(copy="D", source="B")],
    )

    graph = graph_of(
        chain_with_a_copy(), PeterClarkStructureLearner(), domain_knowledge
    )

    first, source, copy = graph.node("A"), graph.node("B"), graph.node("D")
    assert set(graph.edges()) == {
        GraphEdge.directed_edge(first, source),
        GraphEdge.directed_edge(source, copy),
    }


def test_scoring_a_graph_over_too_few_variables_raises():
    domain_knowledge = DomainCausalKnowledge(
        deterministic_copies=[DeterministicCopy(copy="D", source="B")]
    )

    with pytest.raises(TooFewVariablesToScore):
        graph_of(
            chain_with_a_copy(),
            GreedyEquivalenceSearchStructureLearner(),
            domain_knowledge,
        )


def test_knowledge_naming_an_unknown_variable_raises(
    learner: DiscretizedTableStructureLearner,
):
    domain_knowledge = DomainCausalKnowledge(tiers=[["winter"]])

    with pytest.raises(UnknownGraphNode):
        graph_of(season_fork(), learner, domain_knowledge)


def test_learned_fork_supplies_the_confounder(
    learner: DiscretizedTableStructureLearner,
):
    domain_knowledge = DomainCausalKnowledge(
        tiers=[["season"], ["ice_cream_sales", "drowning_incidents"]]
    )
    rows = season_fork()

    graph = graph_of(rows, learner, domain_knowledge)

    identification = AdjustmentSelector(graph).identify(
        graph.node("ice_cream_sales"),
        graph.node("drowning_incidents"),
        {graph.node("season")},
        rows,
    )

    assert identification == IdentifiedEffect(
        AdjustmentSet(frozenset({graph.node("season")}))
    )


def test_continuous_columns_are_tested_after_discretization(
    learner: DiscretizedTableStructureLearner,
):
    rows = season_fork().astype(float)
    rows["season"] += pd.Series(range(len(rows))) / (10 * len(rows))
    domain_knowledge = DomainCausalKnowledge(
        tiers=[["season"], ["ice_cream_sales", "drowning_incidents"]]
    )

    graph = graph_of(rows, learner, domain_knowledge, Continuous)

    assert {(edge.parent.name, edge.child.name) for edge in graph.edges()} == {
        ("season", "ice_cream_sales"),
        ("season", "drowning_incidents"),
    }


# %% orienting an edge by the tiers


@pytest.mark.parametrize(
    "cause, adjacent, parent",
    [
        ("B", "A", "A"),
        ("A", "B", "A"),
        ("C", "B", "C"),
        ("D", "A", "D"),
        ("A", "D", "A"),
    ],
    ids=[
        "adjacent_node_in_an_earlier_tier_reverses_the_edge",
        "cause_in_an_earlier_tier_keeps_it",
        "same_tier_keeps_it",
        "cause_without_tier_keeps_it",
        "adjacent_node_without_tier_keeps_it",
    ],
)
def test_edge_is_reversed_only_where_the_tiers_say_so(
    cause: str, adjacent: str, parent: str
):
    domain_knowledge = DomainCausalKnowledge(tiers=[["A"], ["B", "C"]])
    node_of = {
        name: GraphNode(variable=Integer(name), level=NodeLevel.EXAMPLE_ATTRIBUTE)
        for name in (cause, adjacent)
    }

    edge = domain_knowledge.orient_edge(node_of[cause], node_of[adjacent])

    assert edge == GraphEdge.directed_edge(
        node_of[parent], node_of[({cause, adjacent} - {parent}).pop()]
    )
