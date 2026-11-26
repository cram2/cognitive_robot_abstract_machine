"""
Tests for learning an ancestral graph with fast causal inference, which allows hidden
confounders.
"""

from __future__ import annotations

import pandas as pd
import pytest
from random_events.variable import Integer

from probabilistic_model.causal_structure.ancestral_graph import AncestralGraph
from probabilistic_model.causal_structure.domain_causal_knowledge import (
    DeterministicCopy,
    DomainCausalKnowledge,
    KnownEdge,
)
from probabilistic_model.causal_structure.exceptions import (
    UnsupportedKnowledgeUnderLatentConfounding,
)
from probabilistic_model.causal_structure.graph import (
    EdgeMark,
    EdgeMarkedGraph,
    GraphEdge,
    GraphNode,
    NodeLevel,
)
from probabilistic_model.causal_structure.learning import (
    FastCausalInferenceStructureLearner,
    KnowledgeKind,
)

from . import ancestral_graphs, structural_causal_models

# %% learning from one table


def learned_graph(
    rows: pd.DataFrame, domain_knowledge: DomainCausalKnowledge | None = None
) -> AncestralGraph:
    """
    :param rows: One column per variable.
    :param domain_knowledge: What is known in advance.
    :return: The ancestral graph fast causal inference learns from them.
    """
    nodes = [
        GraphNode(variable=Integer(name), level=NodeLevel.EXAMPLE_ATTRIBUTE)
        for name in rows.columns
    ]
    return FastCausalInferenceStructureLearner().learn(
        rows, nodes, domain_knowledge or DomainCausalKnowledge()
    )


def marks_of(graph: EdgeMarkedGraph) -> set[frozenset[tuple[str, EdgeMark]]]:
    """
    :param graph: A graph.
    :return: Each edge as the names of its ends with the mark at each, whichever end
        it names first.
    """
    return {
        frozenset(
            {
                (edge.first.name, edge.mark_at_first),
                (edge.second.name, edge.mark_at_second),
            }
        )
        for edge in graph.edges()
    }


@pytest.mark.parametrize(
    "rows, truth",
    [
        (
            structural_causal_models.measured_confounder(),
            ancestral_graphs.measured_confounder(),
        ),
        (
            structural_causal_models.pure_hidden_confounding(),
            ancestral_graphs.pure_hidden_confounding(),
        ),
        (
            structural_causal_models.hidden_cause_and_effect_confounder(),
            ancestral_graphs.hidden_cause_and_effect_confounder(),
        ),
    ],
)
def test_the_learned_graph_is_the_partial_ancestral_graph_of_the_truth(
    rows: pd.DataFrame, truth: ancestral_graphs.TrueGraph
):
    """
    The learned graph is the partial ancestral graph of the truth.
    """
    graph = learned_graph(rows)

    assert (type(graph), marks_of(graph)) == (
        AncestralGraph,
        marks_of(truth.partial_ancestral_graph()),
    )


def test_pure_hidden_confounding_is_learned_with_circles_at_both_causes():
    """
    Pure hidden confounding is learned with circles at both causes.
    """
    graph = learned_graph(structural_causal_models.pure_hidden_confounding())
    w, x, y = (graph.node(name) for name in "WXY")

    assert marks_of(graph) == marks_of(
        AncestralGraph.from_edges(
            nodes=[w, x, y],
            edges=[
                GraphEdge.partially_directed_edge(circle_end=w, arrow_end=x),
                GraphEdge.partially_directed_edge(circle_end=y, arrow_end=x),
            ],
        )
    )


def test_the_learner_learns_ancestral_graphs():
    """
    The learner learns ancestral graphs.
    """
    assert FastCausalInferenceStructureLearner.graph_class is AncestralGraph


# %% domain causal knowledge


@pytest.mark.parametrize(
    "domain_knowledge, kind",
    [
        (DomainCausalKnowledge(tiers=[["W"], ["X"], ["Y"]]), KnowledgeKind.TIERS),
        (
            DomainCausalKnowledge(forbidden_edges=[KnownEdge(parent="Y", child="W")]),
            KnowledgeKind.FORBIDDEN_EDGES,
        ),
    ],
)
def test_knowledge_that_would_claim_a_cause_is_refused(
    domain_knowledge: DomainCausalKnowledge, kind: KnowledgeKind
):
    """
    Knowledge that would claim a cause is refused.
    """
    with pytest.raises(UnsupportedKnowledgeUnderLatentConfounding) as refusal:
        learned_graph(
            structural_causal_models.pure_hidden_confounding(), domain_knowledge
        )

    assert refusal.value.kind is kind


def test_a_required_edge_is_learned_directed():
    """
    A required edge is learned directed.
    """
    graph = learned_graph(
        structural_causal_models.measured_confounder(),
        DomainCausalKnowledge(required_edges=[KnownEdge(parent="V", child="X")]),
    )

    assert graph.edge_between(graph.node("V"), graph.node("X")) == (
        GraphEdge.directed_edge(graph.node("V"), graph.node("X"))
    )


def test_a_deterministic_copy_is_restored_as_a_directed_leaf():
    """
    A deterministic copy is restored as a directed leaf.
    """
    rows = structural_causal_models.pure_hidden_confounding()
    rows["X_copy"] = rows["X"]

    graph = learned_graph(
        rows,
        DomainCausalKnowledge(
            deterministic_copies=[DeterministicCopy(source="X", copy="X_copy")]
        ),
    )

    copy = graph.node("X_copy")
    assert graph.edges_at(node=copy) == [GraphEdge.directed_edge(graph.node("X"), copy)]
