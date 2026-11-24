"""
Tests for choosing and checking adjustment sets in an ancestral graph, which allows
hidden confounders: the oracle cases, declared sets, and soundness against the true
directed acyclic graph.
"""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest
from typing_extensions import Callable

from probabilistic_model.causal_structure.ancestral_graph import AncestralGraph
from probabilistic_model.causal_structure.exceptions import InvalidAdjustmentSet
from probabilistic_model.causal_structure.graph import EdgeMark, GraphEdge
from probabilistic_model.causal_structure.identification import (
    AdjustmentCheck,
    AdjustmentCriterion,
    AdjustmentSelector,
    AdjustmentSet,
    GeneralizedAdjustmentCriterion,
    IdentifiedEffect,
    Identification,
    InvalidAdjustmentReason,
    InvalidDeclaredAdjustmentRefusal,
    InvalidDeclaredAdjustmentRepair,
    MemberGraphAdjustmentCriterion,
    UnidentifiedEffect,
    UnidentifiedReason,
)

from .ancestral_graphs import (
    TrueGraph,
    agrees_with,
    bare_confounded,
    declared_confounder_of_covariates,
    hidden_cause_and_effect_confounder,
    hidden_confounder_of_covariates,
    instrument_with_hidden_confounder,
    measured_confounder,
    node_named,
    pure_hidden_confounding,
    random_true_graph,
    two_parents,
)
from .graphs import confounded

# %% asking a graph about the effect of X on Y


def identification_in(graph: AncestralGraph) -> Identification:
    """
    :param graph: An ancestral graph over X, Y and others.
    :return: What it says about the effect of X on Y, any node eligible.
    """
    return AdjustmentSelector(graph).identify(
        graph.node("X"), graph.node("Y"), set(graph.nodes())
    )


def adjustment_of(*names: str) -> AdjustmentSet:
    """
    :param names: The names of the nodes to adjust for.
    :return: The adjustment set of those nodes.
    """
    return AdjustmentSet(frozenset(node_named(name) for name in names))


# %% the oracle cases


@pytest.mark.parametrize(
    "graph_of, expected",
    [
        (
            lambda: measured_confounder().partial_ancestral_graph(),
            IdentifiedEffect(adjustment_of("C")),
        ),
        (
            lambda: two_parents().partial_ancestral_graph(),
            IdentifiedEffect(adjustment_of()),
        ),
        (declared_confounder_of_covariates, IdentifiedEffect(adjustment_of("C"))),
    ],
)
def test_an_amenable_graph_identifies_the_effect_by_its_best_valid_set(
    graph_of: Callable[[], AncestralGraph], expected: Identification
):
    """
    An amenable graph identifies the effect by its best valid set.
    """
    assert identification_in(graph_of()) == expected


@pytest.mark.parametrize(
    "truth",
    [
        hidden_cause_and_effect_confounder(),
        hidden_confounder_of_covariates(),
        instrument_with_hidden_confounder(),
    ],
)
def test_a_possibly_causal_edge_that_may_hide_a_confounder_leaves_it_unidentified(
    truth: TrueGraph,
):
    """
    A possibly causal edge that may hide a confounder leaves it unidentified.
    """
    identification = identification_in(truth.partial_ancestral_graph())

    assert (
        identification.reason,
        identification.unsettled_edge.mark_at(node=node_named("X")),
    ) == (UnidentifiedReason.NOT_AMENABLE, EdgeMark.CIRCLE)


def test_an_edge_that_is_not_visible_names_itself_as_the_one_to_settle():
    """
    An edge that is not visible names itself as the one to settle.
    """
    graph = bare_confounded()

    identification = identification_in(graph)

    assert identification == UnidentifiedEffect(
        UnidentifiedReason.NOT_AMENABLE,
        unsettled_edge=GraphEdge.directed_edge(graph.node("X"), graph.node("Y")),
    )
    assert str(identification.unsettled_edge) in identification.correction


def test_pure_hidden_confounding_has_no_valid_set():
    """
    Pure hidden confounding has no valid set.
    """
    assert identification_in(
        pure_hidden_confounding().partial_ancestral_graph()
    ) == UnidentifiedEffect(UnidentifiedReason.NO_VALID_SET_AMONG_ELIGIBLE)


def test_a_selection_bias_edge_leaves_the_effect_unidentified():
    """
    A selection bias edge leaves the effect unidentified.
    """
    x, y, z = (node_named(name) for name in "XYZ")
    graph = AncestralGraph.from_edges(
        [x, y, z],
        [
            GraphEdge.directed_edge(x, y),
            GraphEdge(z, x, EdgeMark.TAIL, EdgeMark.TAIL),
        ],
    )

    assert identification_in(graph) == UnidentifiedEffect(
        UnidentifiedReason.SELECTION_BIAS_EDGE
    )


def test_a_graph_without_a_representative_leaves_the_effect_unidentified():
    """
    A graph without a representative leaves the effect unidentified.
    """
    a, b, x, y = (node_named(name) for name in "ABXY")
    graph = AncestralGraph.from_edges(
        [a, b, x, y],
        [
            GraphEdge.nondirected_edge(first=x, second=a),
            GraphEdge.nondirected_edge(first=a, second=y),
            GraphEdge.nondirected_edge(first=y, second=b),
            GraphEdge.nondirected_edge(first=b, second=x),
        ],
    )

    assert identification_in(graph) == UnidentifiedEffect(
        UnidentifiedReason.NOT_ANCESTRAL
    )


def test_a_cause_of_the_effect_only_through_a_hidden_confounder_is_no_possible_cause():
    """
    A cause of the effect only through a hidden confounder is no possible cause.
    """
    graph = pure_hidden_confounding().partial_ancestral_graph()

    assert graph.node("X") not in graph.possible_ancestors(graph.node("Y"))


# %% the criterion of each kind of graph


def test_each_kind_of_graph_is_checked_by_its_own_criterion():
    """
    Each kind of graph is checked by its own criterion.
    """
    ancestral = declared_confounder_of_covariates()
    causal = confounded()

    assert [
        type(
            AdjustmentCriterion.from_graph(
                graph=graph, cause=graph.node("X"), effect=graph.node("Y")
            )
        )
        for graph in (ancestral, causal)
    ] == [GeneralizedAdjustmentCriterion, MemberGraphAdjustmentCriterion]


# %% declared adjustment sets


@pytest.mark.parametrize(
    "names, reason",
    [
        ((), InvalidAdjustmentReason.LEAVES_NON_CAUSAL_PATH_OPEN),
        (("V",), InvalidAdjustmentReason.LEAVES_NON_CAUSAL_PATH_OPEN),
        (("Y",), InvalidAdjustmentReason.CONTAINS_CAUSE_OR_EFFECT),
    ],
)
def test_a_declared_set_that_fails_the_criterion_is_refused_with_its_reason(
    names: tuple[str, ...], reason: InvalidAdjustmentReason
):
    """
    A declared set that fails the criterion is refused with its reason.
    """
    graph = measured_confounder().partial_ancestral_graph()

    with pytest.raises(InvalidAdjustmentSet) as refusal:
        InvalidDeclaredAdjustmentRefusal().identify(
            AdjustmentSelector(graph),
            graph.node("X"),
            graph.node("Y"),
            adjustment_of(*names),
            set(graph.nodes()),
        )

    assert (refusal.value.reason, refusal.value.member_index) == (reason, None)


@pytest.mark.parametrize("names", [("C",), ("C", "V")])
def test_a_valid_declared_set_is_kept(names: tuple[str, ...]):
    """
    A valid declared set is kept.
    """
    graph = measured_confounder().partial_ancestral_graph()

    assert AdjustmentSelector(graph).check_declared(
        graph.node("X"), graph.node("Y"), adjustment_of(*names)
    ) == IdentifiedEffect(adjustment_of(*names))


def test_repair_keeps_the_declared_nodes_and_adds_the_confounder():
    """
    Repair keeps the declared nodes and adds the confounder.
    """
    graph = measured_confounder().partial_ancestral_graph()

    assert InvalidDeclaredAdjustmentRepair().identify(
        AdjustmentSelector(graph),
        graph.node("X"),
        graph.node("Y"),
        adjustment_of("V"),
        set(graph.nodes()),
    ) == IdentifiedEffect(adjustment_of("C", "V"))


def test_a_declared_set_in_a_graph_that_identifies_nothing_is_refused_as_such():
    """
    A declared set in a graph that identifies nothing is refused as such.
    """
    graph = hidden_cause_and_effect_confounder().partial_ancestral_graph()

    with pytest.raises(InvalidAdjustmentSet) as refusal:
        AdjustmentSelector(graph).check_declared(
            graph.node("X"), graph.node("Y"), adjustment_of("W")
        )

    assert refusal.value.reason is InvalidAdjustmentReason.NOT_AMENABLE


def test_repair_in_a_graph_that_identifies_nothing_refuses():
    """
    Repair in a graph that identifies nothing refuses.
    """
    graph = hidden_cause_and_effect_confounder().partial_ancestral_graph()

    identification = InvalidDeclaredAdjustmentRepair().identify(
        AdjustmentSelector(graph),
        graph.node("X"),
        graph.node("Y"),
        adjustment_of("W"),
        set(graph.nodes()),
    )

    assert identification.reason is UnidentifiedReason.NOT_AMENABLE


# %% soundness against the true graph


def graphs_standing_for(truth: TrueGraph) -> list[AncestralGraph]:
    """
    :param truth: A directed acyclic graph with hidden nodes.
    :return: Its maximal ancestral graph, and the partial ancestral graph causal-learn
        derives from it where that agrees with the maximal one.
    """
    maximal = truth.maximal_ancestral_graph()
    partial = truth.partial_ancestral_graph()
    return [maximal] + (
        [partial] if agrees_with(partial=partial, maximal=maximal) else []
    )


def test_every_identified_set_is_valid_in_the_true_graph():
    """
    Every identified set is valid in the true graph.
    """
    generator = np.random.default_rng(0)
    started = time.monotonic()
    identified, invalid = 0, []
    for _ in range(300):
        truth = random_true_graph(generator=generator)
        true_graph = truth.causal_graph()
        [member] = true_graph.member_graphs()
        for graph in graphs_standing_for(truth):
            for cause, effect in itertools.permutations(graph.nodes(), 2):
                identification = AdjustmentSelector(graph).identify(
                    cause, effect, set(graph.nodes())
                )
                if not identification.is_identified:
                    continue
                identified += 1
                reason = AdjustmentCheck(
                    member, true_graph.node(cause.name), true_graph.node(effect.name)
                ).failure_reason(identification.adjustment)
                if reason is not None:
                    invalid.append((truth, cause.name, effect.name, reason))

    assert (invalid, identified > 0, time.monotonic() - started < 60) == (
        [],
        True,
        True,
    )


def test_some_valid_set_exists_exactly_when_the_constructive_set_is_valid():
    """
    Some valid set exists exactly when the constructive set is valid.
    """
    generator = np.random.default_rng(1)
    for _ in range(150):
        graph = random_true_graph(generator=generator).partial_ancestral_graph()
        for cause, effect in itertools.permutations(graph.nodes(), 2):
            criterion = GeneralizedAdjustmentCriterion(
                graph=graph, cause=cause, effect=effect
            )
            if criterion.precondition_failure is not None:
                continue
            pool = sorted(set(graph.nodes()) - {cause, effect})
            some_set_is_valid = any(
                criterion.failure_of(adjustment=AdjustmentSet(frozenset(nodes))) is None
                for size in range(len(pool) + 1)
                for nodes in itertools.combinations(pool, size)
            )
            constructive = (
                (graph.possible_ancestors(cause) | graph.possible_ancestors(effect))
                - {cause, effect}
                - criterion.forbidden_nodes
            )

            assert some_set_is_valid == (
                criterion.failure_of(adjustment=AdjustmentSet(frozenset(constructive)))
                is None
            )
