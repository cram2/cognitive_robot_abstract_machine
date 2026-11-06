import pandas as pd
import pytest

from probabilistic_model.causal_structure.exceptions import (
    InvalidAdjustmentSet,
)
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
    GraphNode,
)
from probabilistic_model.causal_structure.identification import (
    AdjustmentCheck,
    AdjustmentSet,
    AdjustmentSelector,
    IdentifiedEffect,
    InvalidAdjustmentReason,
    PartiallyIdentifiedEffect,
    UnidentifiedEffect,
    UnidentifiedReason,
)

from .graphs import (
    cause_with_a_child,
    confounded,
    effect_with_a_child,
    m_bias,
    mediated,
    proxied_confounder,
    two_step_back_door,
    undirected_cause_and_effect,
    undirected_parent,
    without_member_graphs,
)

# %% the adjustment criterion in one graph


def nodes_of(graph: CausalGraph, names: set[str]) -> frozenset[GraphNode]:
    """
    :param graph: A graph.
    :param names: The names of some of its nodes.
    :return: Those nodes.
    """
    return frozenset(graph.node(name) for name in names)


def adjustment_of(graph: CausalGraph, names: set[str]) -> AdjustmentSet:
    """
    :param graph: A graph.
    :param names: The names of some of its nodes.
    :return: The set adjusting for those nodes.
    """
    return AdjustmentSet(nodes_of(graph, names))


def identify(
    graph: CausalGraph,
    eligible: set[str],
    rows: pd.DataFrame | None = None,
    cause: str = "X",
    effect: str = "Y",
):
    """
    :param graph: A graph.
    :param eligible: The names of the nodes that may be adjusted for.
    :param rows: Training rows to rank the valid sets by.
    :param cause: The name of the cause.
    :param effect: The name of the effect.
    :return: What the graph says about the effect of the cause on the effect.
    """
    return AdjustmentSelector(graph).identify(
        graph.node(cause), graph.node(effect), set(nodes_of(graph, eligible)), rows
    )


def failure_of(
    graph: CausalGraph, adjustment: set[str]
) -> InvalidAdjustmentReason | None:
    """
    :param graph: A graph without undirected edges.
    :param adjustment: The adjustment set to check for the effect of X on Y.
    :return: Why the set fails the adjustment criterion, or ``None`` if it passes.
    """
    [member] = graph.member_graphs()
    return AdjustmentCheck(member, graph.node("X"), graph.node("Y")).failure_reason(
        adjustment_of(graph, adjustment)
    )


@pytest.mark.parametrize(
    "graph, adjustment, reason",
    [
        (confounded(), {"Z"}, None),
        (confounded(), set(), InvalidAdjustmentReason.LEAVES_NON_CAUSAL_PATH_OPEN),
        (mediated(), set(), None),
        (mediated(), {"M"}, InvalidAdjustmentReason.CONTAINS_FORBIDDEN_NODE),
        (cause_with_a_child(), {"C"}, None),
        (effect_with_a_child(), {"D"}, InvalidAdjustmentReason.CONTAINS_FORBIDDEN_NODE),
        (m_bias(), set(), None),
        (m_bias(), {"M"}, InvalidAdjustmentReason.LEAVES_NON_CAUSAL_PATH_OPEN),
        (m_bias(), {"M", "A"}, None),
        (
            proxied_confounder(),
            {"E"},
            InvalidAdjustmentReason.LEAVES_NON_CAUSAL_PATH_OPEN,
        ),
        (proxied_confounder(), {"N"}, None),
        (proxied_confounder(), {"N", "E"}, None),
        (confounded(), {"X", "Z"}, InvalidAdjustmentReason.CONTAINS_CAUSE_OR_EFFECT),
    ],
    ids=[
        "confounder_is_valid",
        "confounded_without_adjustment_is_open",
        "mediated_needs_no_adjustment",
        "mediator_is_forbidden",
        "child_of_cause_off_the_causal_path_is_valid",
        "child_of_effect_is_forbidden",
        "m_bias_needs_no_adjustment",
        "m_bias_collider_opens_a_path",
        "m_bias_collider_with_its_parent_is_valid",
        "proxy_alone_is_not_valid",
        "confounder_behind_a_proxy_is_valid",
        "confounder_with_its_proxy_is_valid",
        "cause_is_not_an_adjustment",
    ],
)
def test_adjustment_criterion(
    graph: CausalGraph, adjustment: set[str], reason: InvalidAdjustmentReason | None
):
    assert failure_of(graph, adjustment) is reason


# %% choosing an adjustment set


def test_confounder_is_the_adjustment_set():
    graph = confounded()

    assert identify(graph, {"Z"}) == IdentifiedEffect(adjustment_of(graph, {"Z"}))


def test_proxied_confounder_is_chosen_over_its_proxy():
    graph = proxied_confounder()

    assert identify(graph, {"N", "E"}) == IdentifiedEffect(adjustment_of(graph, {"N"}))


def test_support_ranking_prefers_populated_cells():
    """
    A and B each block the one back-door path.

    A splits the rows into twelve cells of five rows each with the cause, B into four of
    fifteen.
    """
    rows = pd.DataFrame(
        {
            "X": [position % 2 for position in range(60)],
            "A": [(position // 2) % 6 for position in range(60)],
            "B": [(position // 4) % 2 for position in range(60)],
            "Y": [0] * 60,
        }
    )

    graph = two_step_back_door()

    assert identify(graph, {"A", "B"}, rows) == IdentifiedEffect(
        adjustment_of(graph, {"B"})
    )


# %% an equivalence class that does not settle the effect


def test_undirected_edge_gives_partial_identification():
    graph = undirected_parent()

    assert identify(graph, {"A"}) == PartiallyIdentifiedEffect(
        parent_sets=[AdjustmentSet(), adjustment_of(graph, {"A"})],
        includes_member_without_causal_path=False,
    )


def test_effect_may_be_parent_of_cause():
    assert identify(undirected_cause_and_effect(), set()) == PartiallyIdentifiedEffect(
        parent_sets=[AdjustmentSet()],
        includes_member_without_causal_path=True,
    )


def test_graph_without_member_graphs_leaves_the_effect_unidentified():
    identification = identify(
        without_member_graphs(), {"C", "D"}, cause="A", effect="B"
    )

    assert identification == UnidentifiedEffect(UnidentifiedReason.NO_MEMBER_GRAPH)


# %% what an identification offers to adjust for


def test_identified_effect_offers_its_one_set():
    graph = confounded()
    adjustment = adjustment_of(graph, {"Z"})
    identification = IdentifiedEffect(adjustment)

    assert (
        identification.candidate_adjustments,
        identification.is_identified,
        identification.includes_no_effect,
    ) == ([adjustment], True, False)


def test_partially_identified_effect_offers_every_parent_set():
    identification = identify(undirected_cause_and_effect(), set())

    assert (
        identification.candidate_adjustments,
        identification.is_identified,
        identification.includes_no_effect,
    ) == (
        identification.parent_sets,
        False,
        identification.includes_member_without_causal_path,
    )


def test_unidentified_effect_offers_nothing():
    identification = UnidentifiedEffect(UnidentifiedReason.NO_MEMBER_GRAPH)

    assert (
        identification.candidate_adjustments,
        identification.is_identified,
        identification.includes_no_effect,
    ) == ([], False, False)


# %% checking a declared adjustment set


def test_declared_mediator_is_refused():
    graph = mediated()

    with pytest.raises(InvalidAdjustmentSet) as refusal:
        AdjustmentSelector(graph).check_declared(
            graph.node("X"), graph.node("Y"), adjustment_of(graph, {"M"})
        )

    assert refusal.value.reason is InvalidAdjustmentReason.CONTAINS_FORBIDDEN_NODE


def test_declared_confounder_is_accepted():
    graph = confounded()

    identification = AdjustmentSelector(graph).check_declared(
        graph.node("X"), graph.node("Y"), adjustment_of(graph, {"Z"})
    )

    assert identification == IdentifiedEffect(adjustment_of(graph, {"Z"}))


# %% ordering adjustment sets


def test_smaller_adjustment_sets_come_first_then_by_their_names():
    graph = confounded()
    by_size_then_name = [
        AdjustmentSet(),
        adjustment_of(graph, {"Y"}),
        adjustment_of(graph, {"X", "Z"}),
    ]
    assert sorted(reversed(by_size_then_name)) == by_size_then_name
