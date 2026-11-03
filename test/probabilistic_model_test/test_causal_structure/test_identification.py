import pandas as pd
import pytest

from probabilistic_model.causal_structure.exceptions import (
    InvalidAdjustmentSet,
)
from probabilistic_model.causal_structure.graph import (
    CausalGraph,
)
from probabilistic_model.causal_structure.identification import (
    AdjustmentCheck,
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


def failure_of(
    graph: CausalGraph, adjustment: set[str]
) -> InvalidAdjustmentReason | None:
    """
    :param graph: A graph without undirected edges.
    :param adjustment: The adjustment set to check for the effect of X on Y.
    :return: Why the set fails the adjustment criterion, or ``None`` if it passes.
    """
    [member] = graph.member_graphs()
    return AdjustmentCheck(member, "X", "Y").failure_reason(frozenset(adjustment))


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
    identification = AdjustmentSelector(confounded()).identify("X", "Y", {"Z"})

    assert identification == IdentifiedEffect(frozenset({"Z"}))


def test_proxied_confounder_is_chosen_over_its_proxy():
    identification = AdjustmentSelector(proxied_confounder()).identify(
        "X", "Y", {"N", "E"}
    )

    assert identification == IdentifiedEffect(frozenset({"N"}))


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

    identification = AdjustmentSelector(two_step_back_door()).identify(
        "X", "Y", {"A", "B"}, rows
    )

    assert identification == IdentifiedEffect(frozenset({"B"}))


# %% an equivalence class that does not settle the effect


def test_undirected_edge_gives_partial_identification():
    identification = AdjustmentSelector(undirected_parent()).identify("X", "Y", {"A"})

    assert identification == PartiallyIdentifiedEffect(
        candidate_adjustments=[frozenset(), frozenset({"A"})],
        includes_member_without_causal_path=False,
    )


def test_effect_may_be_parent_of_cause():
    identification = AdjustmentSelector(undirected_cause_and_effect()).identify(
        "X", "Y", set()
    )

    assert identification == PartiallyIdentifiedEffect(
        candidate_adjustments=[frozenset()],
        includes_member_without_causal_path=True,
    )


def test_graph_without_member_graphs_leaves_the_effect_unidentified():
    identification = AdjustmentSelector(without_member_graphs()).identify(
        "A", "B", {"C", "D"}
    )

    assert identification == UnidentifiedEffect(UnidentifiedReason.NO_MEMBER_GRAPH)


# %% checking a declared adjustment set


def test_declared_mediator_is_refused():
    with pytest.raises(InvalidAdjustmentSet) as refusal:
        AdjustmentSelector(mediated()).check_declared("X", "Y", frozenset({"M"}))

    assert refusal.value.reason is InvalidAdjustmentReason.CONTAINS_FORBIDDEN_NODE


def test_declared_confounder_is_accepted():
    identification = AdjustmentSelector(confounded()).check_declared(
        "X", "Y", frozenset({"Z"})
    )

    assert identification == IdentifiedEffect(frozenset({"Z"}))
