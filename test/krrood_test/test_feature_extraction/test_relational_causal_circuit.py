"""
Tests for :mod:`probabilistic_model.probabilistic_circuit.relational.causal`: the bridge
from ``RelationalProbabilisticCircuit`` grounding to ``CausalCircuit`` construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from krrood.entity_query_language.factories import variable
from krrood.ormatic.data_access_objects.helper import to_dao
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.causal import (
    RelationalCausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    AmbiguousVariablePathError,
    VariableNotFoundError,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    GroundingMode,
    RelationalProbabilisticCircuit,
)
from random_events.product_algebra import SimpleEvent
from ..dataset.example_classes import SceneRoom, SceneRoomAggregations
from .test_rspns import (  # noqa: F401
    _room_with_chair_count,
    correlated_room_query,
    relational_probabilistic_circuit,
    room_query_4,
    scenario,
)

# %% resolve_variable


def test_resolve_variable_matches_the_full_name(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    variable = RelationalCausalCircuit.resolve_variable(
        grounded, "SceneRoomAggregations.chair_count()"
    )
    assert variable.name == "SceneRoomAggregations.chair_count()"


def test_resolve_variable_matches_an_unambiguous_suffix(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    variable = RelationalCausalCircuit.resolve_variable(grounded, "chair_count()")
    assert variable.name == "SceneRoomAggregations.chair_count()"


def test_resolve_variable_matches_a_relational_index_suffix(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    variable = RelationalCausalCircuit.resolve_variable(grounded, "objects[2].type")
    assert variable.name == "SceneRoom.objects[2].type"


def test_resolve_variable_raises_for_no_match(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    with pytest.raises(VariableNotFoundError):
        RelationalCausalCircuit.resolve_variable(grounded, "not_a_real_variable")


def test_resolve_variable_raises_for_ambiguous_suffix(
    relational_probabilistic_circuit, room_query_4
):
    """
    ``type`` alone matches every ``objects[i].type``, so it must be rejected rather than
    silently picking one.
    """
    np.random.seed(0)
    grounded = relational_probabilistic_circuit.ground(
        room_query_4, grounding_mode=GroundingMode.SAMPLED
    )
    with pytest.raises(AmbiguousVariablePathError):
        RelationalCausalCircuit.resolve_variable(grounded, "type")


# %% RelationalCausalCircuit.ground


def _resolve_against_a_fresh_grounding(
    relational_probabilistic_circuit, query, paths, grounding_mode
):
    """
    Resolve each of ``paths`` against a circuit grounded just for that purpose.

    ``RelationalCausalCircuit.ground`` requires already-resolved Variables (see its
    docstring), since it accepts no path strings itself; a caller who only has names
    grounds once to resolve them, the way this helper does, before grounding again for
    the ``CausalCircuit`` it actually wants.

    :param relational_probabilistic_circuit: The relational circuit to ground.
    :param query: The grounding query.
    :param paths: Dotted access-path strings to resolve.
    :param grounding_mode: The grounding mode to resolve under.
    :return: The resolved Variables, in input order.
    """
    grounded = relational_probabilistic_circuit.ground(
        query, grounding_mode=grounding_mode
    )
    return [RelationalCausalCircuit.resolve_variable(grounded, path) for path in paths]


def test_relational_causal_circuit_ground_returns_a_causal_circuit(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    [chair_count_variable, object_type_variable] = _resolve_against_a_fresh_grounding(
        relational_probabilistic_circuit,
        room_query_4,
        ["chair_count()", "objects[0].type"],
        GroundingMode.SAMPLED,
    )

    np.random.seed(0)
    causal_circuit = RelationalCausalCircuit().ground(
        relational_probabilistic_circuit,
        room_query_4,
        causal_variables=[chair_count_variable],
        effect_variables=[object_type_variable],
    )
    assert isinstance(causal_circuit, CausalCircuit)


def test_relational_causal_circuit_ground_defaults_to_causal_sampled(
    relational_probabilistic_circuit, room_query_4
):
    """
    The default grounding mode must retain undetermined latents, since that is the
    entire point of grounding a CausalCircuit this way.
    """
    np.random.seed(0)
    [chair_count_variable, object_type_variable] = _resolve_against_a_fresh_grounding(
        relational_probabilistic_circuit,
        room_query_4,
        ["chair_count()", "objects[0].type"],
        GroundingMode.SAMPLED,
    )

    np.random.seed(0)
    causal_circuit = RelationalCausalCircuit().ground(
        relational_probabilistic_circuit,
        room_query_4,
        causal_variables=[chair_count_variable],
        effect_variables=[object_type_variable],
    )
    names = {v.name for v in causal_circuit.probabilistic_circuit.variables}
    assert "SceneRoomAggregations.chair_count()" in names


def test_relational_causal_circuit_ground_backdoor_adjustment_runs(
    relational_probabilistic_circuit, room_query_4
):
    np.random.seed(0)
    [chair_count_variable, object_type_variable] = _resolve_against_a_fresh_grounding(
        relational_probabilistic_circuit,
        room_query_4,
        ["chair_count()", "objects[0].type"],
        GroundingMode.SAMPLED,
    )

    np.random.seed(0)
    causal_circuit = RelationalCausalCircuit().ground(
        relational_probabilistic_circuit,
        room_query_4,
        causal_variables=[chair_count_variable],
        effect_variables=[object_type_variable],
    )
    chair_count_variable = RelationalCausalCircuit.resolve_variable(
        causal_circuit.probabilistic_circuit, "chair_count()"
    )
    object_type_variable = RelationalCausalCircuit.resolve_variable(
        causal_circuit.probabilistic_circuit, "objects[0].type"
    )
    interventional_circuit = causal_circuit.backdoor_adjustment(
        cause_variable=chair_count_variable, effect_variable=object_type_variable
    )
    assert interventional_circuit.is_valid()


def test_relational_causal_circuit_ground_warns_on_expensive_adjustment_set(
    relational_probabilistic_circuit, room_query_4, caplog
):
    """
    Under GroundingMode.EXACT, registering more than one relational adjustment variable
    together must warn once their leaf-region product exceeds the configured threshold,
    rather than waiting to discover the cost at query time.
    """
    object_type_0, object_type_1, chair_count_variable, table_count_variable = (
        _resolve_against_a_fresh_grounding(
            relational_probabilistic_circuit,
            room_query_4,
            [
                "objects[0].type",
                "objects[1].type",
                "chair_count()",
                "table_count()",
            ],
            GroundingMode.EXACT,
        )
    )

    with caplog.at_level("WARNING"):
        RelationalCausalCircuit(adjustment_region_count_warning_threshold=0).ground(
            relational_probabilistic_circuit,
            room_query_4,
            causal_variables=[object_type_0],
            effect_variables=[object_type_1],
            adjustment_variables=[chair_count_variable, table_count_variable],
            grounding_mode=GroundingMode.EXACT,
        )
    assert any("leaf regions" in message for message in caplog.messages)


def test_relational_causal_circuit_ground_does_not_warn_below_threshold(
    relational_probabilistic_circuit, room_query_4, caplog
):
    object_type_0, object_type_1, chair_count_variable, table_count_variable = (
        _resolve_against_a_fresh_grounding(
            relational_probabilistic_circuit,
            room_query_4,
            [
                "objects[0].type",
                "objects[1].type",
                "chair_count()",
                "table_count()",
            ],
            GroundingMode.EXACT,
        )
    )

    with caplog.at_level("WARNING"):
        RelationalCausalCircuit(
            adjustment_region_count_warning_threshold=10_000
        ).ground(
            relational_probabilistic_circuit,
            room_query_4,
            causal_variables=[object_type_0],
            effect_variables=[object_type_1],
            adjustment_variables=[chair_count_variable, table_count_variable],
            grounding_mode=GroundingMode.EXACT,
        )
    assert not any("leaf regions" in message for message in caplog.messages)


# %% RelationalCausalCircuit.fit


@pytest.fixture
def many_chair_count_rooms():
    """
    Two large chair-count partitions (20 rooms each), with each room's continuous
    position varied -- enough rows and variance for JointProbabilityTree to split each
    partition further on its own, the precondition the regression test below needs.
    """
    rng = np.random.default_rng(0)
    return [_room_with_chair_count(rng, 1) for _ in range(20)] + [
        _room_with_chair_count(rng, 3) for _ in range(20)
    ]


def test_fit_stratifies_the_class_circuit_by_the_given_variable(many_chair_count_rooms):
    model = RelationalProbabilisticCircuit(SceneRoom)
    chair_count_variable = variable(SceneRoomAggregations).chair_count()
    RelationalCausalCircuit().fit(
        model,
        [to_dao(room) for room in many_chair_count_rooms],
        stratify_by=chair_count_variable._name_,
    )
    resolved_chair_count = next(
        v
        for v in model.class_probabilistic_circuit.variables
        if v.name == "SceneRoomAggregations.chair_count()"
    )
    for value in (1, 3):
        event = SimpleEvent.from_data({resolved_chair_count: value}).as_composite_set()
        probability = model.class_probabilistic_circuit.probability(
            event.fill_missing_variables_pure(
                model.class_probabilistic_circuit.variables
            )
        )
        assert probability == pytest.approx(0.5, abs=0.01)


def test_verify_support_determinism_survives_a_stratified_partitions_own_further_splits(
    many_chair_count_rooms, correlated_room_query
):
    """
    Regression test: a stratified partition's own further JointProbabilityTree splits on
    unrelated variables (here, room position) used to make
    ``verify_support_determinism`` fail even though the stratification itself was
    correct.

    ``CausalCircuit._check_support_disjointness`` used to compute each variable's
    marginal via ``ProbabilisticCircuit.marginal``, whose trailing
    ``SumUnit.simplify()`` flattens nested SumUnits together, merging a partition's own
    further-split branches into siblings of a different partition's branches and erasing
    which stratified value each one actually belonged to. Grounding also used to trigger
    the same flattening on the class circuit itself, via ``_condition_class_circuit``
    unconditionally calling ``log_conditional_in_place`` even when there was nothing to
    condition on.
    """
    model = RelationalProbabilisticCircuit(SceneRoom)
    chair_count_variable = variable(SceneRoomAggregations).chair_count()
    relational_causal_circuit = RelationalCausalCircuit()
    relational_causal_circuit.fit(
        model,
        [to_dao(room) for room in many_chair_count_rooms],
        stratify_by=chair_count_variable._name_,
    )

    np.random.seed(0)
    grounded = model.ground(correlated_room_query, grounding_mode=GroundingMode.SAMPLED)
    resolved_chair_count = next(
        v for v in grounded.variables if v.name == "SceneRoomAggregations.chair_count()"
    )
    object_type_variable = next(
        v for v in grounded.variables if v.name == "SceneRoom.objects[0].type"
    )

    causal_circuit = relational_causal_circuit.from_grounded_circuit(
        grounded,
        causal_variables=[resolved_chair_count],
        effect_variables=[object_type_variable],
        trim_to_registered_variables=True,
    )
    assert isinstance(causal_circuit, CausalCircuit)
