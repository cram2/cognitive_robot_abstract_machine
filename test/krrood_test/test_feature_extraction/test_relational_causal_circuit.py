"""
Tests for :mod:`probabilistic_model.probabilistic_circuit.relational.causal`: the bridge
from ``RelationalProbabilisticCircuit`` grounding to ``CausalCircuit`` construction.
"""

from __future__ import annotations

import numpy as np
import pytest

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
from probabilistic_model.probabilistic_circuit.relational.rspn import GroundingMode
from .test_rspns import rpc, room_query_4, scenario  # noqa: F401

# %% resolve_variable


def test_resolve_variable_matches_the_full_name(rpc, room_query_4):
    np.random.seed(0)
    grounded = rpc.ground(room_query_4, grounding_mode=GroundingMode.SAMPLED)
    variable = RelationalCausalCircuit.resolve_variable(
        grounded, "SceneRoomAggregations.chair_count()"
    )
    assert variable.name == "SceneRoomAggregations.chair_count()"


def test_resolve_variable_matches_an_unambiguous_suffix(rpc, room_query_4):
    np.random.seed(0)
    grounded = rpc.ground(room_query_4, grounding_mode=GroundingMode.SAMPLED)
    variable = RelationalCausalCircuit.resolve_variable(grounded, "chair_count()")
    assert variable.name == "SceneRoomAggregations.chair_count()"


def test_resolve_variable_matches_a_relational_index_suffix(rpc, room_query_4):
    np.random.seed(0)
    grounded = rpc.ground(room_query_4, grounding_mode=GroundingMode.SAMPLED)
    variable = RelationalCausalCircuit.resolve_variable(grounded, "objects[2].type")
    assert variable.name == "SceneRoom.objects[2].type"


def test_resolve_variable_raises_for_no_match(rpc, room_query_4):
    np.random.seed(0)
    grounded = rpc.ground(room_query_4, grounding_mode=GroundingMode.SAMPLED)
    with pytest.raises(VariableNotFoundError):
        RelationalCausalCircuit.resolve_variable(grounded, "not_a_real_variable")


def test_resolve_variable_raises_for_ambiguous_suffix(rpc, room_query_4):
    """
    ``type`` alone matches every ``objects[i].type``, so it must be rejected rather than
    silently picking one.
    """
    np.random.seed(0)
    grounded = rpc.ground(room_query_4, grounding_mode=GroundingMode.SAMPLED)
    with pytest.raises(AmbiguousVariablePathError):
        RelationalCausalCircuit.resolve_variable(grounded, "type")


# %% RelationalCausalCircuit.ground


def test_relational_causal_circuit_ground_returns_a_causal_circuit(rpc, room_query_4):
    np.random.seed(0)
    causal_circuit = RelationalCausalCircuit.ground(
        rpc,
        room_query_4,
        causal_variables=["chair_count()"],
        effect_variables=["objects[0].type"],
    )
    assert isinstance(causal_circuit, CausalCircuit)


def test_relational_causal_circuit_ground_accepts_resolved_variables(rpc, room_query_4):
    """
    Callers may pass already-resolved Variable objects instead of path strings.
    """
    np.random.seed(0)
    grounded = rpc.ground(room_query_4, grounding_mode=GroundingMode.SAMPLED)
    chair_count_variable = RelationalCausalCircuit.resolve_variable(
        grounded, "chair_count()"
    )
    object_type_variable = RelationalCausalCircuit.resolve_variable(
        grounded, "objects[0].type"
    )

    np.random.seed(0)
    causal_circuit = RelationalCausalCircuit.ground(
        rpc,
        room_query_4,
        causal_variables=[chair_count_variable],
        effect_variables=[object_type_variable],
    )
    assert isinstance(causal_circuit, CausalCircuit)


def test_relational_causal_circuit_ground_defaults_to_causal_sampled(rpc, room_query_4):
    """
    The default grounding mode must retain undetermined latents, since that is the
    entire point of grounding a CausalCircuit this way.
    """
    np.random.seed(0)
    causal_circuit = RelationalCausalCircuit.ground(
        rpc,
        room_query_4,
        causal_variables=["chair_count()"],
        effect_variables=["objects[0].type"],
    )
    names = {v.name for v in causal_circuit.probabilistic_circuit.variables}
    assert "SceneRoomAggregations.chair_count()" in names


def test_relational_causal_circuit_ground_backdoor_adjustment_runs(rpc, room_query_4):
    np.random.seed(0)
    causal_circuit = RelationalCausalCircuit.ground(
        rpc,
        room_query_4,
        causal_variables=["chair_count()"],
        effect_variables=["objects[0].type"],
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
    rpc, room_query_4, caplog
):
    """
    Under GroundingMode.EXACT, registering more than one relational adjustment variable
    together must warn once their leaf-region product exceeds the configured threshold,
    rather than waiting to discover the cost at query time.
    """
    with caplog.at_level("WARNING"):
        RelationalCausalCircuit.ground(
            rpc,
            room_query_4,
            causal_variables=["objects[0].type"],
            effect_variables=["objects[1].type"],
            adjustment_variables=["chair_count()", "table_count()"],
            grounding_mode=GroundingMode.EXACT,
            adjustment_region_count_warning_threshold=0,
        )
    assert any("leaf regions" in message for message in caplog.messages)


def test_relational_causal_circuit_ground_does_not_warn_below_threshold(
    rpc, room_query_4, caplog
):
    with caplog.at_level("WARNING"):
        RelationalCausalCircuit.ground(
            rpc,
            room_query_4,
            causal_variables=["objects[0].type"],
            effect_variables=["objects[1].type"],
            adjustment_variables=["chair_count()", "table_count()"],
            grounding_mode=GroundingMode.EXACT,
            adjustment_region_count_warning_threshold=10_000,
        )
    assert not any("leaf regions" in message for message in caplog.messages)
