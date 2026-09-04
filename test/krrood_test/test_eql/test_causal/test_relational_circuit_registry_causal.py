"""
Tests for ``RelationalCircuitRegistry``'s causal query support: registering relational
grounded variables (retained via ``GroundingMode.SAMPLED``/``EXACT``) as causes or
effects through the same ``cause``/``causes_effect`` EQL machinery
``CausalCircuitRegistry`` already supports for static, non-relational circuits.
"""

from __future__ import annotations

import numpy as np

from krrood.entity_query_language.factories import a, cause
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import GroundingMode
from ...dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneObjectType,
    SceneRoom,
)
from ...test_feature_extraction.test_rspns import rpc, scenario  # noqa: F401


def _cause_and_effect_query():
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=cause, y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    query.causes_effect(query.variable.objects[0].type == SceneObjectType.CHAIR)
    return query


def test_registry_returns_a_causal_circuit_for_a_cause_query(rpc):
    """
    A relational Match query with cause/causes_effect markers must resolve to a
    CausalCircuit, not the plain grounded circuit DoRequiresCausalCircuitModel would
    otherwise reject.
    """
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=rpc)
    parameters = UnderspecifiedParameters(_cause_and_effect_query())

    np.random.seed(0)
    result = registry.get_model(parameters)

    assert isinstance(result, CausalCircuit)


def test_registered_causal_circuit_has_the_queried_cause_and_effect_variables(rpc):
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=rpc)
    parameters = UnderspecifiedParameters(_cause_and_effect_query())

    np.random.seed(0)
    result = registry.get_model(parameters)

    assert [v.name for v in result.causal_variables] == ["SceneRoom.position.x"]
    assert [v.name for v in result.effect_variables] == ["SceneRoom.objects[0].type"]


def test_registered_causal_circuit_supports_backdoor_adjustment(rpc):
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=rpc)
    parameters = UnderspecifiedParameters(_cause_and_effect_query())

    np.random.seed(0)
    result = registry.get_model(parameters)

    interventional_circuit = result.backdoor_adjustment(
        cause_variable=result.causal_variables[0],
        effect_variable=result.effect_variables[0],
    )
    assert interventional_circuit.is_valid()


def test_non_causal_query_is_unaffected(rpc):
    """
    A plain (non-cause) relational query must still return the grounded circuit
    directly, exactly as before this registry gained causal support.
    """
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=rpc)
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    parameters = UnderspecifiedParameters(query)

    result = registry.get_model(parameters)

    assert not isinstance(result, CausalCircuit)


def test_grounding_mode_field_is_actually_used(rpc):
    """
    Regression test for the ``grounding_mode`` field: overriding it to
    ``GroundingMode.EXACT`` must actually change grounding behaviour, not be silently
    ignored in favour of a hardcoded ``SAMPLED``.

    Both modes retain the latent as a variable and ground a valid circuit, but only
    exact-partition grounding (or its fallback, which still retains the latent) is
    reproducible across calls under different random state -- exactly what ``SAMPLED``
    alone is not guaranteed to be.
    """
    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=rpc,
        grounding_mode=GroundingMode.EXACT,
    )

    np.random.seed(0)
    first = registry.get_model(UnderspecifiedParameters(_cause_and_effect_query()))
    np.random.seed(123)
    second = registry.get_model(UnderspecifiedParameters(_cause_and_effect_query()))

    assert isinstance(first, CausalCircuit)
    assert {v.name for v in first.probabilistic_circuit.variables} == {
        v.name for v in second.probabilistic_circuit.variables
    }
