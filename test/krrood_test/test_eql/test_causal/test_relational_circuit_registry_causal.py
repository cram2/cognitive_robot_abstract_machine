"""
Tests for ``RelationalCircuitRegistry``'s causal query support: registering relational
grounded variables (retained via ``GroundingMode.CAUSAL_SAMPLED``/``CAUSAL_EXACT``) as
causes or effects through the same ``cause``/``causes_effect`` EQL machinery
``CausalCircuitRegistry`` already supports for static, non-relational circuits.
"""

from __future__ import annotations

import numpy as np

from krrood.entity_query_language.factories import a, cause
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from ...dataset import ormatic_interface  # type: ignore
from ...dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneObjectType,
    SceneRoom,
)


def _fitted_scene_room_model() -> RelationalProbabilisticCircuit:
    objects = [
        SceneObject(type=SceneObjectType.TABLE),
        SceneObject(type=SceneObjectType.CHAIR),
        SceneObject(type=SceneObjectType.CHAIR),
        SceneObject(type=SceneObjectType.CHAIR),
    ]
    room = SceneRoom(
        position=KRROODPosition(x=2.0, y=1.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=objects[:3],
    )
    room2 = SceneRoom(
        position=KRROODPosition(x=4.0, y=3.0, z=0.0),
        orientation=KRROODOrientation(x=0.0, y=0.0, z=0.0, w=1.0),
        objects=objects,
    )
    model = RelationalProbabilisticCircuit(SceneRoom)
    model.fit([to_dao(room), to_dao(room2)])
    return model


def _cause_and_effect_query():
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=cause, y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    query.causes_effect(query.variable.objects[0].type == SceneObjectType.CHAIR)
    return query


def test_registry_returns_a_causal_circuit_for_a_cause_query():
    """
    A relational Match query with cause/causes_effect markers must resolve to a
    CausalCircuit, not the plain grounded circuit DoRequiresCausalCircuitModel would
    otherwise reject.
    """
    model = _fitted_scene_room_model()
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=model)
    parameters = UnderspecifiedParameters(_cause_and_effect_query())

    np.random.seed(0)
    result = registry.get_model(parameters)

    assert isinstance(result, CausalCircuit)


def test_registered_causal_circuit_has_the_queried_cause_and_effect_variables():
    model = _fitted_scene_room_model()
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=model)
    parameters = UnderspecifiedParameters(_cause_and_effect_query())

    np.random.seed(0)
    result = registry.get_model(parameters)

    assert [v.name for v in result.causal_variables] == ["SceneRoom.position.x"]
    assert [v.name for v in result.effect_variables] == ["SceneRoom.objects[0].type"]


def test_registered_causal_circuit_supports_backdoor_adjustment():
    model = _fitted_scene_room_model()
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=model)
    parameters = UnderspecifiedParameters(_cause_and_effect_query())

    np.random.seed(0)
    result = registry.get_model(parameters)

    interventional_circuit = result.backdoor_adjustment(
        cause_variable=result.causal_variables[0],
        effect_variable=result.effect_variables[0],
    )
    assert interventional_circuit.is_valid()


def test_non_causal_query_is_unaffected():
    """
    A plain (non-cause) relational query must still return the grounded circuit
    directly, exactly as before this registry gained causal support.
    """
    model = _fitted_scene_room_model()
    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=model)
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    parameters = UnderspecifiedParameters(query)

    result = registry.get_model(parameters)

    assert not isinstance(result, CausalCircuit)
