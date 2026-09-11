"""
Tests for ``RelationalCircuitRegistry``'s causal query support: registering relational
grounded variables (retained via ``GroundingMode.SAMPLED``/``EXACT``) as causes or
effects through the same ``cause``/``causes_effect`` EQL machinery
``CausalCircuitRegistry`` already supports for static, non-relational circuits.
"""

from __future__ import annotations

import numpy as np

from krrood.entity_query_language.factories import a, cause, variable
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.model_registries import RelationalCircuitRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.causal import (
    RelationalCausalCircuit,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    GroundingMode,
    RelationalProbabilisticCircuit,
)
from ...dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneObjectType,
    SceneRoom,
    SceneRoomAggregations,
)
from ...test_feature_extraction.test_rspns import (  # noqa: F401
    _room_with_chair_count,
    correlated_room_query,
    relational_probabilistic_circuit,
    scenario,
)


def _cause_and_effect_query():
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=cause, y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    query.causes_effect(query.variable.objects[0].type == SceneObjectType.CHAIR)
    return query


def test_registry_returns_a_causal_circuit_for_a_cause_query(
    relational_probabilistic_circuit,
):
    """
    A relational Match query with cause/causes_effect markers must resolve to a
    CausalCircuit, not the plain grounded circuit DoRequiresCausalCircuitModel would
    otherwise reject.
    """
    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=relational_probabilistic_circuit
    )
    parameters = UnderspecifiedParameters(_cause_and_effect_query())

    np.random.seed(0)
    result = registry.get_model(parameters)

    assert isinstance(result, CausalCircuit)


def test_registered_causal_circuit_has_the_queried_cause_and_effect_variables(
    relational_probabilistic_circuit,
):
    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=relational_probabilistic_circuit
    )
    parameters = UnderspecifiedParameters(_cause_and_effect_query())

    np.random.seed(0)
    result = registry.get_model(parameters)

    assert [v.name for v in result.causal_variables] == ["SceneRoom.position.x"]
    assert [v.name for v in result.effect_variables] == ["SceneRoom.objects[0].type"]


def test_registered_causal_circuit_supports_backdoor_adjustment(
    relational_probabilistic_circuit,
):
    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=relational_probabilistic_circuit
    )
    parameters = UnderspecifiedParameters(_cause_and_effect_query())

    np.random.seed(0)
    result = registry.get_model(parameters)

    interventional_circuit = result.backdoor_adjustment(
        cause_variable=result.causal_variables[0],
        effect_variable=result.effect_variables[0],
    )
    assert interventional_circuit.is_valid()


def test_non_causal_query_is_unaffected(relational_probabilistic_circuit):
    """
    A plain (non-cause) relational query must still return the grounded circuit
    directly, exactly as before this registry gained causal support.
    """
    registry = RelationalCircuitRegistry(
        relational_probabilistic_circuit=relational_probabilistic_circuit
    )
    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(4)],
    )
    parameters = UnderspecifiedParameters(query)

    result = registry.get_model(parameters)

    assert not isinstance(result, CausalCircuit)


def test_grounding_mode_field_is_actually_used(relational_probabilistic_circuit):
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
        relational_probabilistic_circuit=relational_probabilistic_circuit,
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


# %% cause/confounder on an aggregation statistic, through real relational grounding


def test_cause_on_an_aggregation_statistic_grounds_through_the_registry():
    """
    Regression test: marking an aggregation statistic (rather than a literal field)
    ``cause`` used to fail during grounding, in two different ways.

    ``Match.construct_instance`` used to pass every keyword argument straight to the
    domain class's constructor, including ``chair_count=cause`` -- not a real
    ``SceneRoom`` field, so construction raised a ``TypeError`` before grounding ever
    ran. And even once that was fixed, the variable name ``UnderspecifiedParameters``
    resolved for the marked keyword (``SceneRoom.chair_count``) did not match the name
    grounding actually gives that variable (``SceneRoomAggregations.chair_count()``), so
    ``verify_support_determinism`` rejected the registration for a variable it could not
    find on the circuit at all.
    """
    rng = np.random.default_rng(0)
    rooms = [_room_with_chair_count(rng, 1) for _ in range(20)] + [
        _room_with_chair_count(rng, 3) for _ in range(20)
    ]
    model = RelationalProbabilisticCircuit(SceneRoom)
    chair_count_variable = variable(SceneRoomAggregations).chair_count()
    RelationalCausalCircuit().fit(
        model,
        [to_dao(room) for room in rooms],
        stratify_by=chair_count_variable._name_,
    )

    query = a(SceneRoom)(
        position=a(KRROODPosition)(x=..., y=..., z=...),
        orientation=a(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[a(SceneObject)(type=...) for _ in range(3)],
        chair_count=cause,
    )
    query.causes_effect(query.variable.objects[0].type == SceneObjectType.CHAIR)

    registry = RelationalCircuitRegistry(relational_probabilistic_circuit=model)

    np.random.seed(0)
    result = registry.get_model(UnderspecifiedParameters(query))

    assert isinstance(result, CausalCircuit)
    assert [v.name for v in result.causal_variables] == [
        "SceneRoomAggregations.chair_count()"
    ]
