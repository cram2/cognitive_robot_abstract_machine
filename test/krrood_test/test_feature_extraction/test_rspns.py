import numpy as np
import pytest

from krrood.entity_query_language.factories import underspecified
from krrood.ormatic.data_access_objects.helper import to_dao
from probabilistic_model.probabilistic_circuit.relational.exceptions import (
    CircuitNotFittedError,
    InvalidMonteCarloSampleCountError,
)
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import SumUnit
from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import (
    KRROODOrientation,
    KRROODPosition,
    SceneObject,
    SceneObjectType,
    SceneRoom,
)


@pytest.fixture
def scenario():
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
    return to_dao(room), to_dao(room2)


@pytest.fixture
def rpc(scenario):
    room_dao, room2_dao = scenario
    model = RelationalProbabilisticCircuit(SceneRoom)
    model.fit([room_dao, room2_dao])
    return model


@pytest.fixture
def room_query_4():
    query = underspecified(SceneRoom)(
        position=underspecified(KRROODPosition)(x=..., y=..., z=...),
        orientation=underspecified(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[underspecified(SceneObject)(type=...) for _ in range(4)],
    )
    query.resolve()
    return query


def test_ground_before_fit_raises(room_query_4):
    model = RelationalProbabilisticCircuit(SceneRoom)
    with pytest.raises(CircuitNotFittedError):
        model.ground(room_query_4)


def test_fit_class_circuit_is_valid(rpc):
    assert rpc.class_probabilistic_circuit is not None
    assert rpc.class_probabilistic_circuit.is_valid()


def test_fit_class_circuit_has_room_scalar_variables(rpc):
    names = {v.name for v in rpc.class_probabilistic_circuit.variables}
    assert "SceneRoom.position.x" in names
    assert "SceneRoom.position.y" in names
    assert "SceneRoom.position.z" in names
    assert "SceneRoom.orientation.x" in names
    assert "SceneRoom.orientation.y" in names
    assert "SceneRoom.orientation.z" in names
    assert "SceneRoom.orientation.w" in names


def test_fit_class_circuit_has_aggregation_variable(rpc):
    names = {v.name for v in rpc.class_probabilistic_circuit.variables}
    assert "SceneRoomAggregations.total_count()" in names


def test_fit_creates_exchangeable_template_for_objects(rpc):
    assert "objects" in rpc.exchangeable_distribution_templates
    template = rpc.exchangeable_distribution_templates["objects"]
    assert template.template_distribution.class_probabilistic_circuit is not None


def test_fit_exchangeable_template_latent_is_total_count(rpc):
    template = rpc.exchangeable_distribution_templates["objects"]
    latent_names = {v.name for v in template.latent_variables}
    assert "SceneRoomAggregations.total_count()" in latent_names


def test_fit_exchangeable_template_models_object_type(rpc):
    template = rpc.exchangeable_distribution_templates["objects"]
    pc = template.template_distribution.class_probabilistic_circuit
    names = {v.name for v in pc.variables}
    assert "type" in names


def test_ground_circuit_is_valid(rpc, room_query_4):
    model = rpc.ground(room_query_4)
    assert model.is_valid()


def test_ground_has_per_object_type_variables(rpc, room_query_4):
    model = rpc.ground(room_query_4)
    names = {v.name for v in model.variables}
    for i in range(4):
        assert f"SceneRoom.objects[{i}].type" in names


def test_ground_preserves_room_scalar_variables(rpc, room_query_4):
    model = rpc.ground(room_query_4)
    names = {v.name for v in model.variables}
    assert "SceneRoom.position.x" in names
    assert "SceneRoom.orientation.w" in names


def test_ground_integrates_out_unavailable_aggregates(rpc, room_query_4):
    """``chair_count`` and ``table_count`` cannot be determined from the
    underspecified query, so the Monte-Carlo path must integrate them out: they
    must not survive as variables, while the object-type variables remain."""
    model = rpc.ground(room_query_4)
    names = {v.name for v in model.variables}
    assert "SceneRoomAggregations.chair_count()" not in names
    assert "SceneRoomAggregations.table_count()" not in names
    for i in range(4):
        assert f"SceneRoom.objects[{i}].type" in names


def test_ground_with_unavailable_aggregate_is_valid(rpc, room_query_4):
    np.random.seed(0)
    assert rpc.ground(room_query_4).is_valid()


def test_non_positive_sample_count_raises_when_integration_needed(rpc, room_query_4):
    """Monte-Carlo integration cannot be disabled: a non-positive sample count is
    rejected when undetermined aggregates must be integrated out."""
    rpc.monte_carlo_sample_count = 0
    with pytest.raises(InvalidMonteCarloSampleCountError):
        rpc.ground(room_query_4)


def test_monte_carlo_sample_count_controls_mixture_size(rpc, room_query_4):
    """Drawing more samples discovers more distinct aggregate values, each adding
    an exchangeable-distribution instance (and its sum units) to the mixture."""
    np.random.seed(0)
    rpc.monte_carlo_sample_count = 1
    single = sum(1 for n in rpc.ground(room_query_4).nodes() if isinstance(n, SumUnit))
    np.random.seed(0)
    rpc.monte_carlo_sample_count = 50
    many = sum(1 for n in rpc.ground(room_query_4).nodes() if isinstance(n, SumUnit))
    assert many > single


def test_ground_variable_count_scales_with_query_size(rpc):
    query_2 = underspecified(SceneRoom)(
        position=underspecified(KRROODPosition)(x=..., y=..., z=...),
        orientation=underspecified(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[underspecified(SceneObject)(type=...) for _ in range(2)],
    )
    query_2.resolve()
    query_4 = underspecified(SceneRoom)(
        position=underspecified(KRROODPosition)(x=..., y=..., z=...),
        orientation=underspecified(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[underspecified(SceneObject)(type=...) for _ in range(4)],
    )
    query_4.resolve()
    assert len(rpc.ground(query_4).variables) > len(rpc.ground(query_2).variables)
