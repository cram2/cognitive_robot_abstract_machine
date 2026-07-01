import numpy as np
import pytest

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    underspecified,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extraction.feature_extractor import FeatureExtractor
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.relational.rspn import (
    RelationalProbabilisticCircuit,
)
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from random_events.variable import Symbolic
from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import (
    NestedAction,
    KRROODPose,
    KRROODPosition,
    KRROODOrientation,
    SceneRoom,
    SceneObject,
    SceneObjectType,
    MissingBaseClass,
    ExampleInt,
    ExampleString,
)
from ..dataset.semantic_world_like_classes import Body


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
    room_dao = to_dao(room)
    room2_dao = to_dao(room2)
    feature_extractor = FeatureExtractor.from_instances([room_dao])
    return room, room2, room_dao, room2_dao, feature_extractor


def test_features_extraction():
    action = underspecified(NestedAction)(
        pose=underspecified(KRROODPose)(
            position=underspecified(KRROODPosition)(x=2.0, y=..., z=...),
            orientation=underspecified(KRROODOrientation)(x=..., y=..., z=..., w=...),
        ),
        obj=Body(name="body"),
    )

    parameters = UnderspecifiedParameters(action)
    fully_factorized_circuit = fully_factorized(parameters.variables.values())
    assert len(parameters.truncation_assignments_from_krrood_variables) == 0

    probabilistic_registry = DictRegistry({NestedAction: fully_factorized_circuit})

    np.random.seed(69)
    backend = ProbabilisticBackend(probabilistic_registry, number_of_samples=50)
    samples = list(backend.evaluate(action))

    assert all(
        [sample.pose.position.x == samples[0].pose.position.x for sample in samples]
    )
    samples_to_daos = [to_dao(sample) for sample in samples]

    feature_extractor = FeatureExtractor.from_instances(samples_to_daos)
    dataframe = feature_extractor.create_dataframe(samples_to_daos)

    assert [
        dataframe[column].dtype in (np.float64, np.int64)
        for column in dataframe.columns
    ]
    assert dataframe.shape == (len(samples_to_daos), len(feature_extractor.features))


def test_feature_extraction_with_aggregations(scenario):
    room, room2, room_dao, room2_dao, feature_extractor = scenario
    rpc = RelationalProbabilisticCircuit(SceneRoom)
    rpc.fit([room_dao, room2_dao])

    room_query = underspecified(SceneRoom)(
        position=underspecified(KRROODPosition)(x=..., y=..., z=...),
        orientation=underspecified(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[underspecified(SceneObject)(type=...) for _ in range(4)],
    )
    room_query.resolve()
    model = rpc.ground(room_query)
    model = model.simplify()

    assert model.is_valid()


def test_create_dataframe_with_aggregations(scenario):
    room, room2, room_dao, room2_dao, feature_extractor = scenario
    dataframe = feature_extractor.create_dataframe([room_dao, room2_dao])
    assert dataframe.shape == (2, len(feature_extractor.features))
    assert dataframe.columns.tolist() == feature_extractor.features
    assert dataframe.iloc[0, 1] == room_dao.position.x
    assert dataframe.iloc[1, 1] == room2_dao.position.x


def test_apply_mapping_with_aggregations(scenario):
    room, room2, room_dao, room2_dao, feature_extractor = scenario
    mapping = feature_extractor.apply_mapping(room_dao)
    assert len(mapping) == 11
    assert mapping[10] == 3  # total count


def test_dataframe_preprocessing(scenario):
    room, room2, room_dao, room2_dao, feature_extractor = scenario
    dataframe = feature_extractor.create_dataframe([room_dao, room2_dao])
    preprocessed_df = feature_extractor.preprocess_dataframe(dataframe)
    assert preprocessed_df.shape == (2, len(feature_extractor.features))
    assert preprocessed_df.columns.tolist() == feature_extractor.features
    assert preprocessed_df.iloc[0, 0] == room_dao.type_in_need_of_preprocessing
    assert preprocessed_df.iloc[1, 0] == room2_dao.type_in_need_of_preprocessing
    assert preprocessed_df.iat[0, 0] == 0  # 0 for False, 1 for True
    assert preprocessed_df.iat[1, 0] == 0  # 0 for False, 1 for True


def test_missing_inheritance_from_mixin():
    instance = MissingBaseClass([ExampleInt(1), ExampleInt(1), ExampleInt(1)])
    result = FeatureExtractor.from_instances([to_dao(instance)])
    assert result.features == []


def test_feature_extractor_on_non_compatible_attribute_types():
    instance = ExampleString("test")
    extractor = FeatureExtractor.from_instances([to_dao(instance)])
    assert extractor.features == []


def test_iterable_literal_with_enum_feature_uses_symbolic_variable():
    query = underspecified(SceneRoom)(
        position=underspecified(KRROODPosition)(x=..., y=..., z=...),
        orientation=underspecified(KRROODOrientation)(x=..., y=..., z=..., w=...),
        objects=[SceneObject(type=SceneObjectType.TABLE)],
    )
    parameters = UnderspecifiedParameters(query)
    type_variables = [
        var for name, var in parameters.variables.items() if name.endswith(".type")
    ]
    assert type_variables
    assert all(isinstance(var, Symbolic) for var in type_variables)


def test_feature_extractor_on_unique_parts_with_none():
    instance = NestedAction(obj=Body(name="test"), pose=None)
    feature_extractor = FeatureExtractor.from_instances([to_dao(instance)])
    assert (
        len(feature_extractor.features) == 1
    )  # None gets filtered out though it is unique part
