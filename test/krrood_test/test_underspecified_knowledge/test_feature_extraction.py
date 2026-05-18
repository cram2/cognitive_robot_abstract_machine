import numpy as np

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    underspecified,
)
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extractor import FeatureExtractor
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from ..dataset import ormatic_interface  # type: ignore
from ..dataset.example_classes import (
    NestedAction,
    KRROODPose,
    KRROODPosition,
    KRROODOrientation,
)
from ..dataset.semantic_world_like_classes import Body


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
