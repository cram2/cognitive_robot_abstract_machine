import numpy as np

from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import underspecified
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extractor import FeatureExtractor
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from probabilistic_model.probabilistic_circuit.relational.learn_rspn import (
    learn_probabilistic_circuit,
)
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from ..dataset.example_classes import (
    NestedAction,
    KRROODPose,
    KRROODPosition,
    KRROODOrientation,
)
from ..dataset.ormatic_interface import *  # type: ignore
from ..dataset.semantic_world_like_classes import Body


def test_rspn_learning():
    action = underspecified(NestedAction)(
        pose=underspecified(KRROODPose)(
            position=underspecified(KRROODPosition)(x=..., y=..., z=...),
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
    assert all([sample.obj == samples[0].obj for sample in samples])

    samples_to_daos = [to_dao(sample) for sample in samples]

    feature_extractor = FeatureExtractor.from_instances(samples_to_daos)
    dataframe = feature_extractor.create_dataframe(samples_to_daos)
    dataframe = feature_extractor.preprocess_dataframe(dataframe)
    sorted = dataframe.sort_index(axis=1)
    final = sorted.to_numpy()
    identical_variables = [
        variable
        for variable in fully_factorized_circuit.variables
        if variable.name in dataframe.columns.values
    ]
    # remove unnecessary variables from circuit (obj_desig, ref_frame, manip)
    circuit = fully_factorized_circuit.marginal(identical_variables)

    learned_circuit = learn_probabilistic_circuit(samples_to_daos)

    assert np.mean(learned_circuit.log_likelihood(final)) > np.mean(
        circuit.log_likelihood(final)
    )
