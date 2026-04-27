import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import rclpy
from matplotlib import pyplot as plt

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import (
    match_variable,
    match,
    variable_from,
    variable,
    underspecified,
)
from krrood.entity_query_language.query.match import Match
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.data_access_objects.to_dao import ToDataAccessObjectState
from krrood.ormatic.utils import create_engine, drop_database
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from krrood_test.dataset.example_classes import (
    KRROODPose,
    KRROODPosition,
    KRROODOrientation,
)
from probabilistic_model.probabilistic_circuit.relational.learn_rspn import (
    LearnRSPN,
    get_features_of_class,
    FeatureExtractor,
    preprocess_dataframe,
    get_features_of_class_bfs,
)
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from pycram.robot_plans.actions.composite.transporting import MoveAndPickUpAction
from random_events.product_algebra import Event
from semantic_digital_twin.orm.model import (
    QuaternionMapping,
    Point3Mapping,
    PoseMapping,
)
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from sqlalchemy.orm import Session, session
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    ApproachDirection,
    Arms,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription, GraspPose
from pycram.orm.ormatic_interface import *
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Quaternion

rclpy.init()
uri = os.environ["SEMANTIC_DIGITAL_TWIN_DATABASE_URI"]
engine = sqlalchemy.create_engine(uri)
# node = rclpy.create_node("simple_viz_node")


@pytest.fixture(scope="function")
def mutable_model_world(pr2_apartment_world):
    world = deepcopy(pr2_apartment_world)
    pr2 = PR2.from_world(world)
    return world, pr2, Context(world, pr2)


@pytest.fixture(scope="function")
def database():
    session = Session(engine)
    Base.metadata.create_all(bind=session.bind)
    yield session
    drop_database(engine)
    session.expunge_all()
    session.close()


@pytest.fixture(scope="function")
def data_preparation(mutable_model_world):
    world, robot_view, context = mutable_model_world

    milk = world.get_body_by_name("milk.stl")

    milk_variable = variable_from([milk])

    move_and_pick_up_description = underspecified(MoveAndPickUpAction)(
        keep_joint_states=...,
        standing_position=underspecified(
            PoseMapping.from_point_mapping_quaternion_mapping
        )(
            position=underspecified(Point3Mapping)(
                x=..., y=..., z=..., reference_frame=None
            ),
            orientation=underspecified(QuaternionMapping)(
                x=..., y=..., z=..., w=..., reference_frame=None
            ),
            reference_frame=variable_from([robot_view.root]),
        ),
        object_designator=milk_variable,
        arm=...,
        grasp_description=underspecified(GraspDescription)(
            approach_direction=...,
            vertical_alignment=...,
            rotate_gripper=...,
            manipulation_offset=0.05,
            manipulator=variable(Manipulator, world.semantic_annotations),
        ),
    )

    parameters = UnderspecifiedParameters(move_and_pick_up_description)
    fully_factorized_circuit = fully_factorized(parameters.variables.values())
    assert len(parameters._events_from_symbolic_expression) == 3

    if parameters.conditioning_event is not None:
        fully_factorized_circuit, _ = fully_factorized_circuit.truncated(
            parameters.conditioning_event, singleton_allowed=True
        )
    probabilistic_registry = DictRegistry(
        {MoveAndPickUpAction: fully_factorized_circuit}
    )

    np.random.seed(69)
    backend = ProbabilisticBackend(probabilistic_registry, number_of_samples=50)
    samples = list(backend.evaluate(move_and_pick_up_description))
    assert all(
        [sample.object_designator == samples[0].object_designator for sample in samples]
    )
    samples_to_daos = [to_dao(sample) for sample in samples]

    return samples_to_daos, fully_factorized_circuit


def test_move_and_pick_up(database, mutable_model_world, data_preparation):
    samples, circuit = data_preparation

    feature_extractor = FeatureExtractor(
        get_features_of_class_bfs(samples[0], variable(MoveAndPickUpAction, []))
    )
    dataframe = feature_extractor.create_dataframe(samples)
    dataframe = preprocess_dataframe(feature_extractor.features, dataframe)
    sorted = dataframe.sort_index(axis=1)
    final = sorted.to_numpy()
    # one_sample = final.tolist()[0]
    # assert sorted.columns == [v.name for v in move_and_pick_up_distribution.variables]
    identical_variables = [
        variable
        for variable in circuit.variables
        if variable.name in dataframe.columns.values
    ]
    # remove unnecessary variables from circuit (obj_desig, ref_frame, manip)
    circuit = circuit.marginal(identical_variables)

    template = LearnRSPN(MoveAndPickUpAction, samples)

    assert np.mean(template.probabilistic_circuit.log_likelihood(final)) > np.mean(
        circuit.log_likelihood(final)
    )


def test_features_extraction(database, data_preparation):
    values, move_and_pick_up_distribution = data_preparation

    features = get_features_of_class(
        to_dao(values[0]), variable(MoveAndPickUpAction, []), [], set()
    )

    feature_extractor = FeatureExtractor(features)
    to_data_access_object_state = ToDataAccessObjectState()
    data_access_objects = [
        to_dao(sample, state=to_data_access_object_state) for sample in values
    ]
    dataframe = feature_extractor.create_dataframe(data_access_objects)

    assert [
        dataframe[column].dtype in (np.float64, np.int64)
        for column in dataframe.columns
    ]
    assert dataframe.shape == (len(values), len(features))
