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
    fill_dataframe_with_parts,
    get_features_of_class,
    FeatureExtractor,
    preprocess_dataframe,
)
from probabilistic_model.probabilistic_circuit.relational.main import Nation
from probabilistic_model.probabilistic_circuit.relational.rspns import (
    RSPNSpecification,
    RSPNTemplate,
)
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from pycram.robot_plans.actions.composite.transporting import MoveAndPickUpAction
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
        standing_position=underspecified(
            PoseMapping.from_point_mapping_quaternion_mapping
        )(
            point_mapping=underspecified(Point3Mapping)(
                x=..., y=..., z=..., reference_frame=None
            ),
            quaternion_mapping=underspecified(QuaternionMapping)(
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

    # move_and_pick_up_description: Match = move_and_pick_up_description

    parameters = UnderspecifiedParameters(move_and_pick_up_description)

    # sampling
    move_and_pick_up_distribution = fully_factorized(parameters.variables.values())

    probabilistic_registry = DictRegistry(
        {MoveAndPickUpAction: move_and_pick_up_distribution}
    )

    backend = ProbabilisticBackend(probabilistic_registry, number_of_samples=50)

    values = list(backend.evaluate(move_and_pick_up_description))
    return values, move_and_pick_up_distribution


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


def test_move_and_pick_up(database, mutable_model_world, data_preparation):
    values, move_and_pick_up_distribution = data_preparation

    # avg log likelihood auf den traingsdaten und dann auf dem gelernten circuit, der sollte hoehere log likelihood haben
    data_access_objects = [to_dao(value) for value in values]
    template = LearnRSPN(MoveAndPickUpAction, data_access_objects)
    feature_extractor = FeatureExtractor(
        get_features_of_class(
            to_dao(values[0]), variable(MoveAndPickUpAction, []), [], set()
        )
    )
    dataframe = feature_extractor.create_dataframe(data_access_objects)
    dataframe = preprocess_dataframe(feature_extractor.features, dataframe)
    sorted = dataframe.sort_index(axis=1)
    final = sorted.to_numpy()
    move_and_pick_up_distribution.plot_structure()
    plt.savefig(f"test_move_and_pick_up_{datetime.datetime.now()}.png")
    # assert sorted.columns == [v.name for v in move_and_pick_up_distribution.variables]
    print("distribution", move_and_pick_up_distribution.leaves)
    print("likelihood", move_and_pick_up_distribution.likelihood(final))
    print(
        f"Log likelihood of the move and pick up distribution: {np.mean(move_and_pick_up_distribution.log_likelihood(final))}"
    )
    assert np.mean(template.probabilistic_circuit.log_likelihood(final)) > np.mean(
        move_and_pick_up_distribution.log_likelihood(final)
    )
    template.probabilistic_circuit.plot_structure()
    plt.savefig(f"test_{datetime.datetime.now()}.png")
    plt.close()

    # grounded = template.ground(values[0])
    # grounded.probabilistic_circuit.plot_structure()
    # plt.savefig(f"test_ground_{datetime.datetime.now()}.png")
    # plt.close()

    # exchangeable = underspecified(Nation)(persons=[underspecified(Person)(name="Checker Chang", age=...)])
    # exchangeable_parameters = UnderspecifiedParameters(exchangeable)
    # print([type(mapped_variable) for mapped_variable in exchangeable._get_mapped_variable_by_name(
    #     "Nation.persons[0].age")._access_path_])
    # print([mapped_variable._type_ for mapped_variable in move_and_pick_up_description._get_mapped_variable_by_name(
    #     "MoveAndPickUpAction.standing_position.pose.position.z")._access_path_])
