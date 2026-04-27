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

    move_and_pick_up_distribution = fully_factorized(parameters.variables.values())
    assert len(parameters.events_from_symbolic_expressions) == 3
    complete_event = parameters.events_from_symbolic_expressions[0]
    complete_event.fill_missing_variables(parameters.variables.values())

    [
        event.fill_missing_variables(parameters.variables.values())
        for event in parameters.events_from_symbolic_expressions
    ]

    [
        event.fill_missing_variables(parameters.variables.values())
        for event in parameters.events_from_literal_values
    ]

    complete_event = parameters.events_from_symbolic_expressions[0]
    for other_event in parameters.events_from_symbolic_expressions[1:]:
        complete_event = complete_event.intersection_with(other_event)
    for other_event in parameters.events_from_literal_values:
        complete_event = complete_event.intersection_with(other_event)

    m2, prob = move_and_pick_up_distribution.truncated(
        complete_event, singleton_allowed=True
    )
    probabilistic_registry = DictRegistry(
        {MoveAndPickUpAction: move_and_pick_up_distribution}
    )

    np.random.seed(69)

    backend = ProbabilisticBackend(probabilistic_registry, number_of_samples=50)

    samples = list(backend.evaluate(move_and_pick_up_description))
    assert all(
        [sample.object_designator == samples[0].object_designator for sample in samples]
    )
    return samples, m2


def test_move_and_pick_up(database, mutable_model_world, data_preparation):
    samples, m2 = data_preparation

    # avg log likelihood auf den traingsdaten und dann auf dem gelernten circuit, der sollte hoehere log likelihood haben
    data_access_objects = [to_dao(value) for value in samples]

    feature_extractor = FeatureExtractor(
        get_features_of_class_bfs(to_dao(samples[0]), variable(MoveAndPickUpAction, []))
    )
    dataframe = feature_extractor.create_dataframe(data_access_objects)
    dataframe = preprocess_dataframe(feature_extractor.features, dataframe)
    sorted = dataframe.sort_index(axis=1)
    final = sorted.to_numpy()
    # one_sample = final.tolist()[0]
    # assert sorted.columns == [v.name for v in move_and_pick_up_distribution.variables]
    identical_variables = [
        variable
        for variable in m2.variables
        if variable.name in dataframe.columns.values
    ]
    # remove unnecessary variables from circuit (obj_desig, ref_frame, manip)
    m2 = m2.marginal(identical_variables)

    template = LearnRSPN(MoveAndPickUpAction, data_access_objects)

    # Debugging
    # print(f"\nLearned circuit: {template.probabilistic_circuit}")
    #
    # for i, row in enumerate(final):
    #     ll = template.probabilistic_circuit.log_likelihood(row.reshape(1, -1))
    #     if ll == -np.inf:
    #         print(f"Sample {i} has -inf LL")
    #         for j, val in enumerate(row):
    #             var = [
    #                 v
    #                 for v in template.probabilistic_circuit.variables
    #                 if template.probabilistic_circuit.variable_to_index_map[v] == j
    #             ][0]
    #             marginal = template.probabilistic_circuit.marginal([var])
    #             leaf_ll = marginal.log_likelihood(np.array([[val]]))
    #             if leaf_ll == -np.inf:
    #                 print(
    #                     f"  Variable {j} ({dataframe.columns[j]}) has -inf LL: value={val}"
    #                 )
    #                 # Try to find the distribution
    #                 for node in marginal.nodes():
    #                     if hasattr(node, "distribution"):
    #                         print(f"    Distribution: {node.distribution}")
    #                         if hasattr(node.distribution, "location"):
    #                             print(
    #                                 f"    Location: {node.distribution.location}, Tolerance: {node.distribution.tolerance}"
    #                             )
    #
    # log_likelihoods = template.probabilistic_circuit.log_likelihood(final)
    # print(f"Log likelihoods: {log_likelihoods}")
    #
    # # Check if any column is constant
    # for i, col in enumerate(final.T):
    #     if len(np.unique(col)) == 1:
    #         print(f"Column {i} ({dataframe.columns[i]}) is constant: {col[0]}")

    impossible_samples = final[
        template.probabilistic_circuit.log_likelihood(final) == -np.inf
    ]

    print(len(impossible_samples))
    print(len(final))

    print(f"Impossible samples: {impossible_samples}")
    assert np.mean(template.probabilistic_circuit.log_likelihood(final)) > np.mean(
        m2.log_likelihood(final)
    )
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
