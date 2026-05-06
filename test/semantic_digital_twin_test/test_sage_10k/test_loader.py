import os
from copy import deepcopy

import numpy as np
import pytest
from nltk.corpus import wordnet
from requests import HTTPError

from krrood.entity_query_language.factories import *
from probabilistic_model.bayesian_network.bayesian_network import Node
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.transporting import (
    MoveAndPickUpAction,
    MoveAndPlaceAction,
)
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.pick_up import PickUpAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from pycram.view_manager import ViewManager
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
    ShapeSource,
)
from semantic_digital_twin.adapters.sage_10k_dataset.loader import (
    Sage10kDatasetLoader,
)
from semantic_digital_twin.adapters.sage_10k_dataset.processing import (
    create_hsrb_in_world,
)
from semantic_digital_twin.adapters.sage_10k_dataset.schema import Sage10kScene
from semantic_digital_twin.adapters.sage_10k_dataset.utils import (
    Sage10kTypeNameCleaner,
    NaturalLanguageWithTypeDescription,
    sage_10k_non_shitty_scenes_demo_configs,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.pipeline.mesh_decomposition.box_decomposer import (
    BoxDecomposer,
)
from semantic_digital_twin.pipeline.pipeline import Pipeline
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.natural_language import (
    most_similar_synonym,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world import World


def verify_scene(world: World, scene: Sage10kScene):
    """
    Verify that the object positions of the scene are the same as in the world.
    Sometimes the scene contains two objects with the same ID. In that case, this check is skipped
    :param world: The world created from the scene.
    :param scene: The scene.
    """

    for room in scene.rooms:
        for obj in room.objects:
            matching_bodies = [b for b in world.bodies if b.name.prefix == obj.id]

            if len(matching_bodies) > 1:
                continue

            body = matching_bodies[0]

            global_position = body.global_pose.to_position()
            assert np.isclose(global_position.x, obj.position.x)
            assert np.isclose(global_position.y, obj.position.y)
            assert np.isclose(global_position.z, obj.position.z)


def get_body_height(body) -> float:
    return body.global_pose.z


def has_book_in_prefix(body) -> bool:
    return body.name.prefix is not None and "_book_" in body.name.prefix.lower()


def get_book_body_by_height(world: World, target_height: float, atol: float = 1e-5):

    book = wordnet.synsets("Book")[1]

    natural_language_annotations = world.get_semantic_annotations_by_type(
        NaturalLanguageWithTypeDescription
    )
    types_of_world = {a.type_description for a in natural_language_annotations}

    object_type_description = variable_from(types_of_world)
    book_type_description = max(
        object_type_description, key=lambda t: most_similar_synonym(t, book)[0]
    ).tolist()[0]
    candidates = [
        a
        for a in natural_language_annotations
        if a.type_description == book_type_description
    ]
    if not candidates:
        candidates = [body for body in world.bodies if has_book_in_prefix(body)]

    if not candidates:
        preview = [
            f"{str(body.name)} ({get_body_height(body):.5f})"
            for body in world.bodies[:20]
        ]
        raise ValueError(
            "No Book semantic annotations and no bodies with 'book' in the name were found. "
            f"First bodies: {preview}"
        )

    exact_matches = [
        candidate.root
        for candidate in candidates
        if np.isclose(get_body_height(candidate.root), target_height, atol=atol)
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    if len(exact_matches) > 1:
        raise ValueError(
            f"Expected a single Book body with height {target_height}, but found "
            f"{[str(body.name) for body in exact_matches]}."
        )

    closest_body = min(
        candidates, key=lambda body: abs(get_body_height(body) - target_height)
    )
    closest_height = get_body_height(closest_body)

    print(
        f"No exact Book body height match for {target_height}. "
        f"Using closest candidate {closest_body.name} with height {closest_height}."
    )
    return closest_body


def get_sage10k_scene():
    try:
        loader = Sage10kDatasetLoader()
        return loader.create_scene(scene_url=Sage10kDatasetLoader.available_scenes()[0])
    except HTTPError as e:
        return None


@pytest.fixture
def sage10k_scene():
    return get_sage10k_scene()


@pytest.mark.skipif(get_sage10k_scene() is None, reason="Sage10k dataset not available")
def test_loader(rclpy_node, sage10k_scene):
    scene = sage10k_scene
    if scene is None:
        return
    world = scene.create_world(Sage10kTypeNameCleaner())
    pub = VizMarkerPublisher(
        _world=world,
        node=rclpy_node,
    )
    pub.with_tf_publisher()
    verify_scene(world, scene)
    assert (
        len(world.get_semantic_annotations_by_type(NaturalLanguageWithTypeDescription))
        > 0
    )


@pytest.mark.skipif(get_sage10k_scene() is None, reason="Sage10k dataset not available")
def test_loader_with_robot(rclpy_node, sage10k_scene, pr2_world_copy):
    pr2_world = pr2_world_copy

    try:
        loader = Sage10kDatasetLoader()
        scene = loader.create_scene(
            scene_url=Sage10kDatasetLoader.available_scenes()[0]
        )
    except HTTPError as e:
        return "Sage10k dataset not available"

    world = scene.create_world()

    VizMarkerPublisher(
        _world=pr2_world,
        node=rclpy_node,
    ).with_tf_publisher()
    navigate_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        3.96, 6.06, 0, yaw=np.pi / 2, reference_frame=pr2_world.root
    )
    context = Context.from_world(pr2_world)
    left_arm = ViewManager.get_arm_view(Arms.LEFT, context.robot)
    manipulator = left_arm.manipulator

    grasp_description = GraspDescription(
        ApproachDirection.BACK,
        VerticalAlignment.NoAlignment,
        manipulator,
    )
    target_body = get_book_body_by_height(pr2_world, 1.22921)
    root = sequential(
        [
            ParkArmsAction(arm=Arms.BOTH),
            NavigateAction(navigate_pose),
            MoveTorsoAction(TorsoState.HIGH),
            PickUpAction(
                object_designator=target_body,
                arm=Arms.LEFT,
                grasp_description=grasp_description,
            ),
            ParkArmsAction(arm=Arms.BOTH),
        ],
        context,
    )
    with simulated_robot:
        root.perform()
    assert (
        pr2_world.get_connection(
            left_arm.manipulator.tool_frame,
            target_body,
        )
        is not None
    )


@pytest.mark.skipif(get_sage10k_scene() is None, reason="Sage10k dataset not available")
def test_non_shitty_scenes_demo(rclpy_node):

    for config in sage_10k_non_shitty_scenes_demo_configs:
        try:
            loader = Sage10kDatasetLoader()
            scene = loader.create_scene(scene_url=config.scene_url)
        except HTTPError as e:
            return "Sage10k dataset not available"

        world = scene.create_world()
        robot = create_hsrb_in_world(world)

        viz = VizMarkerPublisher(
            _world=world,
            node=rclpy_node,
        )
        viz.with_tf_publisher()

        # input(
        #     f"Loaded scene from {config.scene_url}. Press Enter to continue to the next scene..."
        # )

        context = Context(world=world, robot=robot)

        [body] = world.get_bodies_by_global_position(
            config.world_P_object_of_interest, 0.1
        )
        # keep this in here to remind me of a weird bug @tomsch420
        # origin = body.parent_connection.origin
        # print(f"{origin=},{body.parent_kinematic_structure_entity.name=}")
        # input("pre change position")
        # body.parent_connection.origin = origin
        # input("post change position")
        # print(f"{body.parent_connection.origin=}")
        # input("post change position")
        arm = Arms.RIGHT
        grasp_description = GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            robot.arm.manipulator,
        )

        config.pickup_navigation_pose.reference_frame = world.root
        config.place_navigation_pose.reference_frame = world.root
        config.place_pose.reference_frame = world.root

        plan = sequential(
            [
                ParkArmsAction(Arms.BOTH),
                MoveAndPickUpAction(
                    object_designator=body,
                    standing_position=config.pickup_navigation_pose,
                    arm=arm,
                    grasp_description=grasp_description,
                ),
                ParkArmsAction(Arms.BOTH),
                MoveAndPlaceAction(
                    object_designator=body,
                    standing_position=config.place_navigation_pose,
                    arm=arm,
                    target_location=config.place_pose,
                ),
            ],
            context=context,
        ).plan

        with simulated_robot:
            plan.perform()

        viz._tf_publisher.tf_pub.destroy()
        viz.pub.destroy()


@pytest.mark.skipif(get_sage10k_scene() is None, reason="Sage10k dataset not available")
def test_different_decomposition_methods(rclpy_node, sage10k_scene):
    scene = sage10k_scene
    if scene is None:
        return
    for room in scene.rooms:
        new_objects = []
        for obj in room.objects:
            if obj.type in ["bookshelf", "sideboard", "table"]:
                new_objects.append(obj)
        room.objects = new_objects

        room.walls = []
        room.doors = []

    world = scene.create_world()
    decomposer = BoxDecomposer()
    pipeline = Pipeline([decomposer])
    pipeline.apply(world)

    pub = VizMarkerPublisher(
        _world=world,
        node=rclpy_node,
        shape_source=ShapeSource.COLLISION_ONLY,
    )
    pub.with_tf_publisher()
