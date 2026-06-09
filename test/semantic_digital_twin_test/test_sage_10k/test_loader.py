import numpy as np
import pytest
from requests import HTTPError

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
    ShapeSource,
)
from semantic_digital_twin.adapters.sage_10k_dataset.loader import (
    Sage10kDatasetLoader,
)
from semantic_digital_twin.adapters.sage_10k_dataset.schema import Sage10kScene
from semantic_digital_twin.pipeline.mesh_decomposition.box_decomposer import (
    BoxDecomposer,
)
from semantic_digital_twin.pipeline.pipeline import Pipeline
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Book,
    BookFront,
)
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.world import World

from semantic_digital_twin.adapters.mesh import STLParser

from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)

from pycram.motion_executor import simulated_robot

from pycram.plans.factories import execute_single, sequential

from pycram.robot_plans.actions.core.navigation import NavigateAction

from pycram.datastructures.dataclasses import Context

from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment

from pycram.robot_plans.actions.core.pick_up import PickUpAction

from semantic_digital_twin.datastructures.definitions import TorsoState


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
    world = scene.create_world()
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
