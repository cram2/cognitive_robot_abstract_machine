import os
import time

import numpy as np

from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
    ShapeSource,
)
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from physics_simulators.base_simulator import SimulatorState
from semantic_digital_twin.adapters.sage_10k_dataset.loader import Sage10kDatasetLoader
from semantic_digital_twin.adapters.sage_10k_dataset.schema import Sage10kScene
from semantic_digital_twin.pipeline.mesh_decomposition.box_decomposer import (
    BoxDecomposer,
)
from semantic_digital_twin.pipeline.pipeline import Pipeline
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


def test_loader(rclpy_node):
    loader = Sage10kDatasetLoader()
    scene = loader.create_scene(scene_url=Sage10kDatasetLoader.available_scenes()[0])
    world = scene.create_world()
    pub = VizMarkerPublisher(
        _world=world,
        node=rclpy_node,
    )
    pub.with_tf_publisher()
    verify_scene(world, scene)


def test_different_decomposition_methods(
    rclpy_node,
):
    loader = Sage10kDatasetLoader()
    scene = loader.create_scene(scene_url=Sage10kDatasetLoader.available_scenes()[0])

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

def stop_multisim_if_running(multi_sim: MujocoSim) -> None:
    simulator = getattr(multi_sim, "simulator", None)
    if simulator is None:
        return
    if getattr(simulator, "state", None) is SimulatorState.STOPPED:
        return
    multi_sim.stop_simulation()

def test_multi_sim_10_times():
    scene_url = Sage10kDatasetLoader.available_scenes()[0]
    for _ in range(10):
        loader = Sage10kDatasetLoader()
        scene = loader.create_scene(scene_url=scene_url)

        for room in scene.rooms:
            new_objects = []
            for obj in room.objects:
                if obj.type in ["bookshelf", "book"]:
                    new_objects.append(obj)
            room.objects = new_objects

            room.walls = []
            room.doors = []

        world = scene.create_world()
        decomposer = BoxDecomposer()
        pipeline = Pipeline([decomposer])
        pipeline.apply(world)
        headless = os.environ.get("CI", "false").lower() == "true"
        multi_sim = MujocoSim(world=world, headless=headless)

        try:
            multi_sim.start_simulation()
            start_time = time.time()
            time.sleep(1.0)
            multi_sim.stop_simulation()

            assert time.time() - start_time >= 1.0
        finally:
            stop_multisim_if_running(multi_sim)