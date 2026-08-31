"""
The Stretch apartment demonstration against a controller running in its own process.

Every layer the demonstration stands on is verified in isolation elsewhere: the
controller configuration and actuation in ``test/giskardpy_test/test_ros2_stuff``, the
world fetch and synchronization in ``test/semantic_digital_twin_test/test_ros``, and the
motion mappings and action plans in ``test/coraplex_test``. This test is the only one
that runs them together with the controller in its own process, sharing no interpreter
state -- and the only one that notices when the demonstration's own spatial parameters
stop being feasible.
"""

import numpy as np

from coraplex.datastructures.enums import ExecutionType
from experiments.real_stretch_apartment_demo.demo import (
    CEREAL_NAME,
    CEREAL_SHELF_LAYER_NAME,
    CEREAL_SHELF_LAYER_T_CEREAL,
    StretchApartmentDemonstration,
)
from semantic_digital_twin.robots.stretch import Stretch
from semantic_digital_twin.world import World

RESULT_FETCH_TIMEOUT_SECONDS = 60
"""
How long to wait for the controller to serve its world once the demonstration is done.
"""

PLACEMENT_TOLERANCE = 0.2
"""
How far from the pose the plan places it at the cereal box may end up.

Loose enough not to chase controller noise, and far tighter than the distance to the
bedside table it is carried to in between, so a cereal left there fails this.
"""


def test_demonstration_runs_against_a_controller_in_another_process(
    stretch_controller_process, cereal_perception_process
):
    """
    The demonstration drives a controller and a perception pipeline it shares no
    interpreter state with, so every exchange crosses a real process boundary: fetching
    the world, synchronizing the furniture it spawns, detecting the cereal, and
    executing each action.

    The result is read back by fetching the world from the controller again, which
    proves the furniture and the transported object landed in the controller's own
    process rather than only in the demonstration's copy. The plan carries the cereal to
    the bedside table and back, so it ends on its shelf layer again -- released to the
    world rather than still hanging off the shelf it was spawned under or off the
    gripper.
    """
    StretchApartmentDemonstration(
        execution_type=ExecutionType.REAL, used_robot=Stretch
    ).run()

    controller_world = stretch_controller_process.session.fetch_world(
        timeout_seconds=RESULT_FETCH_TIMEOUT_SECONDS
    )

    cereal = controller_world.get_body_by_name(CEREAL_NAME)
    shelf_layer = controller_world.get_body_by_name(CEREAL_SHELF_LAYER_NAME)
    assert cereal.parent_connection.parent is controller_world.root

    root_T_shelf_layer = controller_world.compute_forward_kinematics(
        controller_world.root, shelf_layer
    ).to_np()
    root_T_cereal = controller_world.compute_forward_kinematics(
        controller_world.root, cereal
    ).to_np()
    np.testing.assert_allclose(
        root_T_cereal[:2, 3],
        (root_T_shelf_layer @ CEREAL_SHELF_LAYER_T_CEREAL.to_np())[:2, 3],
        atol=PLACEMENT_TOLERANCE,
    )


def test_the_scene_counts_as_populated_once_the_cereal_is_spawned(apartment_meshes):
    """
    The demonstration decides whether to spawn its scene by the presence of the cereal.
    """
    demonstration = StretchApartmentDemonstration(used_robot=Stretch)
    world = World.create_with_root_body()
    assert not demonstration.is_scene_populated(world)

    demonstration.populate_scene(world)

    assert demonstration.is_scene_populated(world)
