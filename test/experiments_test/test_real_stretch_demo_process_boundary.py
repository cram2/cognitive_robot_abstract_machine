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
from experiments.real_stretch_apartment_demo.demo import StretchApartmentDemonstration
from semantic_digital_twin.robots.stretch import Stretch

RESULT_FETCH_TIMEOUT_SECONDS = 60
"""
How long to wait for the controller to serve its world once the demonstration is done.
"""

PLACEMENT_TOLERANCE = 0.2
"""
How far from the bedside table's centre the cereal box may land.

The measured placement error is about 0.1m, so this discriminates a successful place
from one that missed the furniture without being tight enough to chase controller noise.
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
    process rather than only in the demonstration's copy.
    """
    StretchApartmentDemonstration(
        execution_type=ExecutionType.REAL, used_robot=Stretch
    ).run()

    controller_world = stretch_controller_process.session.fetch_world(
        timeout_seconds=RESULT_FETCH_TIMEOUT_SECONDS
    )

    cereal = controller_world.get_body_by_name("cheeze_it.obj")
    bedside_table = controller_world.get_body_by_name("bedside_table.dae")
    assert cereal.parent_connection.parent is not controller_world.get_body_by_name(
        "shelf_layer2"
    )
    np.testing.assert_allclose(
        controller_world.compute_forward_kinematics(controller_world.root, cereal)
        .to_position()
        .to_np()[:2],
        controller_world.compute_forward_kinematics(
            controller_world.root, bedside_table
        )
        .to_position()
        .to_np()[:2],
        atol=PLACEMENT_TOLERANCE,
    )
