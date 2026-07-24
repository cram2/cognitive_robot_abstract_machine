"""
Underspecified mixing demo: a PR2 mixes the contents of a bowl on the apartment kitchen
counter with a whisk mounted on its right gripper.

Unlike the simple demo, no concrete values are given for the base pose or the mixing
duration: both are left as ellipses and sampled from the probabilistic backend, bounded
only by where-conditions.
"""

# Importing the ORM interface registers the data access objects the probabilistic
# backend needs to condition on the given literal values.
import coraplex.orm.ormatic_interface
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a

from experiments.tool_based_actions.simple_demo.demo_world import (
    BOWL_COLOR,
    MIX_MOUNT,
    parse_object,
)
from experiments.tool_based_actions.underspecified_demo.demo_setup import (
    build_underspecified_navigation,
    place_target_on_counter,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Whisk,
)

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.tool_based import MixingAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import attach_tool, setup_world, start_visualization

MINIMUM_MIX_DURATION = 2.0
"""
Lower bound in seconds of the sampled mixing duration.
"""

MAXIMUM_MIX_DURATION = 6.0
"""
Upper bound in seconds of the sampled mixing duration.
"""


def main() -> None:
    """
    Build the demo world and run the underspecified plan on the simulated robot.
    """
    world = setup_world()

    bowl_body = place_target_on_counter(world, "bowl.stl", BOWL_COLOR)
    start_visualization(world)

    pr2 = PR2.from_world(world)
    context = Context(world=world, robot=pr2, _debug=False, ros_node=None)
    context.query_backend = ProbabilisticBackend()

    whisk_body = attach_tool(
        world, pr2, Arms.RIGHT, parse_object("whisk.stl"), MIX_MOUNT
    )

    whisk = Whisk(root=whisk_body)
    with world.modify_world():
        world.add_semantic_annotations([Bowl(root=bowl_body), whisk])

    context.evaluate_conditions = False

    navigate = build_underspecified_navigation(world)

    mixing = a(MixingAction)(
        container=bowl_body,
        arm=Arms.RIGHT,
        tool=whisk,
        mix_duration=...,
    )
    mixing.where(
        mixing.variable.mix_duration > MINIMUM_MIX_DURATION,
        mixing.variable.mix_duration < MAXIMUM_MIX_DURATION,
    )

    plan = sequential(
        [
            SetGripperAction(Arms.RIGHT, GripperState.CLOSE),
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            navigate,
            mixing,
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()


if __name__ == "__main__":
    main()
