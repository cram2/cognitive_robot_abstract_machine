"""
Underspecified cutting demo: a PR2 slices a bread on the apartment kitchen counter with
a knife mounted on its right gripper.

Unlike the simple demo, no concrete values are given for the base pose or the cutting
parameters: the base pose is sampled from a region in front of the counter, the cutting
technique is left as an ellipsis, and slice thickness and cut count are omitted entirely
so the action derives them from the bread's size.
"""

# Importing the ORM interface registers the data access objects the probabilistic
# backend needs to condition on the given literal values.
import coraplex.orm.ormatic_interface
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.entity_query_language.factories import a

from experiments.tool_based_actions.simple_demo.demo_world import (
    BREAD_COLOR,
    CUT_MOUNT,
    parse_object,
)
from experiments.tool_based_actions.underspecified_demo.demo_setup import (
    build_underspecified_navigation,
    place_target_on_counter,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bread,
    CuttingKnife,
)

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.execution_environment import simulated_robot
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.tool_based import CuttingAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
    SetGripperAction,
)
from coraplex.testing import attach_tool, setup_world, start_visualization


def main() -> None:
    """
    Build the demo world and run the underspecified plan on the simulated robot.
    """
    world = setup_world()

    bread_body = place_target_on_counter(world, "bread.stl", BREAD_COLOR)
    start_visualization(world)

    pr2 = PR2.from_world(world)
    context = Context(world=world, robot=pr2, _debug=False, ros_node=None)
    context.query_backend = ProbabilisticBackend()

    knife_body = attach_tool(
        world, pr2, Arms.RIGHT, parse_object("big-knife.stl"), CUT_MOUNT
    )

    knife = CuttingKnife(root=knife_body)
    with world.modify_world():
        world.add_semantic_annotations([Bread(root=bread_body), knife])

    context.evaluate_conditions = False

    navigate = build_underspecified_navigation(world)

    cutting = a(CuttingAction)(
        object_to_cut=bread_body,
        arm=Arms.RIGHT,
        tool=knife,
        technique=...,
    )

    plan = sequential(
        [
            SetGripperAction(Arms.RIGHT, GripperState.CLOSE),
            ParkArmsAction(Arms.BOTH),
            MoveTorsoAction(TorsoState.HIGH),
            navigate,
            cutting,
        ],
        context=context,
    ).plan

    with simulated_robot:
        plan.perform()


if __name__ == "__main__":
    main()
