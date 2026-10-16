"""
DAiSy opens and closes both grippers, first each one alone, then both at once.

The plan commands preset ``OPEN``/``CLOSE`` states first and flex ``FLEXOPEN``/
``FLEXCLOSE`` states afterwards, so both routing branches of a griplink gripper are
exercised in one run. It touches no objects, so it can be repeated against the same
scene.

The execution type decides what a gripper motion builds: in simulation the fingers
follow the joint positions the configuration commands in this process, a semi-real run
sends the same joint goals to a running controller that simulates the joints without
hardware, and a real run sends ``Grip``/``Release`` and ``Flexgrip``/``Flexrelease`` to
the griplink servers of the grippers.

Run it as a script, choosing the execution type with ``--execution``::

    python3 -m experiments.daisy_gripper_demo --execution simulated

Semi-real and real runs need a running DAiSy controller::

    ros2 launch giskardpy_ros giskardpy_daisy_standalone.launch.py

A semi-real run talks to the standalone controller that simulates the joints without
hardware; a real run fetches its world from the running controller and needs the
griplink driver of the grippers.
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.demonstrations import RobotDemonstration, RobotDemonstrationRosSession
from coraplex.execution_environment import (
    real_robot,
    semi_real_robot,
    kinematically_simulated_robot,
)
from coraplex.language import ParallelNode, SequentialNode
from coraplex.plans.factories import code, make_node
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans import MoveGripperMotion
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction
from coraplex.visualization import VisualizationSession
from semantic_digital_twin.adapters.ros.world_synchronizer import WorldSynchronizer
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World

logger = logging.getLogger(__name__)

EXECUTION_CHOICES = {
    "simulated": ExecutionType.KINEMATICALLY_SIMULATED,
    "semi-real": ExecutionType.SEMI_REAL,
    "real": ExecutionType.REAL,
}
"""
Execution types the ``--execution`` switch selects between.
"""

PAUSE_AFTER_MOTION_SECONDS = 1.0
"""
How long a run waits after a gripper motion before the next step announces itself,
letting the fingers settle into the state they just reached.
"""

STATE_VERBS = {
    GripperState.OPEN: "opening",
    GripperState.CLOSE: "closing",
    GripperState.FLEXOPEN: "flex-opening",
    GripperState.FLEXCLOSE: "flex-closing",
}
"""
The human-readable action each gripper state commands, used in the step announcements.
"""


def plan_node(root: PlanNode, action_likes: list, context: Context) -> PlanNode:
    """
    Build a plan with ``root`` on top and one child per entry of ``action_likes``.

    Mirrors how the plan factories mount children, so this demonstration can build plans
    with its own root node types.

    :param root: The node the plan is built below.
    :param action_likes: The actions, motions and plan nodes the root executes.
    :param context: The plan context shared by every node.
    :return: The root of the built plan.
    """
    plan = Plan(context=context)
    plan.add_node(root)
    for action_like in action_likes:
        child = make_node(action_like)
        if child.plan:
            root.mount_subplan(child)
        else:
            root.add_child(child)
    plan.simplify()
    return root


@dataclass(eq=False, repr=False)
class ExecutesChildrenSeparately(SequentialNode):
    """
    A sequential plan whose children are performed one after another, each on its own.

    Performing a child runs only its subtree, so every gripper motion executes in its
    own motion state chart and the code steps between the motions run when their turn
    comes, instead of all at once before any motion. Children that hold several motions,
    such as a parallel step, still share one chart.
    """

    def perform(self) -> None:
        for child in self.children:
            child.perform()


@dataclass(eq=False, repr=False)
class ExecutesChildrenTogether(ParallelNode):
    """
    A parallel plan whose motions execute together in one motion state chart.

    The inherited notification performs every motion on its own thread before the chart
    runs, which would execute each motion twice here; this node leaves the notification
    to the chart, which drives both grippers at once.
    """

    def notify(self) -> None:
        pass


@dataclass
class DAiSyGripperDemonstration(RobotDemonstration):
    """
    DAiSy opens and closes its grippers through the full preset/flex and
    individual/parallel matrix.
    """

    ros_node_name: ClassVar[str] = "daisy_gripper_demo_node"

    def build_simulated_world(self) -> World:
        """
        Build a world holding nothing but the DAiSy, since gripper motions touch no
        scene objects.
        """
        return WorldSpecification(
            world_parser=None,
            robots=[
                RobotSpecification(
                    semantic_annotation_type=self.used_robot,
                    odom_T_robot_start=HomogeneousTransformationMatrix.from_xyz_rpy(
                        1, 1
                    ),
                )
            ],
        ).to_domain_object()

    def is_scene_populated(self, world: World) -> bool:
        return True

    def populate_scene(self, world: World) -> None:
        """
        Spawn nothing; the demonstration acts only on the robot's grippers.
        """

    def acquire_world(self) -> World:
        """
        Obtain the world to act on.

        A semi-real run acts on the controller's world, like a real run does: the motion
        goals it sends name entities by identity, which only resolve against the world
        the controller itself acts on, and the synchronizer keeps this world in step
        with the controller's while the goals run.

        The controller publishes the state of that world itself, so the demonstration
        starts no visualization of its own: a second marker publisher would render the
        same robot a second time next to the controller's own rendering.
        """
        if self.execution_type is not ExecutionType.SEMI_REAL:
            logger.info(
                f"{self.execution_type.name.capitalize()} run: "
                "acquiring the world through the demonstration base."
            )
            return super().acquire_world()
        logger.info("Semi-real run: fetching the world of the running controller.")
        self.ros_session = RobotDemonstrationRosSession.start(self.ros_node_name)
        VisualizationSession.register(self.stop_visualization)
        world = self.ros_session.fetch_world()
        WorldSynchronizer(_world=world, node=self.ros_session.node)
        return world

    def run(self) -> World:
        """
        Acquire a world and perform the plan in the environment the execution type
        selects.

        Semi-real and real runs execute the chart on the running controller, so they
        enter the semi-real and real environments; a simulated run ticks its own chart.
        """
        try:
            world = self.acquire_world()
            if not self.is_scene_populated(world):
                self.populate_scene(world)
            for _ in range(self.repetitions):
                plan = self.build_plan(self.build_context(world))
                if self.visualization is not None:
                    self.visualization.attach_plan(plan)
                if self.execution_type == ExecutionType.REAL:
                    with real_robot:
                        plan.perform()
                elif self.execution_type == ExecutionType.SEMI_REAL:
                    with semi_real_robot:
                        plan.perform()
                else:
                    with kinematically_simulated_robot:
                        plan.perform()
        finally:
            self.tear_down()
        return world

    def build_context(self, world: World) -> Context:
        """
        Build the plan context around the DAiSy in ``world``.

        ..note:: The ROS node has to be in the context for a real robot.
        """
        return Context(
            world=world,
            robot=world.get_semantic_annotations_by_type(self.used_robot)[0],
            ros_node=self.ros_node,
            evaluate_conditions=False,
            alternative_motion_mappings=self.alternative_motion_mappings,
        )

    def build_plan(self, context: Context) -> PlanNode:
        """
        Open and close both grippers: each one alone, then both together, with preset
        states first and flex states afterwards.

        Every step announces itself before its motions and pauses after them.
        """
        daisy = context.robot
        left_gripper = daisy.left_arm.end_effector
        right_gripper = daisy.right_arm.end_effector

        def gripper_motion(gripper, state_type: GripperState) -> MoveGripperMotion:
            return MoveGripperMotion(
                configuration=gripper.default_configuration(state_type)
            )

        def gripper_step(
            side: str, gripper, state_type: GripperState
        ) -> list[PlanNode]:
            return [
                code(announce(f"the {side} gripper", state_type), context=context),
                gripper_motion(gripper, state_type),
                code(pause_after_motion, context=context),
            ]

        def both_grippers_step(state_type: GripperState) -> list[PlanNode]:
            return [
                code(announce("both grippers", state_type), context=context),
                plan_node(
                    ExecutesChildrenTogether(),
                    [
                        gripper_motion(left_gripper, state_type),
                        gripper_motion(right_gripper, state_type),
                    ],
                    context,
                ),
                code(pause_after_motion, context=context),
            ]

        return plan_node(
            ExecutesChildrenSeparately(),
            [
                # %% home arms first
                ParkArmsAction(arms=daisy.all_arms),
                # %% preset open/close, each gripper individually
                *gripper_step("left", left_gripper, GripperState.OPEN),
                *gripper_step("left", left_gripper, GripperState.CLOSE),
                *gripper_step("right", right_gripper, GripperState.OPEN),
                *gripper_step("right", right_gripper, GripperState.CLOSE),
                # %% preset open/close, both grippers at the same time
                *both_grippers_step(GripperState.OPEN),
                *both_grippers_step(GripperState.CLOSE),
                # %% flex open/close, each gripper individually
                *gripper_step("left", left_gripper, GripperState.FLEXOPEN),
                *gripper_step("left", left_gripper, GripperState.FLEXCLOSE),
                *gripper_step("right", right_gripper, GripperState.FLEXOPEN),
                *gripper_step("right", right_gripper, GripperState.FLEXCLOSE),
                # %% flex open/close, both grippers at the same time
                *both_grippers_step(GripperState.FLEXOPEN),
                *both_grippers_step(GripperState.FLEXCLOSE),
            ],
            context,
        )


def announce(subject: str, state_type: GripperState) -> Callable[[], None]:
    """
    Build the code step that logs what the following gripper motion commands.

    :param subject: Which gripper or grippers the following motion drives, e.g. ``the
        left gripper`` or ``both grippers``.
    :param state_type: The state the following motion drives the gripper to.
    :return: The announcement to run as a plan's code step.
    """

    def announce_step() -> None:
        logger.info(f"{STATE_VERBS[state_type].capitalize()} {subject}.")

    return announce_step


def pause_after_motion() -> None:
    """
    Let the grippers settle after the motion that just finished.
    """
    time.sleep(PAUSE_AFTER_MOTION_SECONDS)


def run_from_command_line() -> None:
    """
    Parse the ``--execution`` switch, make the demonstration's announcements visible in
    the terminal, and run the demonstration.
    """
    show_announcements_in_terminal()
    parser = argparse.ArgumentParser(description="DAiSy gripper open/close demo.")
    parser.add_argument(
        "--execution",
        choices=sorted(EXECUTION_CHOICES),
        default="simulated",
        help="Where the plan executes: in the simulation, on the semi-real joints, or "
        "on the real griplink servers.",
    )
    arguments = parser.parse_args()
    main(execution_type=EXECUTION_CHOICES[arguments.execution])


def show_announcements_in_terminal() -> None:
    """
    Announce the steps at info level on a handler of this module's logger.

    Other libraries, ROS among them, configure the root logger before the demonstration
    starts, so its announcements would be filtered out by the root's level if they only
    propagated there.
    """
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(handler)


def main(execution_type: ExecutionType = ExecutionType.KINEMATICALLY_SIMULATED) -> None:
    """
    Run the demonstration.

    :param execution_type: Whether to drive the real robot, the semi-real joints, or the
        simulation.
    """
    DAiSyGripperDemonstration(used_robot=DAiSy, execution_type=execution_type).run()


if __name__ == "__main__":
    run_from_command_line()
