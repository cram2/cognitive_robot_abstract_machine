"""
DAiSy opens and closes both grippers, first each one alone, then both at once.

The plan brings the grippers into their preset ``OPEN``/``CLOSE`` states first and into
the flex ``FLEXOPEN``/``FLEXCLOSE`` states afterwards, so both kinds of griplink command
are exercised in one run. It touches no objects, so it can be repeated against the same
scene.

The executor decides what a gripper action does: in simulation the fingers follow
the goals in this process, a run against a standalone controller sends the same goals to
a Giskard that moves the joints without hardware, and a run on the real robot sends
``Grip``/``Release`` and ``Flexgrip``/``Flexrelease`` to the griplink servers of the
grippers.

Run it as a script, choosing where it runs with ``--execution``::

    python3 -m experiments.daisy_gripper_demo.demo --execution simulated

Runs against a controller need a running DAiSy controller::

    ros2 launch giskardpy_ros giskardpy_daisy_standalone.launch.py

A real run fetches its world from the running controller and needs the griplink driver
of the grippers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from coraplex.demonstrations import RobotDemonstration
from coraplex.griplink_grippers import RealDAiSyPlanExecutor
from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.executors import (
    KinematicallySimulatedPlanExecutor,
    PlanExecutor,
    RealRobotPlanExecutor,
)
from coraplex.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    SetGripperAction,
)
from cramph.composites import Parallel, Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.monitors import CountSimulationTimeSeconds, Print
from cramph.node import StatechartNode
from cramph.statechart import Statechart
from krrood.entity_query_language.verbalization.morphology import present_participle
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.daisy import DAiSy
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from semantic_digital_twin.world import World

# %% command line choices


class ExecutionChoice(StrEnum):
    """
    The values of the ``--execution`` switch, each selecting where the plan runs.
    """

    SIMULATED = "simulated"
    """
    Perform the plan in the kinematic simulation of this process.
    """

    STANDALONE = "standalone"
    """
    Send the plan to a running standalone controller, which moves the joints itself.
    """

    REAL = "real"
    """
    Send the plan to the real robot, the gripper commands to the griplink servers of the
    grippers.
    """

    @property
    def executor_type(self) -> type[PlanExecutor]:
        """
        :return: The executor running the plan for this choice.
        """
        match self:
            case ExecutionChoice.SIMULATED:
                return KinematicallySimulatedPlanExecutor
            case ExecutionChoice.STANDALONE:
                return RealRobotPlanExecutor
            case ExecutionChoice.REAL:
                return RealDAiSyPlanExecutor


# %% demonstration


@dataclass
class DAiSyGripperDemonstration(RobotDemonstration):
    """
    DAiSy opens and closes its grippers through the full preset/flex and alone/together
    matrix.
    """

    ros_node_name: str = "daisy_gripper_demo_node"

    pause_after_motion: timedelta = timedelta(seconds=1)
    """
    How long a run waits after a gripper motion before the next one starts, letting the
    fingers settle into the state they just reached.
    """

    def scene_specification(self) -> WorldSpecification:
        """
        :return: A world holding nothing, since the gripper motions touch no objects.
        """
        return WorldSpecification()

    def robot_specification(self) -> RobotSpecification:
        """
        :return: The DAiSy at its start position.
        """
        return RobotSpecification(
            semantic_annotation_type=self.used_robot,
            odom_T_robot_start=HomogeneousTransformationMatrix.from_xyz_rpy(1, 1),
        )

    def is_scene_populated(self, world: World) -> bool:
        return True

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        """
        Give the plan the DAiSy in ``world``.
        """
        return [RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0])]

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Open and close both grippers: each one alone, then both together, with the
        preset states first and the flex states afterwards.
        """
        daisy = context.require_extension(RobotAccess).robot
        left_gripper = daisy.left_arm.end_effector
        right_gripper = daisy.right_arm.end_effector
        steps = [ParkArmsAction(arms=daisy.all_arms)]
        for opening, closing in (
            (GripperState.OPEN, GripperState.CLOSE),
            (GripperState.FLEXOPEN, GripperState.FLEXCLOSE),
        ):
            for subject, gripper in (
                ("the left gripper", left_gripper),
                ("the right gripper", right_gripper),
            ):
                for state in (opening, closing):
                    steps.extend(self.create_steps_moving(subject, [gripper], state))
            for state in (opening, closing):
                steps.extend(
                    self.create_steps_moving(
                        "both grippers", [left_gripper, right_gripper], state
                    )
                )
        statechart = Statechart(context=context)
        statechart.add_node(Sequence(nodes=steps))
        return statechart

    def create_steps_moving(
        self, subject: str, grippers: list[EndEffector], state: GripperState
    ) -> list[StatechartNode]:
        """
        :param subject: How the announcement names `grippers`.
        :param grippers: The grippers to move together.
        :param state: The state to bring each of them into.
        :return: New steps announcing the motion, moving `grippers` into the default
            configuration of `state`, and waiting :attr:`pause_after_motion` after it.
        """
        moves = [
            SetGripperAction(configuration=gripper.default_configuration(state))
            for gripper in grippers
        ]
        return [
            Print(message=self.announcement(subject, state)),
            moves[0] if len(moves) == 1 else Parallel(nodes=moves),
            CountSimulationTimeSeconds(seconds=self.pause_after_motion.total_seconds()),
        ]

    @staticmethod
    def announcement(subject: str, state: GripperState) -> str:
        """
        :param subject: How the announcement names the grippers the motion moves.
        :param state: The state the announced motion brings them into.
        :return: What the motion does, such as ``Flex-closing both grippers.``.
        """
        verb = state.name.lower().replace("flex", "flex-")
        return f"{present_participle(verb).capitalize()} {subject}."


# %% running it


def main(
    executor_type: type[PlanExecutor] = KinematicallySimulatedPlanExecutor,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it drives
        the real robot, a standalone controller or the simulation.
    """
    DAiSyGripperDemonstration(used_robot=DAiSy, executor_type=executor_type).run()


def run_from_command_line() -> None:
    """
    Parse the ``--execution`` switch and run the demonstration.
    """
    parser = argparse.ArgumentParser(description="DAiSy gripper open/close demo.")
    parser.add_argument(
        "--execution",
        type=ExecutionChoice,
        choices=list(ExecutionChoice),
        default=ExecutionChoice.SIMULATED,
        help="Where the plan runs: in the simulation, on a standalone controller, or "
        "on the real robot.",
    )
    main(executor_type=parser.parse_args().execution.executor_type)


if __name__ == "__main__":
    run_from_command_line()
