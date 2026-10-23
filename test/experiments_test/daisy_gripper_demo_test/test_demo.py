"""
Plan structure of the DAiSy gripper demonstration.

The demonstration is a plan about gripper motions, so what a run commands is decided
entirely at plan-building time: which gripper is commanded, which state, and whether the
two grippers move together. Execution itself needs a controller and is not covered here.
"""

from __future__ import annotations

from unittest.mock import Mock

import experiments.daisy_gripper_demo.demo as demo_module
import pytest
from coraplex.datastructures.enums import ExecutionType, VisualizationBackend
from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.execution_environment import kinematically_simulated_robot
from coraplex.language import ParallelNode, SequentialNode
from coraplex.plans.plan_node import MotionNode, PlanNode
from coraplex.robot_plans import MoveGripperMotion, PrintMotion, WaitMotion
from experiments.daisy_gripper_demo.demo import (
    DAiSyGripperDemonstration,
    ExecutionChoice,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.daisy import (
    DAiSy,
    DAiSyLeftGripper,
    DAiSyRightGripper,
)
from semantic_digital_twin.utils import daisy_installed

pytestmark = pytest.mark.skipif(not daisy_installed(), reason="DAiSy not installed")


@pytest.fixture(scope="module")
def demonstration_world():
    """
    The world a simulated run of the demonstration builds.
    """
    demonstration = DAiSyGripperDemonstration(used_robot=DAiSy)
    return demonstration.build_simulated_world()


@pytest.fixture(scope="module")
def demonstration_context(demonstration_world):
    """
    The plan context a simulated run of the demonstration builds.
    """
    demonstration = DAiSyGripperDemonstration(used_robot=DAiSy)
    return demonstration.build_context(demonstration_world)


@pytest.fixture(scope="module")
def demonstration_plan(demonstration_context):
    """
    The plan the demonstration performs.
    """
    demonstration = DAiSyGripperDemonstration(used_robot=DAiSy)
    return demonstration.build_plan(demonstration_context)


def gripper_commands(plan_node: PlanNode):
    """
    Flatten the plan depth-first into the gripper each leaf motion commands, its state,
    and whether it executes together with the other gripper.
    """
    commands = []
    for child in plan_node.children:
        if isinstance(child, MotionNode) and isinstance(
            child.designator, MoveGripperMotion
        ):
            specification = child.designator.configuration
            commands.append(
                (
                    specification.end_effector,
                    specification.joint_state.state_type,
                    False,
                )
            )
        elif isinstance(child, ParallelNode):
            for grandchild in child.children:
                specification = grandchild.designator.configuration
                commands.append(
                    (
                        specification.end_effector,
                        specification.joint_state.state_type,
                        True,
                    )
                )
    return commands


def steps(plan_node: PlanNode):
    """
    Group the plan's children into steps of announcement, motion or parallel plan, and
    wait, skipping the park action that homes the arms before the gripper steps.
    """
    children = [
        child
        for child in plan_node.children
        if isinstance(child, (MotionNode, ParallelNode))
    ]
    assert len(children) % 3 == 0
    return [
        children[position : position + 3] for position in range(0, len(children), 3)
    ]


class TestSimulatedWorld:
    def test_world_holds_daisy(self, demonstration_world):
        daisy = demonstration_world.get_semantic_annotations_by_type(DAiSy)[0]
        assert isinstance(daisy, DAiSy)

    def test_scene_needs_no_population(self, demonstration_world):
        assert (
            DAiSyGripperDemonstration(used_robot=DAiSy).is_scene_populated(
                demonstration_world
            )
            is True
        )


class TestBuildContext:
    def test_context_is_built_around_daisy(self, demonstration_context):
        assert isinstance(demonstration_context.robot, DAiSy)
        assert demonstration_context.alternative_motion_mappings


class TestExecutionSwitch:
    def test_switch_covers_all_execution_types(self):
        assert {choice.execution_type for choice in ExecutionChoice} == {
            ExecutionType.KINEMATICALLY_SIMULATED,
            ExecutionType.SEMI_REAL,
            ExecutionType.REAL,
        }


class TestPlan:
    def test_plan_is_sequential(self, demonstration_plan):
        assert isinstance(demonstration_plan, SequentialNode)

    def test_motion_order(self, demonstration_plan):
        """
        Each gripper is opened and closed alone, then both are commanded at once, and
        the preset states are commanded before the flex states.
        """
        side_by_state = [
            (
                type(end_effector),
                state_type,
                in_parallel,
            )
            for end_effector, state_type, in_parallel in gripper_commands(
                demonstration_plan
            )
        ]
        assert side_by_state == [
            # preset, each gripper individually
            (DAiSyLeftGripper, GripperState.OPEN, False),
            (DAiSyLeftGripper, GripperState.CLOSE, False),
            (DAiSyRightGripper, GripperState.OPEN, False),
            (DAiSyRightGripper, GripperState.CLOSE, False),
            # preset, both at the same time
            (DAiSyLeftGripper, GripperState.OPEN, True),
            (DAiSyRightGripper, GripperState.OPEN, True),
            (DAiSyLeftGripper, GripperState.CLOSE, True),
            (DAiSyRightGripper, GripperState.CLOSE, True),
            # flex, each gripper individually
            (DAiSyLeftGripper, GripperState.FLEXOPEN, False),
            (DAiSyLeftGripper, GripperState.FLEXCLOSE, False),
            (DAiSyRightGripper, GripperState.FLEXOPEN, False),
            (DAiSyRightGripper, GripperState.FLEXCLOSE, False),
            # flex, both at the same time
            (DAiSyLeftGripper, GripperState.FLEXOPEN, True),
            (DAiSyRightGripper, GripperState.FLEXOPEN, True),
            (DAiSyLeftGripper, GripperState.FLEXCLOSE, True),
            (DAiSyRightGripper, GripperState.FLEXCLOSE, True),
        ]

    def test_both_grippers_execute_in_parallel_plans(self, demonstration_plan):
        """
        The "both at the same time" steps are parallel plans of exactly the two gripper
        motions, not a sequential sub-plan.
        """
        parallel_steps = [
            child
            for child in demonstration_plan.children
            if isinstance(child, ParallelNode)
        ]
        assert len(parallel_steps) == 4
        for step in parallel_steps:
            assert len(step.children) == 2
            assert all(isinstance(child, MotionNode) for child in step.children)


class TestStepStructure:
    def test_every_step_announces_motions_then_waits(self, demonstration_plan):
        """
        Every gripper step is wrapped in a motion announcing it and a motion waiting
        after it, so a run reports what happens next and settles between motions.
        """
        for announcement, motion_or_parallel, wait in steps(demonstration_plan):
            assert isinstance(announcement.designator, PrintMotion)
            assert isinstance(wait.designator, WaitMotion)
            assert isinstance(motion_or_parallel, (MotionNode, ParallelNode))

    def test_twelve_steps_wrap_sixteen_motions(self, demonstration_plan):
        assert len(steps(demonstration_plan)) == 12
        assert len(gripper_commands(demonstration_plan)) == 16


class TestAnnouncements:
    def test_each_step_announces_its_motion_before_it(self, demonstration_plan):
        """
        Each announcement names the gripper side and the state the following motion
        drives it to.
        """
        expected_verbs = {
            GripperState.OPEN: "Opening",
            GripperState.CLOSE: "Closing",
            GripperState.FLEXOPEN: "Flex-opening",
            GripperState.FLEXCLOSE: "Flex-closing",
        }
        expected = []
        for step in steps(demonstration_plan):
            motion_or_parallel = step[1]
            if isinstance(motion_or_parallel, MotionNode):
                specification = motion_or_parallel.designator.configuration
                side = (
                    "left"
                    if isinstance(specification.end_effector, DAiSyLeftGripper)
                    else "right"
                )
                subject = f"the {side} gripper"
            else:
                specification = motion_or_parallel.children[0].designator.configuration
                subject = "both grippers"
            state_type = specification.joint_state.state_type
            expected.append(f"{expected_verbs[state_type]} {subject}.")
        assert [
            step[0].designator.message for step in steps(demonstration_plan)
        ] == expected


class TestWaits:
    def test_every_step_waits_the_configured_time_after_its_motion(
        self, demonstration_plan
    ):
        pause_seconds = DAiSyGripperDemonstration(
            used_robot=DAiSy
        ).pause_after_motion_seconds
        assert [step[2].designator.seconds for step in steps(demonstration_plan)] == [
            pause_seconds
        ] * len(steps(demonstration_plan))


class TestSemiRealWorld:
    def test_semi_real_acts_on_the_fetched_controller_world(self, monkeypatch):
        """
        A semi-real run does not build its own world but fetches the world of the
        running controller, so the motion goals it sends name entities the controller
        knows, and it synchronizes that world with the controller without publishing a
        second rendering of it.
        """
        controller_world = DAiSyGripperDemonstration(
            used_robot=DAiSy
        ).build_simulated_world()
        session = Mock()
        session.fetch_world.return_value = controller_world
        monkeypatch.setattr(
            RobotDemonstrationRosSession, "start", staticmethod(lambda name: session)
        )
        synchronizer = Mock()
        monkeypatch.setattr(demo_module, "WorldSynchronizer", synchronizer)

        demonstration = DAiSyGripperDemonstration(
            used_robot=DAiSy,
            execution_type=ExecutionType.SEMI_REAL,
            default_visualization_backend=VisualizationBackend.NONE,
        )
        world = demonstration.acquire_world()

        assert world is controller_world
        synchronizer.assert_called_once_with(_world=controller_world, node=session.node)
        assert demonstration.visualization is None


class TestSimulatedRun:
    def test_plan_performs_in_the_simulation(self, demonstration_context):
        """
        The whole plan, announcements and waits included, runs through the kinematic
        simulation as one motion chart without failing.
        """
        plan = DAiSyGripperDemonstration(used_robot=DAiSy).build_plan(
            demonstration_context
        )
        with kinematically_simulated_robot:
            plan.perform()
