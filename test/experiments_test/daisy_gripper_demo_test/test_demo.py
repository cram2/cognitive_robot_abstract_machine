"""
Plan structure of the DAiSy gripper demonstration.

The demonstration is a plan about gripper motions, so what a run commands is decided
entirely at plan-building time: which gripper is commanded, which state, and whether the
two grippers move together. Execution itself needs a controller and is not covered here.
"""

from __future__ import annotations

import logging
from unittest.mock import Mock

import experiments.daisy_gripper_demo.demo as demo_module
import pytest
from coraplex.datastructures.enums import ExecutionType, VisualizationBackend
from coraplex.demonstrations import RobotDemonstrationRosSession
from coraplex.execution_environment import ExecutionEnvironment
from coraplex.language import CodeNode, ParallelNode, SequentialNode
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.plan_node import MotionNode, PlanNode
from experiments.daisy_gripper_demo.demo import (
    EXECUTION_CHOICES,
    PAUSE_AFTER_MOTION_SECONDS,
    STATE_VERBS,
    DAiSyGripperDemonstration,
)
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.robots.daisy import (
    DAiSy,
    DAiSyLeftGripper,
    DAiSyRightGripper,
)
from semantic_digital_twin.utils import daisy_installed

pytestmark = pytest.mark.skipif(not daisy_installed(), reason="DAiSy not installed")

DEMO_LOGGER_NAME = "experiments.daisy_gripper_demo.demo"


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
        if isinstance(child, MotionNode):
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
    pause, skipping the park action that homes the arms before the gripper steps.
    """
    children = [
        child
        for child in plan_node.children
        if isinstance(child, (CodeNode, MotionNode, ParallelNode))
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
        assert set(EXECUTION_CHOICES.values()) == {
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
    def test_every_step_announces_motions_then_pauses(self, demonstration_plan):
        """
        Every gripper step is wrapped in a code node announcing it and a code node
        pausing after it, so a run reports what happens next and settles between
        motions.
        """
        for step in steps(demonstration_plan):
            announcement, motion_or_parallel, pause = step
            assert isinstance(announcement, CodeNode)
            assert isinstance(pause, CodeNode)
            assert isinstance(motion_or_parallel, (MotionNode, ParallelNode))

    def test_twelve_steps_wrap_sixteen_motions(self, demonstration_plan):
        assert len(steps(demonstration_plan)) == 12
        assert len(gripper_commands(demonstration_plan)) == 16


class TestAnnouncements:
    def test_each_step_announces_its_motion_before_it(self, demonstration_plan, caplog):
        """
        Calling an announcement logs what the following motion commands, naming the
        gripper side and the state it is driven to.
        """
        caplog.set_level(logging.INFO, logger=DEMO_LOGGER_NAME)
        messages = []
        for step in steps(demonstration_plan):
            step[0].code()
            messages.append(caplog.records[-1].getMessage())

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
            expected.append(f"{STATE_VERBS[state_type].capitalize()} {subject}.")
        assert messages == expected


class TestPauses:
    def test_every_step_pauses_one_second_after_its_motion(
        self, demonstration_plan, monkeypatch
    ):
        """
        Calling a pause sleeps :data:`PAUSE_AFTER_MOTION_SECONDS`, once per step.
        """
        sleeps = []
        monkeypatch.setattr(
            demo_module.time, "sleep", lambda seconds: sleeps.append(seconds)
        )
        for step in steps(demonstration_plan):
            step[2].code()
        assert sleeps == [PAUSE_AFTER_MOTION_SECONDS] * len(steps(demonstration_plan))


class RecordingHandler(logging.Handler):
    """
    Logging handler that records the message of every record in a shared list.
    """

    def __init__(self, events: list):
        super().__init__()
        self.events = events

    def emit(self, record: logging.LogRecord) -> None:
        self.events.append(("announcement", record.getMessage()))


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


class TestSeparateExecution:
    def test_perform_executes_each_step_in_its_own_chart(
        self, demonstration_context, monkeypatch
    ):
        """
        Performing the plan runs every step's motions in their own motion state chart,
        with the two grippers of a parallel step sharing one chart.
        """
        plan = DAiSyGripperDemonstration(used_robot=DAiSy).build_plan(
            demonstration_context
        )
        monkeypatch.setattr(demo_module.time, "sleep", lambda seconds: None)
        motions_per_chart = []
        original_execute = GiskardExecutable.execute

        def recording_execute(self):
            motions_per_chart.append(len(self.motion_mappings))
            return original_execute(self)

        monkeypatch.setattr(GiskardExecutable, "execute", recording_execute)
        with ExecutionEnvironment(ExecutionType.NO_EXECUTION):
            plan.perform()
        # Code steps build empty executables that execute nothing; only the steps'
        # motion charts count, the first being the park step's chart.
        assert [n for n in motions_per_chart if n] == [
            1,  # park arms
            1,  # left open
            1,  # left close
            1,  # right open
            1,  # right close
            2,  # both open
            2,  # both close
            1,  # left flex-open
            1,  # left flex-close
            1,  # right flex-open
            1,  # right flex-close
            2,  # both flex-open
            2,  # both flex-close
        ]

    def test_announcements_interleave_with_motion_execution(
        self, demonstration_context, monkeypatch
    ):
        """
        Each announcement runs right before its own motion chart, so the logged
        announcement and the executed chart alternate instead of every announcement
        playing before any motion.

        The park step runs first without an announcement.
        """
        plan = DAiSyGripperDemonstration(used_robot=DAiSy).build_plan(
            demonstration_context
        )
        monkeypatch.setattr(demo_module.time, "sleep", lambda seconds: None)
        events: list = []
        handler = RecordingHandler(events)
        original_execute = GiskardExecutable.execute

        def recording_execute(self):
            if self.motion_mappings:
                events.append(("chart", len(self.motion_mappings)))
            return original_execute(self)

        monkeypatch.setattr(GiskardExecutable, "execute", recording_execute)
        demo_logger = logging.getLogger(DEMO_LOGGER_NAME)
        demo_logger.addHandler(handler)
        demo_logger.setLevel(logging.INFO)
        try:
            with ExecutionEnvironment(ExecutionType.NO_EXECUTION):
                plan.perform()
        finally:
            demo_logger.removeHandler(handler)
        kinds = [kind for kind, _ in events]
        assert kinds == ["chart"] + ["announcement", "chart"] * 12
