from dataclasses import dataclass, field

import numpy as np
import pytest
from semantic_digital_twin.spatial_types.spatial_types import Pose
from typing_extensions import Callable, List, Optional

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import TaskStatus
from coraplex.execution_environment import simulated_robot
from coraplex.failure_handling.factories import default_failure_handler
from coraplex.language import SequentialNode
from coraplex.plans.executables import GiskardExecutable
from coraplex.plans.factories import sequential
from coraplex.plans.failures import MotionDidNotFinish, NavigationGoalNotReachedError
from coraplex.plans.plan_node import MotionNode
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.motions.navigation import MoveMotion

# %% a navigation that stops short once


@dataclass
class NavigationChartFailingOnce:
    """
    Stands in for a navigation that does not reach its goal.

    The first motion state chart driving the robot raises the failure a chart that ran
    out of control cycles would raise, attributed to the very motion node that did not
    finish. Every later chart runs for real, so the plan can recover.
    """

    original_execute: Callable[[GiskardExecutable], None]
    """
    The chart execution this stand-in replaces and delegates to.
    """

    remaining_failures: int = 1
    """
    How many navigation charts still fail instead of running.
    """

    executed_poses: List[Pose] = field(default_factory=list, init=False)
    """
    The robot pose at the start of every chart execution, in execution order.
    """

    blamed_node: Optional[MotionNode] = field(default=None, init=False)
    """
    The motion node the failure was attributed to.
    """

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Let every chart execution run through this stand-in for one test.

        :param monkeypatch: The patcher that restores chart execution afterwards.
        """
        monkeypatch.setattr(
            GiskardExecutable, "execute", lambda executable: self.execute(executable)
        )

    def execute(self, executable: GiskardExecutable) -> None:
        """
        Run the chart, or fail it if it is the navigation that is meant to stop short.

        :param executable: The chart that is about to run.
        :raises MotionDidNotFinish: For the sabotaged navigation charts.
        """
        self.executed_poses.append(executable.context.robot.root.global_pose)
        navigation_node = self.navigation_node(executable)
        if navigation_node is None or self.remaining_failures == 0:
            self.original_execute(executable)
            return
        self.remaining_failures -= 1
        self.blamed_node = navigation_node
        raise navigation_node.did_not_finish_failure([])

    @staticmethod
    def navigation_node(executable: GiskardExecutable) -> Optional[MotionNode]:
        """
        :param executable: The chart to search.
        :return: The node of the chart that drives the robot, or None if the chart does
            not drive at all.
        """
        for motion_node in executable.motion_mappings:
            if isinstance(motion_node.motion, MoveMotion):
                return motion_node
        return None


@dataclass
class RecoveredNavigation:
    """
    The outcome of one navigation that failed, recovered and then succeeded, shared by
    every test that inspects it.
    """

    root: SequentialNode
    """
    The plan root that was performed.
    """

    goal_pose: Pose
    """
    The pose the plan navigated to.
    """

    sabotage: NavigationChartFailingOnce
    """
    The stand-in that failed the first navigation and recorded what ran.
    """

    context: Context
    """
    The context the plan performed in.
    """


@pytest.fixture
def recovered_navigation(mutable_model_world, monkeypatch) -> RecoveredNavigation:
    """
    Perform a navigation whose first attempt does not reach its goal, with the shipped
    failure handler in charge of getting the robot there anyway.
    """
    world, view, context = mutable_model_world
    context.failure_handler = default_failure_handler()
    goal_pose = Pose.from_xyz_rpy(0.3, -1.3, 0, reference_frame=world.root)
    root = sequential([NavigateAction(target_location=goal_pose)], context)
    sabotage = NavigationChartFailingOnce(original_execute=GiskardExecutable.execute)
    sabotage.install(monkeypatch)

    with simulated_robot:
        root.perform()

    return RecoveredNavigation(
        root=root, goal_pose=goal_pose, sabotage=sabotage, context=context
    )


# %% the plan survives the failure


def test_a_recovered_navigation_ends_successfully(recovered_navigation):
    assert recovered_navigation.root.status == TaskStatus.SUCCEEDED


def test_the_robot_stands_at_the_goal_it_first_missed(recovered_navigation):
    robot_pose = recovered_navigation.context.robot.root.global_pose

    assert np.allclose(
        robot_pose.to_np(), recovered_navigation.goal_pose.to_np(), atol=0.03
    )


# %% what the handler did


def test_the_motion_failure_was_refined_into_a_navigation_failure(recovered_navigation):
    """
    The node that could not finish records the refined failure, so the plan reports what
    went wrong rather than that some motion stopped.
    """
    reason = recovered_navigation.sabotage.blamed_node.reason

    assert isinstance(reason, NavigationGoalNotReachedError)
    assert isinstance(reason.refined_from, MotionDidNotFinish)


def test_the_recovery_plan_drove_the_robot_before_the_retry(recovered_navigation):
    """
    Three charts run: the navigation that stops short, the recovery navigation, and the
    retry - which starts from somewhere else than the failed attempt did, because the
    recovery moved the robot.
    """
    executed_poses = recovered_navigation.sabotage.executed_poses

    assert len(executed_poses) == 3
    assert not np.allclose(
        executed_poses[2].to_np(), executed_poses[0].to_np(), atol=0.03
    )
