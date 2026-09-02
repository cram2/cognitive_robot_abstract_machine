from dataclasses import dataclass

import numpy as np
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import Arms
from coraplex.failure_handling.attempt_budget import AttemptBudget
from coraplex.failure_handling.detectors import (
    BodyUnfetchableDetector,
    EndEffectorTargetDetector,
    NavigationGoalDetector,
)
from coraplex.failure_handling.factories import default_failure_handler
from coraplex.failure_handling.failure_handling_strategy import Propagate, RetryNode
from coraplex.failure_handling.strategies.navigation_recovery_strategy import (
    NavigationRecoveryStrategy,
)
from coraplex.failure_handling.strategies.retry_strategy import (
    EndEffectorRetryStrategy,
    MotionRetryStrategy,
    RetryStrategy,
)
from coraplex.failure_handling.strategies.underspecified_reparameterization_strategy import (
    UnderspecifiedReparameterizationStrategy,
)
from coraplex.plans.factories import code, execute_single
from coraplex.plans.failures import (
    BodyUnfetchable,
    EndEffectorDidNotReachTarget,
    MotionDidNotFinish,
    NavigationGoalNotReachedError,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction

from .conftest import child_of
from .test_failure_handler import HandledFailure

# %% stub strategies


@dataclass
class StubFailureRetryStrategy(RetryStrategy):
    """
    Retries the stub failure, so the bounding every retry strategy shares can be
    exercised without a world.
    """

    handled_failure_type = HandledFailure


# %% attempt budgets


def test_a_budget_grants_its_maximum_and_then_refuses():
    node = code(lambda: None)
    budget = AttemptBudget(maximum_attempts=2)

    assert [budget.grant(node) for _ in range(3)] == [True, True, False]


def test_budgets_are_kept_per_node():
    """
    A node that spent its attempts must not consume the attempts of another node, so a
    plan failing repeatedly in different places still recovers everywhere.
    """
    budget = AttemptBudget(maximum_attempts=1)
    exhausted_node = code(lambda: None)
    fresh_node = code(lambda: None)
    budget.grant(exhausted_node)

    assert budget.grant(exhausted_node) is False
    assert budget.grant(fresh_node) is True


# %% what a retry runs again


def test_a_retry_targets_the_enclosing_action_node():
    """
    The whole action is run again rather than the node that happened to raise, because a
    single motion of an action is rarely meaningful on its own.
    """
    action_node = execute_single(
        NavigateAction(target_location=Pose()), Context(world=None, robot=None)
    )
    failing_node = child_of(action_node)

    resolution = StubFailureRetryStrategy().resolve(HandledFailure(node=failing_node))

    assert isinstance(resolution, RetryNode)
    assert resolution.target_node is action_node


def test_a_retry_targets_the_failing_node_when_no_action_encloses_it(code_node):
    resolution = StubFailureRetryStrategy().resolve(HandledFailure(node=code_node))

    assert isinstance(resolution, RetryNode)
    assert resolution.target_node is code_node


# %% bounded retries


def test_exhausted_retries_propagate(code_node):
    """
    ``PlanNode.perform`` retries as long as a resolution reaches its target, so a
    strategy that never gives up would spin forever on a deterministic failure.
    """
    strategy = StubFailureRetryStrategy(
        attempt_budget=AttemptBudget(maximum_attempts=2)
    )

    resolutions = [strategy.resolve(HandledFailure(node=code_node)) for _ in range(3)]

    assert [type(resolution) for resolution in resolutions] == [
        RetryNode,
        RetryNode,
        Propagate,
    ]


def test_retries_spent_on_one_node_leave_another_node_untouched(code_node):
    strategy = StubFailureRetryStrategy(
        attempt_budget=AttemptBudget(maximum_attempts=1)
    )
    other_node = code(lambda: None)
    strategy.resolve(HandledFailure(node=code_node))

    assert isinstance(strategy.resolve(HandledFailure(node=code_node)), Propagate)
    assert isinstance(strategy.resolve(HandledFailure(node=other_node)), RetryNode)


# %% declared failure types


def test_the_retry_strategies_apply_only_to_the_failure_they_declare(
    code_node, immutable_model_world
):
    world, view, context = immutable_model_world
    motion_failure = MotionDidNotFinish(node=code_node, failed_motions=[])
    end_effector_failure = EndEffectorDidNotReachTarget(
        node=code_node, end_effector=view.left_arm.end_effector, target=Pose()
    )

    assert MotionRetryStrategy().applies(motion_failure)
    assert not MotionRetryStrategy().applies(end_effector_failure)
    assert EndEffectorRetryStrategy().applies(end_effector_failure)
    assert not EndEffectorRetryStrategy().applies(motion_failure)


# %% the shipped ensemble


def test_the_default_handler_carries_every_shipped_detector():
    detectors = default_failure_handler().refiner.failure_detectors

    assert {type(detector) for detector in detectors} == {
        NavigationGoalDetector,
        EndEffectorTargetDetector,
        BodyUnfetchableDetector,
    }


def test_the_default_handler_recovers_from_a_navigation_failure(code_node):
    failure = NavigationGoalNotReachedError(
        node=code_node, current_pose=Pose(), goal_pose=Pose()
    )

    strategy = default_failure_handler().most_specific_strategy(failure)

    assert isinstance(strategy, NavigationRecoveryStrategy)


def test_the_default_handler_retries_a_motion_failure_no_detector_recognised(code_node):
    failure = MotionDidNotFinish(node=code_node, failed_motions=[])

    strategy = default_failure_handler().most_specific_strategy(failure)

    assert isinstance(strategy, MotionRetryStrategy)


def test_the_default_handler_retries_an_end_effector_that_missed_its_target(
    immutable_model_world,
):
    world, view, context = immutable_model_world
    failure = EndEffectorDidNotReachTarget(
        node=code(lambda: None, context=context),
        end_effector=view.left_arm.end_effector,
        target=Pose(),
    )

    strategy = default_failure_handler().most_specific_strategy(failure)

    assert isinstance(strategy, EndEffectorRetryStrategy)


def test_the_default_handler_reparameterizes_an_unfetchable_body(
    immutable_model_world,
):
    """
    An object that cannot be grasped is answered by the baseline strategy, which asks an
    enclosing underspecified node for another candidate.
    """
    world, view, context = immutable_model_world
    failure = BodyUnfetchable(
        node=code(lambda: None, context=context), body=world.root, arm=Arms.LEFT
    )

    strategy = default_failure_handler().most_specific_strategy(failure)

    assert isinstance(strategy, UnderspecifiedReparameterizationStrategy)


# %% navigation recovery


def test_an_exhausted_navigation_recovery_propagates(code_node):
    """
    The budget is consulted before any pose is generated, so giving up costs no world
    iteration.
    """
    strategy = NavigationRecoveryStrategy(
        attempt_budget=AttemptBudget(maximum_attempts=0)
    )
    failure = NavigationGoalNotReachedError(
        node=code_node, current_pose=Pose(), goal_pose=Pose()
    )

    resolution = strategy.resolve(failure)

    assert isinstance(resolution, Propagate)
    assert resolution.failure is failure


def test_the_navigation_recovery_drives_to_a_regenerated_standing_pose(
    immutable_model_world,
):
    """
    Driving to the very pose the robot just failed to reach is no recovery, so the
    recovery navigates to a freshly generated standing pose instead.
    """
    world, view, context = immutable_model_world
    goal_pose = view.root.global_pose
    failure = NavigationGoalNotReachedError(
        node=code(lambda: None, context=context),
        current_pose=goal_pose,
        goal_pose=goal_pose,
    )

    recovery_plan = NavigationRecoveryStrategy().recovery_plan(failure)

    assert isinstance(recovery_plan, NavigateAction)
    assert not np.allclose(
        recovery_plan.target_location.to_np(), goal_pose.to_np(), atol=0.03
    )
