from coraplex.plans.plan_node import ActionNode, MotionNode
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.motions.gripper import MoveToolCenterPointMotion

from .conftest import PerformedMotionFailure

# %% attribution of a motion that does not reach its goal


def test_a_failing_motion_is_attributed_to_a_motion_node(attributed_motion_failure):
    outcome: PerformedMotionFailure = attributed_motion_failure

    assert isinstance(outcome.failure.node, MotionNode)
    assert outcome.failure.node in outcome.root.plan.nodes


def test_the_stuck_motion_itself_is_blamed(attributed_motion_failure):
    """
    The robot stands out of reach, so the reach towards the pre-grasp pose is the motion
    that cannot finish; the gripper motions after it must not be blamed.
    """
    outcome: PerformedMotionFailure = attributed_motion_failure
    blamed_motion = outcome.failure.node.designator

    assert isinstance(blamed_motion, MoveToolCenterPointMotion)
    assert blamed_motion.allow_gripper_collision is False


def test_an_attributed_failure_resolves_the_action_owning_the_motion(
    attributed_motion_failure,
):
    """
    The resolved action is the innermost one, so a failure inside a composite action
    names the sub-action that owns the motion rather than the whole composite.
    """
    outcome: PerformedMotionFailure = attributed_motion_failure
    action_node = outcome.failure.action_node

    assert isinstance(action_node, ActionNode)
    assert action_node in outcome.failure.node.path
    assert any(
        isinstance(ancestor, ActionNode) and isinstance(ancestor.action, PickUpAction)
        for ancestor in outcome.failure.node.path
    )


def test_an_attributed_failure_resolves_its_context(attributed_motion_failure):
    outcome: PerformedMotionFailure = attributed_motion_failure

    assert outcome.failure.context is outcome.context


def test_the_failed_motions_are_kept_for_diagnostics(attributed_motion_failure):
    outcome: PerformedMotionFailure = attributed_motion_failure

    assert outcome.failure.failed_motions
