import numpy as np
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose

from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.plans.failures import MotionDidNotFinish
from coraplex.failure_handling.detectors import (
    BodyUnfetchableDetector,
    EndEffectorTargetDetector,
    NavigationGoalDetector,
)
from coraplex.failure_handling.failure_refiner import FailureRefiner
from coraplex.plans.factories import code, execute_single
from coraplex.plans.failures import (
    BodyUnfetchable,
    EndEffectorDidNotReachTarget,
    NavigationGoalNotReachedError,
    PlanFailure,
)
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import MoveManipulatorAction

from .conftest import milk_pick_up, move_the_robot_out_of_reach

# %% world setup


def move_the_robot_within_reach(view) -> None:
    """
    Drive the robot to where it can grasp the milk, the pose at which ``PickUpAction``'s
    own reachability pre-condition holds.
    """
    view.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.9, 1.4, 0
    )


# %% failures to refine


def navigation_failure(context) -> MotionDidNotFinish:
    """
    :return: A failure of a navigation whose destination the robot is not standing at.
    """
    node = execute_single(NavigateAction(target_location=Pose()), context)
    return MotionDidNotFinish(node=node, failed_motions=[])


def arrived_navigation_failure(context) -> MotionDidNotFinish:
    """
    :return: A failure of a navigation whose destination the robot already stands at.
    """
    node = execute_single(
        NavigateAction(target_location=context.robot.root.global_pose), context
    )
    return MotionDidNotFinish(node=node, failed_motions=[])


def manipulation_failure(world, view, context) -> MotionDidNotFinish:
    action = milk_pick_up(world, view)
    return MotionDidNotFinish(node=execute_single(action, context), failed_motions=[])


def end_effector_failure(view, context) -> MotionDidNotFinish:
    """
    :return: A failure of a motion whose end effector is not at the pose it was sent to.
    """
    action = MoveManipulatorAction(
        end_effector=view.left_arm.end_effector, target_pose=Pose()
    )
    return MotionDidNotFinish(node=execute_single(action, context), failed_motions=[])


def arrived_end_effector_failure(view, context) -> MotionDidNotFinish:
    """
    :return: A failure of a motion whose end effector already holds the pose it was sent
        to.
    """
    end_effector = view.left_arm.end_effector
    action = MoveManipulatorAction(
        end_effector=end_effector, target_pose=end_effector.tool_frame.global_pose
    )
    return MotionDidNotFinish(node=execute_single(action, context), failed_motions=[])


def transport_failure(world, view, context) -> MotionDidNotFinish:
    """
    :return: A failure of an action that carries the parameters of two detectors at
        once, so the number of parameters decides which one is asked first.
    """
    action = TransportAction(
        target_object=world.get_semantic_annotations_by_type(Milk)[0],
        arm=Arms.LEFT,
        grasp_description=GraspDescription(
            ApproachDirection.FRONT,
            VerticalAlignment.NoAlignment,
            view.left_arm.end_effector,
        ),
        target_location=Pose(),
    )
    return MotionDidNotFinish(node=execute_single(action, context), failed_motions=[])


# %% navigation


def test_the_navigation_detector_applies_to_a_navigation_action(immutable_model_world):
    world, view, context = immutable_model_world

    assert NavigationGoalDetector().applies(navigation_failure(context))


def test_the_navigation_detector_reports_the_goal_it_did_not_reach(
    immutable_model_world,
):
    world, view, context = immutable_model_world
    failure = navigation_failure(context)

    refined = NavigationGoalDetector().detect(failure)

    assert isinstance(refined, NavigationGoalNotReachedError)
    assert refined.goal_pose is failure.action_node.action.target_location


def test_the_navigation_detector_reports_where_the_robot_ended_up(
    immutable_model_world,
):
    world, view, context = immutable_model_world

    refined = NavigationGoalDetector().detect(navigation_failure(context))

    assert np.allclose(
        refined.current_pose.to_np(), context.robot.root.global_pose.to_np()
    )


# %% end effector


def test_the_end_effector_detector_applies_to_an_end_effector_action(
    immutable_model_world,
):
    world, view, context = immutable_model_world

    assert EndEffectorTargetDetector().applies(end_effector_failure(view, context))


def test_the_end_effector_detector_reports_the_end_effector_and_its_target(
    immutable_model_world,
):
    world, view, context = immutable_model_world
    failure = end_effector_failure(view, context)

    refined = EndEffectorTargetDetector().detect(failure)

    assert isinstance(refined, EndEffectorDidNotReachTarget)
    assert refined.end_effector is failure.action_node.action.end_effector
    assert refined.target is failure.action_node.action.target_pose


# %% manipulation


def test_the_body_unfetchable_detector_applies_to_a_manipulation_action(
    immutable_model_world,
):
    world, view, context = immutable_model_world

    assert BodyUnfetchableDetector().applies(manipulation_failure(world, view, context))


def test_the_body_unfetchable_detector_reports_the_body_and_the_arm(
    mutable_model_world,
):
    world, view, context = mutable_model_world
    move_the_robot_out_of_reach(view)
    failure = manipulation_failure(world, view, context)

    refined = BodyUnfetchableDetector().detect(failure)

    assert isinstance(refined, BodyUnfetchable)
    assert refined.body is failure.action_node.action.target_object.root
    assert refined.arm is Arms.LEFT


# %% detectors decline what they cannot confirm


def test_the_navigation_detector_declines_when_the_robot_stands_at_its_goal(
    immutable_model_world,
):
    world, view, context = immutable_model_world
    failure = arrived_navigation_failure(context)

    assert NavigationGoalDetector().detect(failure) is failure


def test_the_end_effector_detector_declines_when_the_end_effector_is_at_its_target(
    immutable_model_world,
):
    world, view, context = immutable_model_world
    failure = arrived_end_effector_failure(view, context)

    assert EndEffectorTargetDetector().detect(failure) is failure


def test_the_body_unfetchable_detector_declines_when_the_body_is_reachable(
    mutable_model_world,
):
    world, view, context = mutable_model_world
    move_the_robot_within_reach(view)
    failure = manipulation_failure(world, view, context)

    assert BodyUnfetchableDetector().detect(failure) is failure


def test_a_declined_failure_is_left_unrefined_by_the_ensemble(immutable_model_world):
    world, view, context = immutable_model_world
    refiner = FailureRefiner(failure_detectors=[NavigationGoalDetector()])
    failure = arrived_navigation_failure(context)

    assert refiner.refine(failure) is failure


# %% the refined failure is attributed like the one it came from


def test_a_refined_failure_keeps_the_node_of_the_failure_it_came_from(
    immutable_model_world,
):
    world, view, context = immutable_model_world
    failure = navigation_failure(context)

    refined = NavigationGoalDetector().detect(failure)

    assert refined.node is failure.node


# %% gating


def test_a_detector_does_not_apply_to_an_action_without_its_mixins(
    immutable_model_world,
):
    world, view, context = immutable_model_world

    assert not NavigationGoalDetector().applies(
        manipulation_failure(world, view, context)
    )


def test_a_detector_does_not_apply_to_another_failure_type(immutable_model_world):
    world, view, context = immutable_model_world
    node = execute_single(NavigateAction(target_location=Pose()), context)

    assert not NavigationGoalDetector().applies(PlanFailure(node=node))


def test_a_detector_does_not_apply_without_an_action_node(immutable_model_world):
    world, view, context = immutable_model_world
    node = code(lambda: None, context)

    assert not NavigationGoalDetector().applies(
        MotionDidNotFinish(node=node, failed_motions=[])
    )


# %% the ensemble


def test_the_ensemble_refines_navigation_and_manipulation_differently(
    mutable_model_world,
):
    world, view, context = mutable_model_world
    move_the_robot_out_of_reach(view)
    refiner = FailureRefiner(
        failure_detectors=[
            NavigationGoalDetector(),
            EndEffectorTargetDetector(),
            BodyUnfetchableDetector(),
        ]
    )

    navigation = refiner.refine(navigation_failure(context))
    manipulation = refiner.refine(manipulation_failure(world, view, context))

    assert isinstance(navigation, NavigationGoalNotReachedError)
    assert isinstance(manipulation, BodyUnfetchable)


def test_the_ensemble_records_where_a_refined_failure_came_from(immutable_model_world):
    world, view, context = immutable_model_world
    refiner = FailureRefiner(failure_detectors=[NavigationGoalDetector()])
    failure = navigation_failure(context)

    refined = refiner.refine(failure)

    assert refined.refined_from is failure


def test_the_detector_requiring_more_mixins_is_asked_first(mutable_model_world):
    """
    Transporting carries both a target location and a graspable object with an arm, so
    the detector requiring more of those parameters is the more specific one.
    """
    world, view, context = mutable_model_world
    move_the_robot_out_of_reach(view)
    refiner = FailureRefiner(
        failure_detectors=[NavigationGoalDetector(), BodyUnfetchableDetector()]
    )

    refined = refiner.refine(transport_failure(world, view, context))

    assert isinstance(refined, BodyUnfetchable)


def test_a_declining_detector_lets_the_next_most_specific_one_try(mutable_model_world):
    """
    The object of a transport is reachable, so the more specific detector declines and
    the navigation detector still gets to report that the robot never arrived.
    """
    world, view, context = mutable_model_world
    move_the_robot_within_reach(view)
    refiner = FailureRefiner(
        failure_detectors=[NavigationGoalDetector(), BodyUnfetchableDetector()]
    )

    refined = refiner.refine(transport_failure(world, view, context))

    assert isinstance(refined, NavigationGoalNotReachedError)
