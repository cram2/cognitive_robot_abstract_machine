import numpy as np
import pytest

from coraplex.alternative_motion_mapping import AlternativeMotion
from coraplex.datastructures.dataclasses import Context, MotionToleranceConfig
from coraplex.datastructures.enums import Arms, ExecutionType
from coraplex.exceptions import TipLinkDoesNotMatchAnyArm
from coraplex.execution_environment import ExecutionEnvironment, simulated_robot
from coraplex.locations import pose_validator
from coraplex.locations.pose_validator import (
    IsGraspReachableBy,
    IsObjectReachableBy,
    IsReachableBy,
    AreReachableBy,
)
from coraplex.robot_plans import MoveToolCenterPointMotion
from giskardpy.motion_statechart.exceptions import NoProgressError
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.monitors.progress_monitors import StillProgressing
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.goals.collision_avoidance import (
    ExternalCollisionAvoidance,
    SelfCollisionAvoidance,
    UpdateTemporaryCollisionRules,
)
from krrood.entity_query_language.factories import evaluate_condition
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.view_manager import ViewManager
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowSelfCollisions,
)
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Handle,
    Milk,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3


def test_pose_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose = Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)

    assert IsReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose=pose,
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )


def test_pose_not_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose = Pose(Point3.from_iterable([2.3, 2, 1]), reference_frame=world.root)

    assert not IsReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose=pose,
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )


def test_pose_sequence_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = Pose(Point3.from_iterable([1.6, 1.4, 1]), reference_frame=world.root)
    pose2 = Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)
    pose3 = Pose(Point3.from_iterable([1.7, 1.4, 1.1]), reference_frame=world.root)

    assert AreReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose_sequence=[pose1, pose2, pose3],
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )


def test_pose_sequence_not_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = Pose(Point3.from_iterable([2.6, 1.4, 1]), reference_frame=world.root)
    pose2 = Pose(Point3.from_iterable([2.7, 1.4, 1]), reference_frame=world.root)
    pose3 = Pose(Point3.from_iterable([2.7, 1.4, 1.1]), reference_frame=world.root)

    assert not AreReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose_sequence=[pose1, pose2, pose3],
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )


class _MoveTcpAlternativeForPr2(MoveToolCenterPointMotion, AlternativeMotion[PR2]):
    """
    Minimal alternative used to exercise the unmatched-tip-link guard.
    """

    execution_type = ExecutionType.SIMULATED


def test_unmatched_tip_link_raises(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose = Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)

    validator = AreReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=[_MoveTcpAlternativeForPr2],
        ),
        pose_sequence=[pose],
        tip_link=robot_view.root,
    )

    with simulated_robot, pytest.raises(TipLinkDoesNotMatchAnyArm):
        validator.create_msc()


def test_pose_sequence_one_not_reachable(immutable_model_world):
    world, robot_view, context = immutable_model_world

    pose1 = Pose(Point3.from_iterable([1.6, 1.4, 1]), reference_frame=world.root)
    pose2 = Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)
    pose3 = Pose(Point3.from_iterable([2.7, 2.4, 1.5]), reference_frame=world.root)

    assert not IsReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose=pose3,
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )

    assert not AreReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose_sequence=[pose1, pose2, pose3],
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )


def test_is_grasp_reachable_by_copies_current_world_lazily(
    immutable_model_world, monkeypatch
):
    """
    The world copy and pose sequence must be produced when the predicate is
    *evaluated*, not when it is constructed, so the check reflects the current
    world state. We capture what the predicate hands to ``AreReachableBy``.
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")

    captured = {}

    def fake_call(self, *args, **kwargs):
        captured["world"] = self.world
        captured["pose_sequence"] = self.pose_sequence
        captured["tip_link"] = self.tip_link
        return True

    monkeypatch.setattr(AreReachableBy, "__call__", fake_call)

    predicate = IsGraspReachableBy(
        context=Context(
            robot=view,
            world=world,
        ),
        arm=Arms.RIGHT,
        grasp_pose=Pose(reference_frame=milk),
        object_designator=milk,
    )

    # Move the object *after* the predicate has been constructed.
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 0.7, 0, 0, 0, reference_frame=milk.parent_connection.parent
    )

    assert predicate()

    # The validator runs on a throwaway copy, never the live world.
    assert captured["world"] is not world
    # The copy reflects the *current* (moved) object pose, not the parse-time one.
    copied_milk = captured["world"].get_body_by_name("milk.stl")
    assert np.allclose(
        copied_milk.global_pose.to_position().to_np()[:3],
        milk.global_pose.to_position().to_np()[:3],
    )
    # A full grasp sequence (pre-pose, grasp, lift) is generated.
    assert len(captured["pose_sequence"]) == 3
    # The tip link is resolved in the copy, not the live world.
    assert captured["tip_link"]._world is captured["world"]


def test_is_grasp_reachable_by_uses_the_grasp_pose_sequence(
    immutable_model_world, monkeypatch
):
    """
    With a grasp pose set, the reach pose sequence is checked.
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    target = Pose(Point3.from_iterable([2, 1.5, 0.7]), reference_frame=world.root)

    captured = {}
    monkeypatch.setattr(
        AreReachableBy,
        "__call__",
        lambda self, *a, **k: captured.setdefault("seq", self.pose_sequence) or True,
    )

    assert IsGraspReachableBy(
        context=Context(
            robot=view,
            world=world,
        ),
        arm=Arms.RIGHT,
        grasp_pose=target,
        object_designator=milk,
    )()

    assert len(captured["seq"]) == 3


def test_opening_a_container_checks_reaching_its_handle(
    immutable_model_world, monkeypatch
):
    """
    A handle is reached for, not grasped around: there is one pose to arrive at and no
    approach to clear the body's geometry, so opening a container asks
    :class:`IsReachableBy` about the handle rather than a grasp sequence.
    """
    world, view, context = immutable_model_world
    handle_body = world.get_body_by_name("milk.stl")
    with world.modify_world():
        world.add_semantic_annotation_recursively(handle := Handle(root=handle_body))

    sequence_calls = []
    single_calls = []
    monkeypatch.setattr(
        AreReachableBy,
        "__call__",
        lambda self, *a, **k: sequence_calls.append(self) or True,
    )
    monkeypatch.setattr(
        IsReachableBy,
        "__call__",
        lambda self, *a, **k: single_calls.append(self.pose) or True,
    )
    handle_body.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 0.7, 0, 0, 0, reference_frame=handle_body.parent_connection.parent
    )

    open_action = OpenAction(handle, Arms.RIGHT)
    sequential([open_action], context=context)
    assert evaluate_condition(
        OpenAction.pre_condition(
            open_action.bound_variables, context, open_action.designator_parameter
        )
    )

    assert not sequence_calls
    assert len(single_calls) == 1
    assert np.allclose(
        world.transform(single_calls[0], world.root).to_position().to_np()[:3],
        handle_body.global_pose.to_position().to_np()[:3],
    )


def test_is_object_reachable_by_reachable(immutable_model_world):
    """
    End-to-end: a graspable object in front of the robot is reachable.
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2, 1.5, 0.7, 0, 0, 0, reference_frame=milk.parent_connection.parent
    )

    assert IsObjectReachableBy(
        context=Context(
            robot=view,
            world=world,
        ),
        arm=Arms.RIGHT,
        graspable=world.get_semantic_annotations_by_type(Milk)[0],
    )


def test_is_object_reachable_by_not_reachable(immutable_model_world):
    """
    End-to-end: an object far away from the robot is not reachable.
    """
    world, view, context = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    milk.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        5, 5, 0.7, 0, 0, 0, reference_frame=milk.parent_connection.parent
    )

    assert not IsObjectReachableBy(
        context=Context(
            robot=view,
            world=world,
        ),
        arm=Arms.RIGHT,
        graspable=world.get_semantic_annotations_by_type(Milk)[0],
    )


# %% validation runs what execution runs


def _reachability_validator(world, robot_view, context):
    """
    :return: A validator for a pose within the robot's reach.
    """
    return AreReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose_sequence=[
            Pose(Point3.from_iterable([1.7, 1.4, 1]), reference_frame=world.root)
        ],
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )


def test_validation_avoids_collisions_when_the_run_does(immutable_model_world):
    """
    Collision avoidance does not only reject poses the robot would collide on, it
    changes the trajectory the solver produces at all.

    A validation run without it answers for a different trajectory than the one the plan
    goes on to execute, so it carries the same collision goals the executed chart does.
    """
    world, robot_view, context = immutable_model_world
    validator = _reachability_validator(world, robot_view, context)

    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=True):
        msc = validator.create_msc()

    assert len(msc.get_nodes_by_type(ExternalCollisionAvoidance)) == 1
    assert len(msc.get_nodes_by_type(SelfCollisionAvoidance)) == 1


def test_validation_leaves_out_collision_avoidance_when_the_run_does(
    immutable_model_world,
):
    """
    A run that does not avoid collisions is validated the same way.
    """
    world, robot_view, context = immutable_model_world
    validator = _reachability_validator(world, robot_view, context)

    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=False):
        msc = validator.create_msc()

    assert msc.get_nodes_by_type(ExternalCollisionAvoidance) == []
    assert msc.get_nodes_by_type(SelfCollisionAvoidance) == []


def test_validation_frees_the_gripper_like_the_reach_it_validates(
    immutable_model_world,
):
    """
    The reach being validated allows the gripper to touch what it grasps, so the
    validation has to allow it too.

    Without that, the grasp pose lies inside the buffer zone the probe keeps around the
    object, no trajectory ever converges on it, and every candidate is reported
    unreachable.
    """
    world, robot_view, context = immutable_model_world
    validator = _reachability_validator(world, robot_view, context)

    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=True):
        msc = validator.create_msc()

    [rules_node] = msc.get_nodes_by_type(UpdateTemporaryCollisionRules)
    (rule,) = rules_node.temporary_rules
    assert rule.end_effector is ViewManager.get_end_effector_view(
        Arms.RIGHT, robot_view
    )


def test_validation_uses_the_same_goal_tolerances_the_motions_do(
    immutable_model_world,
):
    """
    A reach is only finished once it is within the tolerance its motion was given, so a
    probe that settles for a looser one reports poses reachable that the motion would
    still be working towards.
    """
    world, robot_view, context = immutable_model_world
    validator = _reachability_validator(world, robot_view, context)

    msc = validator.create_msc()

    [sequence] = msc.get_nodes_by_type(Sequence)
    goals = [node for node in sequence.nodes if isinstance(node, CartesianPose)]
    tolerances = validator.context.motion_tolerances
    assert goals
    for goal in goals:
        assert goal.translation_threshold == tolerances.default_tcp_position_threshold
        assert goal.orientation_threshold == tolerances.tool_orientation_threshold


def test_validation_gives_up_on_a_pose_it_stops_approaching(immutable_model_world):
    """
    A probe that cannot get any closer to its goal would otherwise hold the whole tick
    budget before being called unreachable, and a location grounds by trying candidates
    until one works.

    Watching the sequence for a stall abandons a bad candidate as soon as it stops
    making progress.
    """
    world, robot_view, context = immutable_model_world
    validator = _reachability_validator(world, robot_view, context)

    msc = validator.create_msc()

    [progress_monitor] = msc.get_nodes_by_type(StillProgressing)
    [sequence] = msc.get_nodes_by_type(Sequence)
    assert progress_monitor.monitored_node is sequence


def test_validation_gives_back_the_collision_rules_it_found(immutable_model_world):
    """
    A location judges every candidate against one world copy, so a validation run must
    leave the collision rules exactly as it found them.

    The reach installs a gripper allowance of its own while it runs, and that allowance
    outranks the rules of the run it was probing, so a candidate evaluated after another
    would otherwise be judged against the previous candidate's rules.
    """
    world, robot_view, context = immutable_model_world
    validator = _reachability_validator(world, robot_view, context)
    rule_of_the_run = AllowSelfCollisions(robot=robot_view)
    world.collision_manager.clear_temporary_rules()
    world.collision_manager.add_temporary_rule(rule_of_the_run)

    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=True):
        validator()

    assert world.collision_manager.temporary_rules == [rule_of_the_run]


def test_an_unreachable_pose_is_given_up_on_by_the_stall_monitor(immutable_model_world):
    """
    The stall monitor is what ends a hopeless probe, so the validator does not need a
    tick budget of its own to stop one.
    """
    world, robot_view, context = immutable_model_world
    validator = AreReachableBy(
        context=Context(
            world=world,
            robot=robot_view,
            alternative_motion_mappings=context.alternative_motion_mappings,
        ),
        pose_sequence=[
            Pose(Point3.from_iterable([2.3, 2, 1]), reference_frame=world.root)
        ],
        tip_link=world.get_body_by_name("r_gripper_tool_frame"),
    )

    with world.reset_state_context():
        executor = validator.create_executor(validator.create_msc())

        with pytest.raises(NoProgressError):
            executor.tick_until_end()


def test_validation_is_run_with_the_motion_settings_of_the_plan(
    immutable_model_world, monkeypatch
):
    """
    The probe answers about the motion the plan will execute, so the copy it runs in is
    addressed by a context carrying the plan's own motion settings: the alternatives a
    motion is replaced by, and the tolerances a reach counts as finished at.

    A probe given the defaults instead simulates a motion the plan never runs.
    """
    world, robot_view, _ = immutable_model_world
    milk = world.get_body_by_name("milk.stl")
    context = Context(
        world=world,
        robot=robot_view,
        alternative_motion_mappings=[_MoveTcpAlternativeForPr2],
        motion_tolerances=MotionToleranceConfig(
            default_tcp_position_threshold=0.05, tool_orientation_threshold=0.5
        ),
    )
    validator = IsGraspReachableBy(
        context=context,
        arm=Arms.RIGHT,
        grasp_pose=Pose(reference_frame=milk),
        object_designator=milk,
    )

    probe_contexts = []
    monkeypatch.setattr(
        AreReachableBy,
        "__call__",
        lambda self, *a, **k: probe_contexts.append(self.context) or True,
    )

    assert validator()

    [probe_context] = probe_contexts
    assert (
        probe_context.alternative_motion_mappings == context.alternative_motion_mappings
    )
    assert probe_context.motion_tolerances == context.motion_tolerances


# %% grasping from a standing pose


def _milk_within_reach(world):
    """
    Put the milk where :func:`test_pose_reachable` establishes the right arm can reach.

    :return: The milk annotation, moved.
    """
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    milk.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        1.7, 1.4, 1.0, reference_frame=world.root
    )
    return milk


def test_any_grasp_validator_keeps_the_grasp_it_reached(immutable_model_world):
    """
    The pose that is accepted and the grasp it was accepted for belong together, so the
    validator hands back the grasp rather than only a verdict.
    """
    world, robot_view, context = immutable_model_world
    milk = _milk_within_reach(world)
    validator = IsObjectReachableBy(context=context, arm=Arms.RIGHT, graspable=milk)

    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=True):
        assert validator()

    assert validator.reachable_grasp is not None
    assert validator.reachable_grasp.reference_frame is milk.root
    reachable_grasps = [
        grasp
        for grasp in ViewManager.get_end_effector_view(
            Arms.RIGHT, robot_view
        ).grasp_poses_by_distance(milk)
    ]
    assert any(
        np.allclose(
            validator.reachable_grasp.to_homogeneous_matrix().to_np(),
            grasp.to_homogeneous_matrix().to_np(),
        )
        for grasp in reachable_grasps
    ), "the grasp handed back must be one of the grasps the object offers"


def test_any_grasp_validator_tries_the_nearest_grasps_first(
    immutable_model_world, monkeypatch
):
    """
    The order the grasps are tried in is the gripper's own ranking, so the grasp that is
    accepted is the nearest one that can be reached rather than whichever the object
    happened to generate first.
    """
    world, robot_view, context = immutable_model_world
    milk = _milk_within_reach(world)
    validator = IsObjectReachableBy(context=context, arm=Arms.RIGHT, graspable=milk)

    tried = []
    monkeypatch.setattr(
        IsObjectReachableBy,
        "_reaches",
        lambda self, grasp_pose, *a, **k: tried.append(grasp_pose) or False,
    )

    assert not validator()

    expected = ViewManager.get_end_effector_view(
        Arms.RIGHT, robot_view
    ).grasp_poses_by_distance(milk)
    assert len(tried) == len(expected)
    for actual, wanted in zip(tried, expected):
        np.testing.assert_allclose(
            actual.to_homogeneous_matrix().to_np(),
            wanted.to_homogeneous_matrix().to_np(),
            atol=1e-9,
        )


def test_any_grasp_validator_forgets_a_grasp_when_it_fails(immutable_model_world):
    """
    The recorded grasp belongs to the pose the validator was last asked about, so a
    failed call must not leave the previous answer behind for a caller to read.
    """
    world, robot_view, context = immutable_model_world
    milk = _milk_within_reach(world)
    validator = IsObjectReachableBy(context=context, arm=Arms.RIGHT, graspable=milk)

    with ExecutionEnvironment(ExecutionType.SIMULATED, collision_avoidance=True):
        assert validator()
        assert validator.reachable_grasp is not None
        milk.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                20, 20, 0.8, reference_frame=world.root
            )
        )
        assert not validator()

    assert validator.reachable_grasp is None


# %% watching a grasp probe happen


class _RecordsWhatWasPublished:
    """
    Stands in for the marker publisher, so a test need not run a ROS node.
    """

    def __init__(self, published):
        self.published = published

    def __call__(self, _world, node):
        self.published.append((_world, node))
        return self

    def with_collision_visualization(self):
        return self


def test_a_grasp_probe_is_published_while_debugging(immutable_model_world, monkeypatch):
    """
    A grasp probe runs in a copy of the world, so nothing of it reaches Rviz unless the
    copy is published: a run being watched would show the robot standing still through
    every reach it judges.
    """
    world, robot_view, context = immutable_model_world
    milk = _milk_within_reach(world)
    published = []
    monkeypatch.setattr(
        pose_validator, "VizMarkerPublisher", _RecordsWhatWasPublished(published)
    )
    node = object()
    validator = IsObjectReachableBy(
        context=Context(world=world, robot=robot_view, ros_node=node, _debug=True),
        arm=Arms.RIGHT,
        graspable=milk,
    )

    probe = validator._copied_world()

    assert published == [(probe.world, node)]


def test_a_grasp_probe_is_not_published_otherwise(immutable_model_world, monkeypatch):
    """
    Publishing a world costs something on every candidate a location tries, so a run
    that is not being watched does not pay it.
    """
    world, robot_view, context = immutable_model_world
    milk = _milk_within_reach(world)
    published = []
    monkeypatch.setattr(
        pose_validator, "VizMarkerPublisher", _RecordsWhatWasPublished(published)
    )
    validator = IsObjectReachableBy(
        context=Context(world=world, robot=robot_view),
        arm=Arms.RIGHT,
        graspable=milk,
    )

    validator._copied_world()

    assert published == []
