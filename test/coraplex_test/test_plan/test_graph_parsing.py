from coraplex.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from coraplex.datastructures.grasp import GraspDescription
from coraplex.execution_environment import simulated_robot
from coraplex.plans.attachment_nodes import ModelChangeNode
from coraplex.plans.executables import GiskardExecutable, ModelChangeExecutable
from coraplex.plans.factories import execute_single, sequential
from coraplex.robot_plans import MoveToolCenterPointMotion
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.pick_up import ReachAction, PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import MoveTorsoAction, ParkArmsAction
from coraplex.utils import split_list_by_type
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3


def test_parse_simple_action(immutable_model_world):
    world, view, context = immutable_model_world

    plan = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)

    plan.notify()

    executable = plan.parse()

    assert type(executable) == GiskardExecutable
    assert executable.pre_condition_node
    assert executable.post_condition_node
    assert len(executable.motion_mappings) == 1
    assert type(list(executable.motion_mappings.values())[0]) == JointPositionList


def test_merge_motions(immutable_model_world, rclpy_node):
    world, view, context = immutable_model_world

    world.get_body_by_name("milk.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(2, 1.5, 0.7, 0, 0, 0)
    )

    plan = execute_single(
        ReachAction(
            Pose.from_xyz_rpy(2, 1.5, 0.7, reference_frame=world.root),
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                view.right_arm.end_effector,
            ),
            world.get_body_by_name("milk.stl"),
        ),
        context=context,
    )

    plan.notify()

    executable = plan.parse()

    assert type(executable) == GiskardExecutable
    assert len(executable.motion_mappings) == 2
    assert executable.pre_condition_node
    assert executable.post_condition_node

    with simulated_robot:
        executable.execute()


def test_parse_pick_up(immutable_model_world):
    world, view, context = immutable_model_world

    plan = execute_single(
        PickUpAction(
            world.get_body_by_name("milk.stl"),
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                view.right_arm.end_effector,
            ),
        ),
        context=context,
    )

    plan.notify()

    # plan.plan.plot()

    executable = plan.parse()

    assert len(executable.execution_list) == 3
    assert type(executable.execution_list[0]) == GiskardExecutable
    assert type(executable.execution_list[1]) == ModelChangeExecutable
    assert type(executable.execution_list[2]) == GiskardExecutable


def test_parse_pick_up_merges_motions_around_model_change(immutable_model_world):
    """
    The motions on each side of the model change (the attach) must be merged into a
    single giskard executable per side, so the model change splits the plan into
    exactly [merged motions, model change, merged motions].
    """
    world, view, context = immutable_model_world

    plan = execute_single(
        PickUpAction(
            world.get_body_by_name("milk.stl"),
            Arms.RIGHT,
            GraspDescription(
                ApproachDirection.FRONT,
                VerticalAlignment.NoAlignment,
                view.right_arm.end_effector,
            ),
        ),
        context=context,
    )

    plan.notify()
    executable = plan.parse()

    # The four motions before the attach (open gripper, reach pre-pose, reach pose,
    # close gripper) merge into one executable; the lift after it into another.
    assert len(executable.execution_list[0].motion_mappings) == 4
    assert len(executable.execution_list[2].motion_mappings) == 1


def test_parse_complex_plan(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            ReachAction(
                target_pose=Pose(
                    Point3.from_iterable([1, -2, 0.8]), reference_frame=world.root
                ),
                object_designator=world.get_body_by_name("milk.stl"),
                arm=Arms.LEFT,
                grasp_description=GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    view.right_arm.end_effector,
                ),
            ),
        ],
        context=context,
    )

    plan.notify()
    exec = plan.parse()
    assert type(exec) == GiskardExecutable
    assert len(exec.motion_mappings) == 3


def test_parsing_two_actions_into_one_exec(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            ParkArmsAction(Arms.BOTH),
            ReachAction(
                target_pose=Pose(
                    Point3.from_iterable([1, -2, 0.8]), reference_frame=world.root
                ),
                object_designator=world.get_body_by_name("milk.stl"),
                arm=Arms.LEFT,
                grasp_description=GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    view.right_arm.end_effector,
                ),
            ),
        ],
        context=context,
    )

    plan.notify()
    exec = plan.parse()

    assert type(exec) == GiskardExecutable
    assert len(exec.motion_mappings) == 3


def test_parse_pick_place(immutable_model_world):
    world, view, context = immutable_model_world

    plan = sequential(
        [
            PickUpAction(
                world.get_body_by_name("milk.stl"),
                Arms.RIGHT,
                GraspDescription(
                    ApproachDirection.FRONT,
                    VerticalAlignment.NoAlignment,
                    view.right_arm.end_effector,
                ),
            ),
            PlaceAction(
                world.get_body_by_name("milk.stl"),
                Pose(reference_frame=world.root),
                Arms.RIGHT,
            ),
        ],
        context=context,
    )

    plan.notify()

    # plan.plan.plot()

    executable = plan.parse()

    assert len(executable.execution_list) == 2
    assert len(executable.execution_list[0].execution_list) == 3
    assert len(executable.execution_list[1].execution_list) == 3


def test_parse_transport_plan(mutable_model_world, rclpy_node):
    world, view, context = mutable_model_world

    plan = sequential(
        [
            MoveTorsoAction(TorsoState.HIGH),
            ParkArmsAction(Arms.BOTH),
            TransportAction(
                world.get_body_by_name("milk.stl"),
                Pose.from_xyz_rpy(2.37, 2.5, 1.05, reference_frame=world.root),
                Arms.RIGHT,
            ),
        ],
        context=context,
    )

    plan.notify()
    exec = plan.parse()

    with simulated_robot:
        exec.execute()


def test_split_by_type(immutable_model_world):
    world, view, context = immutable_model_world

    split_list = [
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
        ModelChangeNode(body=world.get_body_by_name("milk.stl"), new_parent=world.root),
        MoveToolCenterPointMotion(Pose(), Arms.RIGHT),
    ]

    splitted_list = split_list_by_type(split_list, ModelChangeNode)

    assert len(splitted_list) == 3
    assert len(splitted_list[0]) == 1
    assert len(splitted_list[1]) == 1
    assert len(splitted_list[2]) == 1


def test_split_by_type_empty_list():
    assert split_list_by_type([], ModelChangeNode) == []


def test_split_by_type_without_match_stays_one_group():
    no_model_change = [
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
        MoveToolCenterPointMotion(Pose(), Arms.RIGHT),
    ]

    splitted_list = split_list_by_type(no_model_change, ModelChangeNode)

    assert len(splitted_list) == 1
    assert splitted_list[0] == no_model_change


def test_split_by_type_groups_consecutive_elements(immutable_model_world):
    world, view, context = immutable_model_world
    model_change = ModelChangeNode(
        body=world.get_body_by_name("milk.stl"), new_parent=world.root
    )

    split_list = [
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
        MoveToolCenterPointMotion(Pose(), Arms.RIGHT),
        model_change,
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
    ]

    splitted_list = split_list_by_type(split_list, ModelChangeNode)

    assert [len(group) for group in splitted_list] == [2, 1, 1]
    assert splitted_list[1] == [model_change]
    assert all(not isinstance(element, ModelChangeNode) for element in splitted_list[0])


def test_split_by_type_leading_and_trailing_match(immutable_model_world):
    world, view, context = immutable_model_world
    first_model_change = ModelChangeNode(
        body=world.get_body_by_name("milk.stl"), new_parent=world.root
    )
    last_model_change = ModelChangeNode(
        body=world.get_body_by_name("milk.stl"), new_parent=world.root
    )

    split_list = [
        first_model_change,
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
        last_model_change,
    ]

    splitted_list = split_list_by_type(split_list, ModelChangeNode)

    assert [len(group) for group in splitted_list] == [1, 1, 1]
    assert splitted_list[0] == [first_model_change]
    assert splitted_list[2] == [last_model_change]
