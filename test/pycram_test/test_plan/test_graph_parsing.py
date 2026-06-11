from pycram.plans.attachment_nodes import ModelChangeNode
from pycram.plans.executables import (
    ConditionExecutable,
    MotionExecutable,
    LanguageExecutable,
)
from pycram.datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import execute_single
from pycram.robot_plans.actions.core.pick_up import ReachAction, PickUpAction
from pycram.robot_plans.actions.core.robot_body import MoveTorsoAction
from pycram.robot_plans.motions.gripper import MoveToolCenterPointMotion
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.semantic_annotations.position_descriptions import (
    VerticalSemanticDirection,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from pycram.utils import split_list_by_type


def test_parse_simple_action(immutable_model_world):
    world, view, context = immutable_model_world

    plan = execute_single(MoveTorsoAction(TorsoState.HIGH), context=context)

    plan.notify()

    executable = plan.parse()

    assert len(executable.execution_list) == 3
    assert type(executable.execution_list[0]) == ConditionExecutable
    assert type(executable.execution_list[1]) == MotionExecutable


def test_merge_motions(immutable_model_world):
    world, view, context = immutable_model_world

    plan = execute_single(
        ReachAction(
            Pose(reference_frame=world.root),
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

    assert len(executable.execution_list) == 3
    assert type(executable.execution_list[0]) == ConditionExecutable
    assert type(executable.execution_list[1]) == LanguageExecutable


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
    assert type(executable.execution_list[0]) == ConditionExecutable
    assert type(executable.execution_list[1]) == LanguageExecutable
    assert len(executable.execution_list[1].execution_list) == 5


def test_split_by_type():

    split_list = [
        MoveToolCenterPointMotion(Pose(), Arms.LEFT),
        ModelChangeNode(body=None, new_parent=None),
        MoveToolCenterPointMotion(Pose(), Arms.RIGHT),
    ]

    splitted_list = split_list_by_type(split_list, ModelChangeNode)

    assert len(splitted_list) == 3
    assert len(splitted_list[0]) == 1
    assert len(splitted_list[1]) == 1
    assert len(splitted_list[2]) == 1
