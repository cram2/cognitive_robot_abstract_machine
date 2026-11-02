from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cramph.composites import (
    ChildChooser,
    ChildChooserAccess,
    CompositeNodeChoosingItsChild,
)
from cramph.monitors import CountTicks
from cramph.node import StatechartNode
from cramph.nodes_for_testing import ConstTrueNode

from cramph.composites import Sequence
from cramph.context import StatechartContext
from cramph.data_types import LifeCycleValues
from cramph.executor import StatechartExecutor
from cramph.statechart import Statechart
from cramph.world_modification_nodes import MoveBranch
from giskardpy.motion_control import MotionControl
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% helpers


def _add_box_next_to_the_robot(world: World) -> Body:
    """
    :return: A box fixed to the root of `world`, half a metre in front of the origin.
    """
    box = Body(name=PrefixedName("box"))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.5
                ),
            )
        )
    return box


# %% a motion built before the branch moved


def test_a_motion_built_before_a_body_was_moved_moves_it_with_its_new_parent(
    cylinder_bot_world: World,
) -> None:
    world = cylinder_bot_world
    box = _add_box_next_to_the_robot(world)
    robot = world.get_kinematic_structure_entity_by_name("bot")
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    goal_pose = Pose.from_xyz_rpy(x=1.5, reference_frame=world.root)
    statechart.add_node(
        carry := Sequence(
            nodes=[
                CartesianPose(
                    root_link=world.root,
                    tip_link=robot,
                    goal_pose=Pose.from_xyz_rpy(x=0.2, reference_frame=world.root),
                    name="approach",
                ),
                MoveBranch(branch_root=box, new_parent=robot),
                CartesianPose(
                    root_link=world.root,
                    tip_link=box,
                    goal_pose=goal_pose,
                    name="carry",
                ),
            ]
        )
    )
    statechart.add_node(EndMotion.when_true(carry))

    executor.compile(statechart)
    executor.tick_until_end()

    assert carry.life_cycle_state == LifeCycleValues.SUCCEEDED
    assert np.allclose(
        world.compute_forward_kinematics(world.root, box), goal_pose, atol=0.02
    )


def test_building_the_nodes_again_registers_no_further_variables(
    cylinder_bot_world: World,
) -> None:
    world = cylinder_bot_world
    box = _add_box_next_to_the_robot(world)
    robot = world.get_kinematic_structure_entity_by_name("bot")
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[MotionControl()]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        CartesianPose(
            root_link=world.root,
            tip_link=robot,
            goal_pose=Pose.from_xyz_rpy(x=0.2, reference_frame=world.root),
        )
    )
    executor.compile(statechart)
    variable_count = len(executor.context.float_variable_data.data)

    world.move_branch(box, robot)
    executor.tick()

    assert len(executor.context.float_variable_data.data) == variable_count


# %% coming to rest before the chart blocks its tick

MAXIMUM_TICKS = 500
"""
How many ticks a test waits for the robot to come to rest before it gives up.
"""


@dataclass
class ChooserNotingTheSpeed(ChildChooser[CompositeNodeChoosingItsChild]):
    """
    Chooses one prepared child, noting how fast the robot was moving when it was asked.
    """

    world: World
    """
    The world whose robot moves.
    """

    child: StatechartNode
    """
    The child to choose.
    """

    speeds_when_asked: list[float] = field(default_factory=list)
    """
    The fastest velocity of any degree of freedom, once per question.
    """

    def choose_child(
        self, node: CompositeNodeChoosingItsChild
    ) -> StatechartNode | None:
        self.speeds_when_asked.append(_fastest_velocity(self.world))
        return self.child


def _fastest_velocity(world: World) -> float:
    """
    :return: The fastest velocity of any degree of freedom of `world`.
    """
    return float(np.max(np.abs(world.state.velocities)))


def test_a_moving_robot_decelerates_to_rest_before_a_child_is_chosen(
    cylinder_bot_world: World,
) -> None:
    """
    Choosing blocks the tick, so the robot is brought to rest first, by the controller
    slowing it down rather than by its velocities being cut to zero.
    """
    world = cylinder_bot_world
    robot = world.get_kinematic_structure_entity_by_name("bot")
    motion_control = MotionControl()
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[motion_control]
    )
    chooser = ChooserNotingTheSpeed(world=world, child=ConstTrueNode())
    executor.context.add_extension(ChildChooserAccess(choosers=[chooser]))
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        CartesianPose(
            root_link=world.root,
            tip_link=robot,
            goal_pose=Pose.from_xyz_rpy(x=5, reference_frame=world.root),
        )
    )
    statechart.add_node(driving_a_while := CountTicks(ticks=20))
    statechart.add_node(choosing := CompositeNodeChoosingItsChild(name="choosing"))
    choosing.start_condition = driving_a_while.observes_true
    executor.compile(statechart)

    speeds_while_holding = []
    for _ in range(MAXIMUM_TICKS):
        if chooser.speeds_when_asked:
            break
        executor.tick()
        holding_still = (
            choosing.life_cycle_state == LifeCycleValues.RUNNING
            and not chooser.speeds_when_asked
        )
        if holding_still:
            speeds_while_holding.append(_fastest_velocity(world))

    [speed_when_chosen] = chooser.speeds_when_asked
    assert speed_when_chosen <= motion_control.speed_at_rest
    assert len(speeds_while_holding) > 2
    assert speeds_while_holding[0] > motion_control.speed_at_rest
    assert speeds_while_holding == sorted(speeds_while_holding, reverse=True)


def test_a_degree_of_freedom_the_controller_does_not_command_leaves_the_robot_at_rest(
    cylinder_bot_world: World,
) -> None:
    """
    Only what the controller drives has to come to rest; a free body sliding about is
    not the robot moving.
    """
    world = cylinder_bot_world
    robot = world.get_kinematic_structure_entity_by_name("bot")
    free_box = Body(name=PrefixedName("free_box"))
    with world.modify_world():
        world.add_connection(
            free_connection := Connection6DoF.create_with_dofs(
                world=world, parent=world.root, child=free_box
            )
        )
    motion_control = MotionControl()
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[motion_control]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        CartesianPose(
            root_link=world.root,
            tip_link=robot,
            goal_pose=Pose.from_xyz_rpy(x=0.2, reference_frame=world.root),
        )
    )
    executor.compile(statechart)
    world.state[free_connection.dofs[0].id].velocity = 1.0

    assert motion_control.before_recompile(executor)


def test_the_speed_at_rest_given_to_a_motion_control_decides_when_the_robot_is_at_rest(
    cylinder_bot_world: World,
) -> None:
    world = cylinder_bot_world
    robot = world.get_kinematic_structure_entity_by_name("bot")
    motion_control = MotionControl(speed_at_rest=0.5)
    executor = StatechartExecutor(
        context=StatechartContext(world=world), extensions=[motion_control]
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        CartesianPose(
            root_link=world.root,
            tip_link=robot,
            goal_pose=Pose.from_xyz_rpy(x=0.2, reference_frame=world.root),
        )
    )
    executor.compile(statechart)
    commanded = motion_control.qp_controller.active_degrees_of_freedom[0]

    world.state[commanded.id].velocity = motion_control.speed_at_rest / 2
    assert motion_control.before_recompile(executor)

    world.state[commanded.id].velocity = motion_control.speed_at_rest * 2
    assert not motion_control.before_recompile(executor)
