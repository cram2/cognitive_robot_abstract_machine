"""
Tests a plan that takes a bowl and a cup out of a kitchen drawer, one in each hand and
both at once, and sets them down on the kitchen island, while avoiding collisions; also
with the arms left to the pick-ups and the places referring to them.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from coraplex.robot_plans.mixins import MovesToolCenterPoint
from cramph.composites import DefiniteStatementNode, Parallel, Sequence, StatementNode
from cramph.context import ContextExtension
from cramph.node import StatechartNode
from giskardpy.motion_control import WorldStateTrajectoryRecording, MotionControl
from giskardpy.motion_statechart.tasks.cartesian_tasks import HoldPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.qp.qp_controller_config import QPControllerConfig
from krrood.entity_query_language.factories import a, the, variable
from krrood.entity_query_language.query.match import Match
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from krrood_spatial_math.spatial_types import Pose
from krrood_spatial_math.spatial_types import Pose2D
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.robots.robot_parts import Arm
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Cup,
    Handle,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain
from ..plan_running import robot_extensions, run_plan

# %% a kitchen with a bowl and a cup in a drawer


@dataclass
class KitchenWithFilledDrawer:
    """
    A PR2 in the kitchen, whose closed drawer holds a bowl and a cup.
    """

    world: World
    """
    The world holding the kitchen, the robot and the two objects.
    """

    robot: PR2
    """
    The robot performing the plan.
    """

    context_extensions: list[ContextExtension]
    """
    The context extensions the plans of :attr:`robot` read.
    """

    drawer_handle: Handle
    """
    The handle of the drawer the objects lie in.
    """

    bowl: Bowl
    """
    The bowl in the drawer.
    """

    cup: Cup
    """
    The cup in the drawer.
    """

    @property
    def kitchen_island(self) -> Body:
        """
        :return: The surface of the kitchen island the objects are set down on.
        """
        return self.world.get_body_by_name("kitchen_island_surface")

    @property
    def world_T_bowl_goal(self) -> Pose:
        """
        :return: Where the bowl is set down on the kitchen island.
        """
        return self.world.transform(
            Pose.from_xyz_rpy(x=0.2, z=0.05, reference_frame=self.kitchen_island),
            self.world.root,
        )

    @property
    def world_T_cup_goal(self) -> Pose:
        """
        :return: Where the cup is set down on the kitchen island.
        """
        return self.world.transform(
            Pose.from_xyz_rpy(
                x=0.15, y=0.25, z=0.07, reference_frame=self.kitchen_island
            ),
            self.world.root,
        )


def add_cylinder(
    world: World,
    name: str,
    radius: float,
    height: float,
    parent_T_cylinder: HomogeneousTransformationMatrix,
) -> Body:
    """
    :param world: The world the cylinder is added to.
    :param name: The name of the new body.
    :param radius: The radius of the cylinder in meters.
    :param height: The height of the cylinder in meters.
    :param parent_T_cylinder: Where the cylinder stands, relative to the body it is
        fixed to.
    :return: A cylinder fixed to the reference frame of `parent_T_cylinder`.
    """
    shapes = ShapeCollection([Cylinder(width=2 * radius, height=height)])
    cylinder = Body(name=PrefixedName(name), collision=shapes, visual=shapes)
    with world.modify_world():
        world.add_kinematic_structure_entity(cylinder)
        world.add_connection(
            FixedConnection(
                parent=parent_T_cylinder.reference_frame,
                child=cylinder,
                parent_T_connection_expression=parent_T_cylinder,
            )
        )
    return cylinder


@pytest.fixture
def kitchen_with_filled_drawer(
    pr2_world_copy: World, kitchen_world: World
) -> KitchenWithFilledDrawer:
    world = pr2_world_copy
    world.merge_world(deepcopy(kitchen_world))
    robot = world.get_semantic_annotations_by_type(PR2)[0]
    # Collision avoidance moves the base, so the goals have to hold still in the world.
    robot.mobile_base.full_body_controlled = True
    drawer = world.get_body_by_name("sink_area_left_middle_drawer_main")
    cup = Cup(
        root=add_cylinder(
            world,
            name="cup",
            radius=0.04,
            height=0.07,
            parent_T_cylinder=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.1, y=0.2, z=-0.05, reference_frame=drawer
            ),
        )
    )
    bowl = Bowl(
        root=add_cylinder(
            world,
            name="bowl",
            radius=0.07,
            height=0.05,
            parent_T_cylinder=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=0.1, y=-0.2, z=-0.05, reference_frame=drawer
            ),
        )
    )
    drawer_handle = Handle(
        root=world.get_body_by_name("sink_area_left_middle_drawer_handle")
    )
    with world.modify_world():
        world.add_semantic_annotations([cup, bowl, drawer_handle])
    return KitchenWithFilledDrawer(
        world=world,
        robot=robot,
        context_extensions=robot_extensions(robot),
        drawer_handle=drawer_handle,
        bowl=bowl,
        cup=cup,
    )


# %% taking both objects out of the drawer and setting them down


def pick_ups_from_above(kitchen: KitchenWithFilledDrawer) -> list[PickUpAction]:
    """
    :param kitchen: The kitchen the plan is performed in.
    :return: The pick-up of the bowl with the left arm and of the cup with the right
        arm, each coming down onto it from above.
    """
    return [
        PickUpAction.from_above(graspable=kitchen.bowl, arm=kitchen.robot.left_arm),
        PickUpAction.from_above(graspable=kitchen.cup, arm=kitchen.robot.right_arm),
    ]


def transport_plan(
    kitchen: KitchenWithFilledDrawer, *, pick_ups: list[PickUpAction | Match]
) -> Sequence:
    """
    :param kitchen: The kitchen the plan is performed in.
    :param pick_ups: The pick-up of the bowl and of the cup, run at once.
    :return: The plan that opens the drawer, takes the bowl with the left arm and the
        cup with the right arm at once, and sets both down on the kitchen island at
        once.
    """
    robot = kitchen.robot
    return Sequence(
        nodes=transport_steps(
            kitchen,
            pick_ups=pick_ups,
            places=[
                PlaceAction(
                    arm=robot.left_arm,
                    graspable=kitchen.bowl,
                    goal_pose=kitchen.world_T_bowl_goal,
                ),
                PlaceAction(
                    arm=robot.right_arm,
                    graspable=kitchen.cup,
                    goal_pose=kitchen.world_T_cup_goal,
                ),
            ],
        )
    )


def transport_steps(
    kitchen: KitchenWithFilledDrawer,
    *,
    pick_ups: list[PickUpAction | Match],
    places: list[PlaceAction | Match],
) -> list[StatechartNode]:
    """
    :param kitchen: The kitchen the plan is performed in.
    :param pick_ups: The pick-up of the bowl and of the cup, run at once.
    :param places: The place of the bowl and of the cup, run at once.
    :return: The steps that open the drawer, take the bowl and the cup out of it at
        once, and set both down on the kitchen island at once.
    """
    world, robot = kitchen.world, kitchen.robot
    in_front_of_the_closed_drawer = Pose2D.from_xy_yaw(
        x=0.4, y=0.7, reference_frame=world.root
    )
    in_front_of_the_open_drawer = Pose2D.from_xy_yaw(
        x=0.3, y=0.89, reference_frame=world.root
    )
    facing_the_kitchen_island = Pose2D.from_xy_yaw(
        x=-0.25, y=1.85, yaw=np.pi, reference_frame=world.root
    )
    return [
        # The arms start stretched out, which would reach into the counter.
        ParkArmsAction(arms=robot.all_arms),
        NavigateAction(planar_goal_pose=in_front_of_the_closed_drawer),
        OpenAction.from_handle(kitchen.drawer_handle, robot.left_arm),
        ParkArmsAction(arms=robot.all_arms),
        NavigateAction(planar_goal_pose=in_front_of_the_open_drawer),
        Parallel(
            nodes=[
                HoldPose(
                    kinematic_chain=KinematicChain(
                        root=kitchen.world.root, tip=robot.mobile_base.root
                    )
                ),
                *pick_ups,
            ]
        ),
        ParkArmsAction(arms=robot.all_arms),
        NavigateAction(planar_goal_pose=facing_the_kitchen_island),
        MoveTorsoAction(torso_state=TorsoState.HIGH),
        Parallel(
            nodes=[
                HoldPose(
                    kinematic_chain=KinematicChain(
                        root=kitchen.world.root, tip=robot.mobile_base.root
                    )
                ),
                *places,
            ]
        ),
        ParkArmsAction(arms=robot.all_arms),
    ]


def assert_bowl_and_cup_stand_on_the_kitchen_island(
    kitchen: KitchenWithFilledDrawer,
) -> None:
    """
    Assert that both objects were released at their goals on the kitchen island.
    """
    # A pick-up and a place each end within the position threshold of their goal.
    position_tolerance = 2 * MovesToolCenterPoint.position_threshold
    for placed, world_T_goal in [
        (kitchen.bowl, kitchen.world_T_bowl_goal),
        (kitchen.cup, kitchen.world_T_cup_goal),
    ]:
        assert placed.root.parent_connection.parent is kitchen.world.root
        np.testing.assert_allclose(
            placed.root.global_pose.position.to_np(),
            world_T_goal.position.to_np(),
            atol=position_tolerance,
        )


def test_bowl_and_cup_are_moved_from_the_drawer_onto_the_kitchen_island(
    kitchen_with_filled_drawer: KitchenWithFilledDrawer, tmp_path: Path
) -> None:
    """
    The plan sets both objects down at their goals, and the trajectory it recorded
    plots.
    """
    kitchen = kitchen_with_filled_drawer

    recording = WorldStateTrajectoryRecording()

    run_plan(
        transport_plan(kitchen, pick_ups=pick_ups_from_above(kitchen)),
        kitchen.context_extensions,
        collision_avoidance=True,
        extensions=[
            MotionControl(
                qp_controller_config=QPControllerConfig.create_with_fast_simulation_defaults()
            ),
            recording,
        ],
    )
    recording.plotter.plot_trajectory(file_name=tmp_path / "trajectory.pdf")

    assert_bowl_and_cup_stand_on_the_kitchen_island(kitchen)


def test_bowl_and_cup_picked_up_as_statements_are_moved_onto_the_kitchen_island(
    kitchen_with_filled_drawer: KitchenWithFilledDrawer,
) -> None:
    """
    A pick-up described by a statement is grounded only once it starts, so the places
    after it are built before anything says how the objects will be held.
    """
    kitchen = kitchen_with_filled_drawer
    pick_ups = [
        a(PickUpAction)(grasp_candidate=pick_up.grasp_candidate, arm=pick_up.arm)
        for pick_up in pick_ups_from_above(kitchen)
    ]

    run_plan(
        transport_plan(kitchen, pick_ups=pick_ups),
        kitchen.context_extensions,
        collision_avoidance=True,
    )

    assert_bowl_and_cup_stand_on_the_kitchen_island(kitchen)


def test_bowl_and_cup_placed_by_the_arms_their_pick_ups_chose_are_moved_onto_the_kitchen_island(
    kitchen_with_filled_drawer: KitchenWithFilledDrawer,
) -> None:
    """
    The pick-ups leave their arms open and the places refer to them.

    Each place is built in the trial of the pick-ups before anything is grasped there,
    and measures the grasp only when it starts, in the trial as on the robot.
    """
    kitchen = kitchen_with_filled_drawer
    robot = kitchen.robot
    bowl_pick_up, cup_pick_up = pick_ups_from_above(kitchen)
    bowl_arm = variable(Arm, domain=[robot.left_arm, robot.right_arm])
    cup_arm = variable(Arm, domain=[robot.right_arm, robot.left_arm])
    pick_ups = [
        a(PickUpAction)(grasp_candidate=bowl_pick_up.grasp_candidate, arm=bowl_arm),
        a(PickUpAction)(grasp_candidate=cup_pick_up.grasp_candidate, arm=cup_arm),
    ]
    places = [
        the(PlaceAction)(
            arm=bowl_arm, graspable=kitchen.bowl, goal_pose=kitchen.world_T_bowl_goal
        ),
        the(PlaceAction)(
            arm=cup_arm, graspable=kitchen.cup, goal_pose=kitchen.world_T_cup_goal
        ),
    ]
    plan = Sequence(
        nodes=[
            a(Sequence)(
                nodes=transport_steps(kitchen, pick_ups=pick_ups, places=places)
            )
        ]
    )

    run_plan(plan, kitchen.context_extensions, collision_avoidance=True)

    assert_bowl_and_cup_stand_on_the_kitchen_island(kitchen)
    statement_nodes = [
        node for node in plan.descendants if isinstance(node, StatementNode)
    ]
    for pick_up, place in zip(pick_ups, places):
        [picked_up] = next(
            node for node in statement_nodes if node.statement is pick_up
        ).grounded_nodes
        placing = next(node for node in statement_nodes if node.statement is place)
        assert isinstance(placing, DefiniteStatementNode)
        assert [placed.arm for placed in placing.grounded_nodes] == [picked_up.arm]


def test_the_drawer_stays_open_while_its_contents_are_carried_away(
    kitchen_with_filled_drawer: KitchenWithFilledDrawer,
) -> None:
    kitchen = kitchen_with_filled_drawer
    drawer_joint = kitchen.world.get_body_by_name(
        "sink_area_left_middle_drawer_main"
    ).parent_connection

    run_plan(
        transport_plan(kitchen, pick_ups=pick_ups_from_above(kitchen)),
        kitchen.context_extensions,
        collision_avoidance=True,
    )

    # Opening ends once the goal driving the drawer to the joint's limit is within its
    # threshold, wherever the last control step within it landed.
    assert drawer_joint.position == pytest.approx(
        drawer_joint.raw_dof.limits.upper.position,
        abs=JointPositionList.threshold,
    )
