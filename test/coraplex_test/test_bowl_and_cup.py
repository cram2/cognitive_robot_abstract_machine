"""
Tests a plan that takes a bowl and a cup out of a kitchen drawer, one in each hand and
both at once, and sets them down on the kitchen island, while avoiding collisions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pytest

from coraplex.robot_plans.mixins import MovesToolCenterPoint
from coraplex.robot_plans.actions.core.container import OpenAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.pick_up import PickUpAction
from coraplex.robot_plans.actions.core.placing import PlaceAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from cramph.composites import Parallel, Sequence
from cramph.context import ContextExtension
from giskardpy.motion_statechart.tasks.cartesian_tasks import HoldPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Cup,
    Handle,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Cylinder
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

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


def transport_plan(kitchen: KitchenWithFilledDrawer) -> Sequence:
    """
    :param kitchen: The kitchen the plan is performed in.
    :return: The plan that opens the drawer, takes the bowl with the left arm and the
        cup with the right arm at once, and sets both down on the kitchen island at
        once.
    """
    world, robot = kitchen.world, kitchen.robot
    in_front_of_the_closed_drawer = Pose.from_xyz_rpy(
        x=0.4, y=0.7, reference_frame=world.root
    )
    in_front_of_the_open_drawer = Pose.from_xyz_rpy(
        x=0.3, y=0.89, reference_frame=world.root
    )
    facing_the_kitchen_island = Pose.from_xyz_rpy(
        x=-0.25, y=1.85, yaw=np.pi, reference_frame=world.root
    )
    return Sequence(
        nodes=[
            # The arms start stretched out, which would reach into the counter.
            ParkArmsAction(arms=robot.all_arms),
            NavigateAction(target_location=in_front_of_the_closed_drawer),
            OpenAction.from_handle(kitchen.drawer_handle, robot.left_arm),
            ParkArmsAction(arms=robot.all_arms),
            NavigateAction(target_location=in_front_of_the_open_drawer),
            Parallel(
                nodes=[
                    HoldPose(
                        root_link=kitchen.world.root, tip_link=robot.mobile_base.root
                    ),
                    PickUpAction.from_above(kitchen.bowl, robot.left_arm),
                    PickUpAction.from_above(kitchen.cup, robot.right_arm),
                ]
            ),
            ParkArmsAction(arms=robot.all_arms),
            NavigateAction(target_location=facing_the_kitchen_island),
            MoveTorsoAction(torso_state=TorsoState.HIGH),
            Parallel(
                nodes=[
                    HoldPose(
                        root_link=kitchen.world.root, tip_link=robot.mobile_base.root
                    ),
                    PlaceAction(
                        object_designator=kitchen.bowl,
                        target_location=kitchen.world_T_bowl_goal,
                    ),
                    PlaceAction(
                        object_designator=kitchen.cup,
                        target_location=kitchen.world_T_cup_goal,
                    ),
                ]
            ),
            ParkArmsAction(arms=robot.all_arms),
        ]
    )


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
    kitchen_with_filled_drawer: KitchenWithFilledDrawer,
) -> None:
    kitchen = kitchen_with_filled_drawer

    run_plan(
        transport_plan(kitchen), kitchen.context_extensions, collision_avoidance=True
    )

    assert_bowl_and_cup_stand_on_the_kitchen_island(kitchen)


def test_the_drawer_stays_open_while_its_contents_are_carried_away(
    kitchen_with_filled_drawer: KitchenWithFilledDrawer,
) -> None:
    kitchen = kitchen_with_filled_drawer
    drawer_joint = kitchen.world.get_body_by_name(
        "sink_area_left_middle_drawer_main"
    ).parent_connection

    run_plan(
        transport_plan(kitchen), kitchen.context_extensions, collision_avoidance=True
    )

    # Opening ends once the goal driving the drawer to the joint's limit is within its
    # threshold, wherever the last control step within it landed.
    assert drawer_joint.position == pytest.approx(
        drawer_joint.raw_dof.limits.upper.position,
        abs=JointPositionList.threshold,
    )
