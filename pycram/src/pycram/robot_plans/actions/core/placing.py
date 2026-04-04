from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body
from typing_extensions import Union, Optional, Type, Any, Iterable
from typing_extensions import Optional, Any

from pycram.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
)
from pycram.datastructures.grasp import GraspDescription


from pycram.plans.factories import sequential, execute_single
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from pycram.robot_plans.motions.gripper import (
    MoveToolCenterPointMotion,
    MoveGripperMotion,
)
from pycram.validation.error_checkers import PoseErrorChecker
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class PlaceAction(ActionDescription):
    """
    Places an Object at a position using an arm.
    """

    object_designator: Body
    """
    Object designator_description describing the object that should be place
    """
    target_location: Pose
    """
    Pose in the world at which the object should be placed
    """
    arm: Arms
    """
    Arm that is currently holding the object
    """

    def execute(self) -> None:
        arm = ViewManager.get_arm_view(self.arm, self.robot)
        manipulator = arm.manipulator

        previous_pick = self.plan_node.get_previous_node_by_designator_type(
            PickUpAction
        )
        previous_grasp = (
            previous_pick.designator.grasp_description
            if previous_pick
            else GraspDescription(
                ApproachDirection.FRONT, VerticalAlignment.NoAlignment, manipulator
            )
        )

        self.add_subplan(
            sequential(
                [
                    ReachAction(
                        self.target_location,
                        self.arm,
                        previous_grasp,
                        self.object_designator,
                        reverse_reach_order=True,
                    ),
                    MoveGripperMotion(GripperState.OPEN, self.arm),
                ]
            )
        ).perform()

        # Detaches the object from the robot
        world_root = self.world.root
        obj_transform = self.world.compute_forward_kinematics(
            world_root, self.object_designator
        )
        with self.world.modify_world():
            self.world.remove_connection(self.object_designator.parent_connection)
            connection = Connection6DoF.create_with_dofs(
                parent=world_root, child=self.object_designator, world=self.world
            )
            self.world.add_connection(connection)
            connection.origin = obj_transform

        _, _, retract_pose = previous_grasp._pose_sequence(
            self.target_location, self.object_designator, reverse=True
        )

        self.add_subplan(
            execute_single(MoveToolCenterPointMotion(retract_pose, self.arm))
        ).perform()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        """
        Check if the object is placed at the target location.
        """
        self.validate_loss_of_contact()
        self.validate_placement_location()

    def validate_loss_of_contact(self):
        """
        Check if the object is still in contact with the robot after placing it.
        """
        contact_links = self.object_designator.get_contact_points_with_body(
            World.robot
        ).get_all_bodies()
        if contact_links:
            raise ObjectStillInContact(
                self.object_designator,
                contact_links,
                self.target_location,
                World.robot,
                self.arm,
            )

    def validate_placement_location(self):
        """
        Check if the object is placed at the target location.
        """
        pose_error_checker = PoseErrorChecker(World.conf.get_pose_tolerance())
        if not pose_error_checker.is_error_acceptable(
            self.object_designator.pose, self.target_location
        ):
            raise ObjectNotPlacedAtTargetLocation(
                self.object_designator, self.target_location, World.robot, self.arm
            )
