from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from typing_extensions import Any, Dict

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.factories import or_, not_, and_
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import (
    Arms,
    ApproachDirection,
    VerticalAlignment,
    MovementType,
)
from pycram.datastructures.grasp import GraspDescription
from pycram.plans.factories import sequential, execute_single
from pycram.querying.predicates import GripperIsFree
from pycram.robot_plans.actions.base import ActionDescription, DescriptionType
from pycram.robot_plans.actions.core.pick_up import PickUpAction, ReachAction
from pycram.robot_plans.motions.gripper import (
    MoveGripperMotion,
    MoveToolCenterPointMotion,
)
from pycram.view_manager import ViewManager
from semantic_digital_twin.datastructures.definitions import GripperState
from semantic_digital_twin.reasoning.predicates import allclose
from semantic_digital_twin.reasoning.robot_predicates import is_body_in_gripper
from semantic_digital_twin.spatial_types.spatial_types import Pose
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

        target_pre_pose, target_pose, retract_pose = previous_grasp._pose_sequence(
            self.target_location, self.object_designator, reverse=True
        )

        self.add_subplan(
            sequential(
                [
                    MoveToolCenterPointMotion(
                        target_pre_pose, self.arm, allow_gripper_collision=False
                    ),
                    MoveToolCenterPointMotion(
                        target_pose,
                        self.arm,
                        allow_gripper_collision=False,
                        movement_type=MovementType.CARTESIAN,
                    ),
                    MoveGripperMotion(GripperState.OPEN, self.arm),
                    MoveToolCenterPointMotion(retract_pose, self.arm),
                ]
            )
        ).perform()

        # Detaches the object from the robot
        world_root = self.world.root
        obj_transform = self.world.compute_forward_kinematics(
            world_root, self.object_designator
        )
        # with self.world.modify_world():
        #     self.world.remove_connection(self.object_designator.parent_connection)
        #     connection = Connection6DoF.create_with_dofs(
        #         parent=world_root, child=self.object_designator, world=self.world
        #     )
        #     self.world.add_connection(connection)
        #     connection.origin = obj_transform

        # _, _, retract_pose = previous_grasp._pose_sequence(
        #     self.target_location, self.object_designator, reverse=True
        # )
        #
        # self.add_subplan(
        #     execute_single(MoveToolCenterPointMotion(retract_pose, self.arm))
        # ).perform()

    @staticmethod
    def pre_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        """
        The object needs to be in the gripper frame
        """
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        return or_(
            not_(GripperIsFree(manipulator)),
            is_body_in_gripper(kwargs["object_designator"], manipulator) > 0.9,
        )

    @staticmethod
    def post_condition(
        variables, context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        """
        the gripper must be free again and the object needs to be at the target location
        """
        manipulator = ViewManager.get_end_effector_view(variables["arm"], context.robot)
        return and_(
            GripperIsFree(manipulator),
            is_body_in_gripper(kwargs["object_designator"], manipulator) < 0.1,
            allclose(
                kwargs["object_designator"].global_pose,
                kwargs["target_location"].to_spatial_type(),
                atol=0.03,
            ),
        )
