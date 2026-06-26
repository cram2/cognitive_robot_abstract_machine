from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional, Type

from coraplex.datastructures.enums import DetectionTechnique, DetectionState
from coraplex.datastructures.grasp import GraspDescription
from coraplex.perception import PerceptionQuery
from coraplex.plans.factories import sequential, execute_single
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveManipulatorAction
from coraplex.robot_plans.motions.misc import DetectingMotion
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    Pose,
    Point3,
    Pose2D,
)
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import (
    Region,
    SemanticAnnotation,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)


@dataclass
class DetectAction(ActionDescription):
    """
    Detects an object that fits the object description and returns an object designator_description describing the object.

    If no object is found, an PerceptionObjectNotFound error is raised.
    """

    technique: DetectionTechnique
    """
    The technique that should be used for detection
    """
    state: Optional[DetectionState] = None
    """
    The state of the detection, e.g Start Stop for continues perception
    """
    object_sem_annotation: Type[SemanticAnnotation] = None
    """
    The type of the object that should be detected, only considered if technique is equal to Type.

    .. note:: Defaults to ``None``; kept as ``Type[...]`` (not ``Optional``) because
        ormatic cannot map ``Optional[Type]`` and would otherwise drop this column.
    """
    region: Optional[Region] = None
    """
    The region in which the object should be detected
    """

    @property
    def _action_plan(self) -> PlanNode:
        return execute_single(DetectingMotion(query=self._build_query()))

    def _build_query(self) -> PerceptionQuery:
        """
        Build the perception query from this action's parameters.

        :return: The perception query that the detection motion answers.
        """
        if not self.object_sem_annotation and not self.region:
            raise AttributeError(
                "Either a Semantic Annotation or a Region must be provided."
            )
        region_bb = (
            self.region.area.as_bounding_box_collection_in_frame(
                self.robot.root
            ).bounding_box
            if self.region
            else BoundingBox(
                origin=HomogeneousTransformationMatrix(reference_frame=self.robot.root),
                min_x=-1,
                min_y=-1,
                min_z=0,
                max_x=3,
                max_y=3,
                max_z=3,
            )
        )
        object_sem_annotation = (
            self.object_sem_annotation or SemanticEnvironmentAnnotation
        )
        return PerceptionQuery(object_sem_annotation, region_bb, self.robot, self.world)


@dataclass
class MoveToReach(ActionDescription):
    """
    Let the robot move to a position facing the target and reach with a end_effector.
    """

    target_pose_offset_robot: Pose2D
    """
    The pose where the robot should stand with regard to the end_effector target pose. 2D since z-axis is not relevant.
    """

    hip_rotation: float
    """
    Additional yaw applied to the orientation facing the target directly.
    """

    target_pose_end_effector: Pose
    """
    Pose that should be reached by the end_effector.
    """

    grasp_description: GraspDescription
    """
    The semantic description for the reaching.
    """

    @property
    def _action_plan(self) -> PlanNode:
        grasp_orientation = self.grasp_description.grasp_orientation()
        target_pose = Pose(
            self.target_pose_end_effector.to_position(),
            (
                self.target_pose_end_effector.to_rotation_matrix()
                @ grasp_orientation.to_rotation_matrix()
            ).to_quaternion(),
            self.target_pose_end_effector.reference_frame,
        )
        return sequential(
            [
                NavigateAction(self.standing_pose),
                MoveManipulatorAction(
                    target_pose,
                    self.grasp_description.end_effector,
                    allow_gripper_collision=False,
                ),
            ]
        )

    @property
    def standing_pose(self) -> Pose:
        """
        Calculates the pose where the robot should stand to reach the target.

        :return: The calculated standing pose on the floor.
        """
        reference_T_target = self.target_pose_end_effector.to_homogeneous_matrix()
        target_V_robot = -Vector3(
            x=self.target_pose_offset_robot.x, y=self.target_pose_offset_robot.y
        )
        target_V_robot.scale(1.0)
        world_z = Vector3.Z()
        target_R_robot_pointing_to_target = RotationMatrix.from_vectors(
            x=target_V_robot, z=world_z
        ) @ RotationMatrix.from_rpy(yaw=self.hip_rotation)

        target_T_robot = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            point=Point3(
                x=self.target_pose_offset_robot.x,
                y=self.target_pose_offset_robot.y,
            ),
            rotation_matrix=target_R_robot_pointing_to_target,
            reference_frame=self.target_pose_end_effector.reference_frame,
        )
        reference_T_robot = reference_T_target @ target_T_robot
        world_T_robot = self.world.transform(
            reference_T_robot.to_pose(), self.world.root
        )
        return Pose.from_xyz_rpy(
            x=world_T_robot.x,
            y=world_T_robot.y,
            yaw=world_T_robot.yaw,
            reference_frame=self.world.root,
        )
