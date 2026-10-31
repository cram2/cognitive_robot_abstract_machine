from __future__ import annotations

from dataclasses import dataclass, field


from krrood.ormatic.utils import classproperty
from coraplex.datastructures.enums import (
    DetectionState,
    DetectionTechnique,
    PerceptionSource,
)
from coraplex.perception import PerceptionQuery, PerceptionTask
from cramph.node import StatechartNode
from coraplex.robot_plans.actions.base import SequentialAction
from coraplex.robot_plans.actions.core.exceptions import DetectionTargetMissing
from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import MoveManipulatorAction
from coraplex.robot_plans.mixins import HasApproachesGraspPoses, MovesToolCenterPoint
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
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SemanticEnvironmentAnnotation,
)


@dataclass(eq=False, repr=False, kw_only=True)
class DetectAction(SequentialAction):
    """
    Detects an object and writes the pose it is seen at into the world.

    The action fails if no such object is seen.
    """

    technique: DetectionTechnique
    """
    The technique that should be used for detection.
    """

    state: DetectionState | None = None
    """
    The state of the detection, e.g Start Stop for continues perception.
    """

    object_sem_annotation: type[SemanticAnnotation] = None
    """
    The type of the object that should be detected, only considered if technique is
    equal to Type.

    .. note:: Defaults to ``None``; kept as ``Type[...]`` (not ``Optional``) because
        ormatic cannot map ``Optional[Type]`` and would otherwise drop this column.
    """

    detection_box: VolumetricBoundingBox | None = None
    """
    The box in which the object should be detected, None for the box around the robot's
    base that :attr:`default_detection_distance` spans.
    """

    trust_detected_orientation: bool = True
    """
    Whether to trust the perception source's detected orientation.

    When False, only the detected position corrects the object's pose in the world; its
    existing orientation is kept. See
    :attr:`~coraplex.perception.PerceptionQuery.trust_detected_orientation`.
    """

    accept_first_if_multiple: bool = False
    """
    Whether several candidates may be resolved by taking the first one.

    When False, several candidates raise
    :class:`~coraplex.exceptions.UnidentifiedDetections` instead of being chosen between.
    """

    perception_source: PerceptionSource = field(
        default=PerceptionSource.WORLD_MODEL, init=False
    )
    """
    What answers the detection.
    """

    @classproperty
    def default_detection_distance(cls) -> float:
        """
        :return: How far from the robot's base, horizontally in every direction and
            upwards, an object is detected when no :attr:`detection_box` is given.
        """
        return 3.0

    def create_steps(self) -> list[StatechartNode]:
        return [
            PerceptionTask(
                query=self._build_query(),
                answered_by=self.perception_source,
                accept_first_if_multiple=self.accept_first_if_multiple,
            )
        ]

    def _build_query(self) -> PerceptionQuery:
        """
        Build the perception query from this action's parameters.

        :return: The perception query that the detection motion answers.
        :raises DetectionTargetMissing: If neither an object type nor a box is given.
        """
        if not self.object_sem_annotation and self.detection_box is None:
            raise DetectionTargetMissing(self)
        return PerceptionQuery(
            self.object_sem_annotation or SemanticEnvironmentAnnotation,
            (
                self._box_around_robot()
                if self.detection_box is None
                else self.detection_box
            ),
            self.robot,
            self.world,
            trust_detected_orientation=self.trust_detected_orientation,
        )

    def _box_around_robot(self) -> VolumetricBoundingBox:
        """
        :return: The box reaching :attr:`default_detection_distance` from the robot's
            base horizontally in every direction and upwards from it.
        """
        distance = self.default_detection_distance
        return VolumetricBoundingBox(
            origin=HomogeneousTransformationMatrix(reference_frame=self.robot.root),
            min_x=-distance,
            min_y=-distance,
            min_z=0,
            max_x=distance,
            max_y=distance,
            max_z=distance,
        )


@dataclass(eq=False, repr=False, kw_only=True)
class MoveToReach(MovesToolCenterPoint, SequentialAction, HasApproachesGraspPoses):
    """
    Let the robot move to a position facing the target and reach with a end_effector.
    """

    target_pose_offset_robot: Pose2D
    """
    The pose where the robot should stand with regard to the end_effector target pose.

    2D since z-axis is not relevant.
    """

    hip_rotation: float
    """
    Additional yaw applied to the orientation facing the target directly.
    """

    reference_T_grasp: Pose
    """
    The grasp frame the end effector is to reach.
    """

    end_effector: EndEffector
    """
    The end effector that should reach it.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [
            NavigateAction(target_location=self.standing_pose),
            MoveManipulatorAction(
                target_pose=self.end_effector.tool_frame_goal_for_grasp(
                    self.reference_T_grasp
                ),
                end_effector=self.end_effector,
                allow_gripper_collision=False,
                position_threshold=self.position_threshold,
                orientation_threshold=self.orientation_threshold,
            ),
        ]

    @property
    def standing_pose(self) -> Pose:
        """
        Calculates the pose where the robot should stand to reach the target.

        :return: The calculated standing pose on the floor.
        """
        reference_T_target = self.reference_T_grasp.homogeneous_matrix
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
            reference_frame=self.reference_T_grasp.reference_frame,
        )
        reference_T_robot = reference_T_target @ target_T_robot
        world_T_robot = self.world.transform(reference_T_robot.pose, self.world.root)
        return Pose.from_xyz_rpy(
            x=world_T_robot.x,
            y=world_T_robot.y,
            yaw=world_T_robot.yaw,
            reference_frame=self.world.root,
        )
