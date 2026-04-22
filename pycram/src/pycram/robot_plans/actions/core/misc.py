from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Any, Dict

import numpy as np
from typing_extensions import Optional, Type, Any

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import Variable
from pycram.datastructures.dataclasses import Context

from pycram.datastructures.enums import DetectionTechnique, DetectionState
from pycram.datastructures.grasp import GraspDescription
from pycram.perception import PerceptionQuery
from pycram.plans.factories import sequential
from pycram.plans.failures import NavigationGoalNotReachedError
from pycram.robot_plans import MoveManipulatorMotion
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import MoveManipulatorAction
from semantic_digital_twin.robots.abstract_robot import Manipulator
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Quaternion
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.world_entity import (
    Region,
    SemanticAnnotation,
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
    The type of the object that should be detected, only considered if technique is equal to Type
    """
    region: Optional[Region] = None
    """
    The region in which the object should be detected
    """

    def execute(self) -> None:
        if not self.object_sem_annotation and self.region:
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
        if not self.object_sem_annotation:
            self.object_sem_annotation = SemanticEnvironmentAnnotation
        query = PerceptionQuery(
            self.object_sem_annotation, region_bb, self.robot, self.world
        )

        return query.from_world()

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        return
        # if not result:
        #     raise PerceptionObjectNotFound(self.object_designator, self.technique, self.region)


@dataclass
class MoveToReach(ActionDescription):
    """
    Let the robot move to a position facing the target and reach with a manipulator.
    """

    robot_x: float
    """
    The x position where the robot should stand w. r. t. the target.
    """

    robot_y: float
    """
    The y position where the robot should stand w. r. t. the target.
    """

    hip_rotation: float
    """
    Additional yaw applied to the orientation facing the target directly.
    """

    target_pose: Pose
    """
    Pose that should be reached.
    """

    grasp_description: GraspDescription
    """
    The semantic description for the reaching.
    """

    def execute(self):
        self.add_subplan(
            sequential(
                [
                    NavigateAction(self.standing_pose),
                    MoveManipulatorAction(
                        self.target_pose,
                        self.grasp_description.manipulator,
                        allow_gripper_collision=False,
                    ),
                ]
            )
        ).perform()

    @property
    def manipulator(self) -> Manipulator:
        """
        Returns the manipulator that is used for reaching.

        :return: The manipulator.
        """
        return self.grasp_description.manipulator

    @property
    def standing_pose(self) -> Pose:
        """
        Calculates the pose where the robot should stand to reach the target.

        :return: The calculated standing pose.
        """
        target_homogeneous_matrix = self.target_pose.to_homogeneous_matrix()
        relative_position = Point3(self.robot_x, self.robot_y, 0)
        standing_position = target_homogeneous_matrix @ relative_position
        standing_position.z = self.robot.root.global_pose.z

        difference_x = self.target_pose.x - standing_position.x
        difference_y = self.target_pose.y - standing_position.y
        base_yaw = np.arctan2(difference_y, difference_x)
        total_yaw = base_yaw + self.hip_rotation

        standing_pose = Pose(
            standing_position,
            Quaternion.from_rpy(0, 0, total_yaw),
            reference_frame=self.robot.root,
        )
        return standing_pose

    @staticmethod
    def post_condition(
        variables: Dict[str, Variable], context: Context, kwargs: Dict[str, Any]
    ) -> SymbolicExpression:
        target_pose = variables["target_pose"]

        target_homogeneous_matrix = target_pose.to_homogeneous_matrix()
        relative_position = Point3(variables["robot_x"], variables["robot_y"], 0)
        standing_position = target_homogeneous_matrix @ relative_position
        standing_position.z = context.robot.root.global_pose.z

        difference_x = target_pose.x - standing_position.x
        difference_y = target_pose.y - standing_position.y
        base_yaw = np.arctan2(difference_y, difference_x)
        total_yaw = base_yaw + variables["hip_rotation"]

        expected_pose = Pose(
            standing_position,
            Quaternion.from_rpy(0, 0, total_yaw),
            reference_frame=context.robot.root,
        )

        root_T_robot_base = context.world.transform(expected_pose, context.world.root)
        return np.allclose(
            root_T_robot_base, context.robot.base.root.global_pose.to_np(), atol=0.1
        )
