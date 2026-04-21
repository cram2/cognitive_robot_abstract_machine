from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Optional, Type, Any

from pycram.datastructures.enums import (
    DetectionTechnique,
    DetectionState,
    ExecutionType,
)
from pycram.motion_executor import MotionExecutor
from pycram.perception import PerceptionQuery
from pycram.robot_plans.actions.base import ActionDescription
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
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

        return (
            query.from_world()
            if MotionExecutor.execution_type == ExecutionType.SIMULATED
            else query.from_robokudo()
        )

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        return
        # if not result:
        #     raise PerceptionObjectNotFound(self.object_designator, self.technique, self.region)
