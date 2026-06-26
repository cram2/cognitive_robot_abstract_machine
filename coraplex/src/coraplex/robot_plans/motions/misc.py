from dataclasses import dataclass, field
from typing import Optional, List

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import Task
from coraplex.perception import PerceptionQuery
from coraplex.robot_plans.motions.base import BaseMotion
from semantic_digital_twin.semantic_annotations.mixins import IsPerceivable
from semantic_digital_twin.world_description.world_entity import Body


@dataclass(eq=False, repr=False)
class PerceptionTask(Task):
    """
    Motion statechart node that runs a perception query as a payload.

    The node adds no motion constraints; on the first tick it answers the query and
    writes every detected object into the world/belief state, then reports its
    observation as ``TRUE`` so the surrounding motion sequence can continue.
    """

    query: PerceptionQuery = field(kw_only=True)
    """
    The perception query that is answered when this node runs.
    """

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        self._write_into_belief(self.query.from_world())
        return ObservationStateValues.TRUE

    def _write_into_belief(self, detected_bodies: List[Body]) -> None:
        """
        Mark the semantic annotations of the detected bodies as perceived in the world.

        :param detected_bodies: The bodies returned by the perception query.
        """
        # TODO Change this to actual behaviour with RoboKudo and move it to a separate thread
        detected = set(detected_bodies)
        if not detected:
            return
        annotations = self.query.world.get_semantic_annotations_by_type(
            self.query.semantic_annotation
        )
        for annotation in annotations:
            if isinstance(annotation, IsPerceivable) and detected.intersection(
                annotation.bodies
            ):
                annotation.class_label = type(annotation).__name__


@dataclass
class DetectingMotion(BaseMotion):
    """
    Tries to detect an object in the FOV of the robot.

    The detection is performed inside the motion statechart so it merges with the
    surrounding motions; detected objects are written into the world/belief state
    rather than returned.
    """

    query: PerceptionQuery
    """
    Query for the perception system that should be answered
    """

    @property
    def _motion_chart(self) -> Task:
        return PerceptionTask(query=self.query)
