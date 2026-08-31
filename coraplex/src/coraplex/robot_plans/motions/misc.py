from __future__ import annotations

from dataclasses import dataclass, field

from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import NodeArtifacts, Task
from giskardpy.motion_statechart.ros_context import RosContextExtension
from typing_extensions import Optional

from coraplex.datastructures.enums import ExecutionType
from coraplex.perception import PerceptionInterface, PerceptionQuery
from coraplex.plans.executables import GiskardExecutable
from coraplex.robot_plans.motions.base import BaseMotion

# %% perceiving inside the motion chart


@dataclass(eq=False, repr=False)
class PerceptionTask(Task):
    """
    Motion statechart node that answers a perception query and writes what it saw into
    the world.

    The node adds no motion constraints. The query is answered on the node's first tick,
    so the whole detection takes one tick however long the source needs to reply, and the
    node then observes ``TRUE`` so the surrounding sequence continues.

    ..warning:: That tick blocks until the source replies, which on the real robot holds
        up the control loop for as long as the pipeline takes to answer.
    """

    query: PerceptionQuery = field(kw_only=True)
    """
    What to look for and where.
    """

    execution_type: Optional[ExecutionType] = field(kw_only=True)
    """
    Which source answers the query.

    Carried by the node rather than read from the execution environment, because on the
    real robot the chart is answered in the controller's process, where that environment
    does not exist. None when the chart was built without one, which :meth:`build`
    rejects.
    """

    perception_source: Optional[PerceptionInterface] = field(init=False, default=None)
    """
    The source answering the query, resolved during :meth:`build`.
    """

    _detections_applied: bool = field(init=False, default=False, repr=False)
    """
    Whether the query has already been answered and written into the world.
    """

    accept_first_if_multiple: bool = False
    """
    Whether several candidates may be resolved by taking the first one.

    When False, several candidates raise
    :class:`~coraplex.exceptions.UnidentifiedDetections` instead of being chosen between.
    """

    def build(self, context: MotionStatechartContext) -> NodeArtifacts:
        self.perception_source = PerceptionInterface.for_execution_type(
            self.execution_type,
            context.require_extension(RosContextExtension).ros_node,
        )
        return NodeArtifacts()

    def on_start(self, context: MotionStatechartContext) -> None:
        self._detections_applied = False

    def on_tick(
        self, context: MotionStatechartContext
    ) -> Optional[ObservationStateValues]:
        if self._detections_applied:
            return ObservationStateValues.TRUE
        detection = self.perception_source.detect(
            self.query, self.accept_first_if_multiple
        )
        detection.apply_to(
            self.query.world, trust_orientation=self.query.trust_detected_orientation
        )
        self._detections_applied = True
        return ObservationStateValues.TRUE


@dataclass
class DetectingMotion(BaseMotion):
    """
    Detects the object a perception query asks for and moves it to where it was seen.

    The detection runs inside the motion statechart, so it merges with the surrounding
    motions and a motion planned after it binds the corrected pose when it starts.
    """

    query: PerceptionQuery
    """
    Query for the perception system that should be answered.
    """

    accept_first_if_multiple: bool = False
    """
    Whether several candidates may be resolved by taking the first one.

    When False, several candidates raise
    :class:`~coraplex.exceptions.UnidentifiedDetections` instead of being chosen between.
    """

    @property
    def _motion_chart(self) -> Task:
        return PerceptionTask(
            query=self.query,
            execution_type=GiskardExecutable.execution_type,
            accept_first_if_multiple=self.accept_first_if_multiple,
        )
