from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Tuple, Type

from coraplex.datastructures.enums import DetectionTechnique
from coraplex.locations.factories import visibility_location
from coraplex.plans.factories import sequential, try_in_order
from coraplex.plans.plan_node import PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import NavigateAction, LookAtAction
from krrood.entity_query_language.factories import a, variable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


@dataclass
class SearchAction(ActionDescription):
    """
    Searches for a target object around the given location.

    The robot drives to a pose the location is visible from and looks at the location,
    then to either side of it, until the object is detected.
    """

    target_location: Pose
    """
    Location around which to look for a target object.
    """

    object_semantic_annotation: Type[SemanticAnnotation]
    """
    Type of the object which is searched for.
    """

    sideway_look_offset: Tuple[float, ...] = (0.0, -0.5, 0.5)
    """
    Distances along the target location's y-axis that are looked at, in order.
    """

    @property
    def _action_plan(self) -> PlanNode:
        return sequential(
            [
                a(NavigateAction)(
                    target_location=variable(
                        Pose,
                        domain=visibility_location(
                            target=self.target_location, context=self.context
                        ),
                    )
                ),
                try_in_order(
                    [
                        self._look_and_detect(sideways_offset)
                        for sideways_offset in self.sideway_look_offset
                    ]
                ),
            ]
        )

    def _look_and_detect(self, sideways_offset: float) -> PlanNode:
        """
        Look at the target location moved sideways, and detect the object there.

        :param sideways_offset: Distance along the target location's y-axis.
        :return: The root node of the plan looking there.
        """
        reference_T_target = self.target_location.to_homogeneous_matrix()
        target_T_look = HomogeneousTransformationMatrix.from_xyz_rpy(
            y=sideways_offset, reference_frame=self.target_location.reference_frame
        )
        return sequential(
            [
                LookAtAction((reference_T_target @ target_T_look).to_pose()),
                DetectAction(
                    DetectionTechnique.TYPES,
                    object_sem_annotation=self.object_semantic_annotation,
                ),
            ]
        )
