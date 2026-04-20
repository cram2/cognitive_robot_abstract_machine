from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta

from typing_extensions import Optional, Type, Any

from pycram.datastructures.enums import DetectionTechnique
from pycram.plans.failures import PerceptionObjectNotFound
from pycram.plans.factories import sequential, execute_single, try_in_order
from pycram.robot_plans.actions.base import ActionDescription
from pycram.robot_plans.actions.core.misc import DetectAction
from pycram.robot_plans.actions.core.navigation import NavigateAction, LookAtAction
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


@dataclass
class SearchAction(ActionDescription):
    """
    Searches for a target object around the given location.
    """

    target_location: Pose
    """
    Location around which to look for a target object.
    """

    object_sem_annotation: Type[SemanticAnnotation]
    """
    Type of the object which is searched for.
    """

    def execute(self) -> None:

        # go to a location where the target location is visible
        self.add_subplan(
            execute_single(
                NavigateAction(
                    next(
                        iter(CostmapLocation(target=self.target_location, visible=True))
                    )
                )
            )
        ).perform()

        # define searching cone
        target_base = self.world.transform(self.target_location, self.world.root)

        target_base_left = deepcopy(target_base)
        target_base_left.y -= 0.5

        target_base_right = deepcopy(target_base)
        target_base_right.y += 0.5

        self.add_subplan(
            searching := try_in_order(
                [
                    sequential(
                        [
                            LookAtAction(target),
                            DetectAction(
                                DetectionTechnique.TYPES,
                                object_sem_annotation=self.object_sem_annotation,
                            ),
                        ]
                    )
                    for target in [target_base, target_base_left, target_base_right]
                ]
            )
        )

        # get the found objects
        old_annotation_ids = {
            annotation.id for annotation in self.world.semantic_annotations
        }
        searching.perform()
        new_annotation_ids = {
            annotation.id for annotation in self.world.semantic_annotations
        } - old_annotation_ids

        if not new_annotation_ids:
            raise PerceptionObjectNotFound(self)

    def validate(
        self, result: Optional[Any] = None, max_wait_time: Optional[timedelta] = None
    ):
        pass
