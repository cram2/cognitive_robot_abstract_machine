from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Self

from coraplex.datastructures.enums import DetectionTechnique
from coraplex.locations.locations import VisibilityLocation
from coraplex.robot_plans.actions.base import SequentialAction
from coraplex.robot_plans.actions.core.misc import DetectAction
from coraplex.robot_plans.actions.core.navigation import LookAtAction, NavigateAction
from cramph.composites import TryInOrder
from cramph.context import StatechartContext
from cramph.node import StatechartNode
from krrood.entity_query_language.factories import a, variable
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import VolumetricBoundingBox
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation

# %% looking for an object at one place


@dataclass(eq=False, repr=False, kw_only=True)
class LookAtAndDetectAction(SequentialAction):
    """
    Looks at a place, then detects an object while looking there.
    """

    look_at: LookAtAction
    """
    The look at the place.
    """

    detect: DetectAction
    """
    The detection once the robot looks at the place.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [self.look_at, self.detect]


# %% searching for an object


@dataclass(eq=False, repr=False, kw_only=True)
class SearchAction(SequentialAction):
    """
    Searches for an object: drives to where it may be seen from, then tries its looks in
    turn until one detects the object.

    The action fails if none does.
    """

    navigate: NavigateAction
    """
    The drive to where the robot looks from.
    """

    looks: list[LookAtAndDetectAction]
    """
    The looks tried in turn, until one detects the object.
    """

    @classmethod
    def looking_around(
        cls,
        target_location: Pose,
        object_semantic_annotation: type[SemanticAnnotation],
        context: StatechartContext,
        cross_arm_length: float = 0.5,
        detection_margin: float = 0.5,
    ) -> Self:
        """
        A search driving to a pose `target_location` is visible from, looking at
        `target_location` and then at the ends of a cross around it along the x- and
        y-axes of its reference frame.

        :param target_location: Where to look for the object.
        :param object_semantic_annotation: The type of the object searched for.
        :param context: The context of the plan the search runs in, whose robot performs
            it.
        :param cross_arm_length: How far from `target_location` the ends of the cross
            lie.
        :param detection_margin: How far beyond the places looked at an object may lie
            and still be detected.
        :return: The search, with the standing pose left open.
        """
        reference_frame = target_location.reference_frame
        half_extent = cross_arm_length + detection_margin
        detection_box = VolumetricBoundingBox(
            origin=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=target_location.position, reference_frame=reference_frame
            ),
            min_x=-half_extent,
            min_y=-half_extent,
            min_z=-half_extent,
            max_x=half_extent,
            max_y=half_extent,
            max_z=half_extent,
        )
        cross = [
            (0, 0),
            (cross_arm_length, 0),
            (-cross_arm_length, 0),
            (0, cross_arm_length),
            (0, -cross_arm_length),
        ]
        return cls(
            navigate=a(NavigateAction)(
                target_location=variable(
                    Pose,
                    domain=VisibilityLocation(
                        target_pose=target_location, context=context
                    ),
                )
            ),
            looks=[
                LookAtAndDetectAction(
                    look_at=LookAtAction(
                        target=Pose.from_xyz_rpy(
                            target_location.x + x_offset,
                            target_location.y + y_offset,
                            target_location.z,
                            reference_frame=reference_frame,
                        )
                    ),
                    detect=DetectAction(
                        technique=DetectionTechnique.TYPES,
                        object_sem_annotation=object_semantic_annotation,
                        detection_box=detection_box,
                    ),
                )
                for x_offset, y_offset in cross
            ],
        )

    def create_steps(self) -> list[StatechartNode]:
        return [self.navigate, TryInOrder(nodes=list(self.looks))]
