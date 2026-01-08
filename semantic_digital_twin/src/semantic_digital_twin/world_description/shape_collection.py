from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from random_events.product_algebra import Event, SimpleEvent
from trimesh import Trimesh
from trimesh.util import concatenate
from typing_extensions import Dict, Any, Self, Optional, List, Iterator
from typing_extensions import TYPE_CHECKING

from krrood.adapters.json_serializer import SubclassJSONSerializer
from .geometry import Shape, BoundingBox
from ..datastructures.variables import SpatialVariables
from ..spatial_types import HomogeneousTransformationMatrix, Point3

if TYPE_CHECKING:
    from .world_entity import KinematicStructureEntity
    from ..world import World

logger = logging.getLogger(__name__)


@dataclass
class ShapeCollection(SubclassJSONSerializer):
    """
    A collection of shapes.
    """

    shapes: List[Shape] = field(default_factory=list)
    """
    The shapes contained in this collection.
    """

    reference_frame: Optional[KinematicStructureEntity] = None
    """
    Backreference to the kinematic structure entity this collection belongs to.
    """

    @property
    def world(self) -> Optional[World]:
        """
        The world this collection belongs to.
        """
        if self.reference_frame is not None:
            return self.reference_frame._world
        return None

    def transform_all_shapes_to_own_frame(self):
        """
        Transform all shapes into this collections' frame in-place.
        """
        if self.reference_frame is None:
            return
        for shape in self.shapes:
            self._transform_to_own_frame(shape)

    def _transform_to_own_frame(self, shape: Shape):
        """
        Transform the shape to this collections' frame in-place.
        :param shape: The shape to transform.
        """
        if shape.origin.reference_frame is None:
            # If we don’t have a world, fall back to the owning body/frame
            shape.origin.reference_frame = self.reference_frame
        elif (
            self.reference_frame is not None
            and shape.origin.reference_frame != self.reference_frame
            and self.reference_frame._world is not None
        ):
            logger.warning(
                f"Transformed shape {shape} to {self.reference_frame} since it was in a different "
                f"reference frame than the collection."
            )
            shape.origin = self.reference_frame._world.transform(
                shape.origin,
                self.reference_frame,
            )

    def __getitem__(self, index: int) -> Shape:
        return self.shapes[index]

    def __len__(self) -> int:
        return len(self.shapes)

    def __iter__(self) -> Iterator[Shape]:
        return iter(self.shapes)

    def __contains__(self, shape: Shape) -> bool:
        return shape in self.shapes

    def append(self, shape: Shape):
        if self.world is not None:
            self._transform_to_own_frame(
                shape,
            )
        self.shapes.append(shape)

    @cached_property
    def combined_mesh(self) -> Trimesh:
        """
        Combines all shapes into a single mesh, applying the respective transformations.
        :return: A single Trimesh representing the combined collision geometry.
        """
        transformed_meshes = []
        for shape in self.shapes:
            transform = shape.origin.to_np()
            mesh = shape.mesh.copy()
            mesh.apply_transform(transform)
            transformed_meshes.append(mesh)
        return concatenate(transformed_meshes)

    def as_bounding_box_collection_at_origin(
        self, origin: HomogeneousTransformationMatrix
    ) -> BoundingBoxCollection:
        """
        Provides the bounding box collection for this entity given a transformation matrix as origin.
        :param origin: The origin to express the bounding boxes from.
        :returns: A collection of bounding boxes in world-space coordinates.
        """
        world_bboxes = []

        for shape in self.shapes:
            if shape.origin.reference_frame is None:
                continue
            local_bb: BoundingBox = shape.local_frame_bounding_box
            world_bb = local_bb.transform_to_origin(origin)
            world_bboxes.append(world_bb)

        return BoundingBoxCollection(
            world_bboxes,
            origin.reference_frame,
        )

    def as_bounding_box_collection_in_frame(
        self, reference_frame: KinematicStructureEntity
    ) -> BoundingBoxCollection:
        """
        Provides the bounding box collection for this entity in the given reference frame.
        :param reference_frame: The reference frame to express the bounding boxes in.
        :returns: A collection of bounding boxes in world-space coordinates.
        """
        return self.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=reference_frame)
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "shapes": [shape.to_json() for shape in self.shapes],
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(shapes=[Shape.from_json(d, **kwargs) for d in data["shapes"]])

    def center_of_mass_in_world(self) -> Point3:
        """
        :return: The center of mass of this shape collection in the world coordinate frame.
        """
        # Center of mass in the body's local frame (collision geometry)
        com_local: np.ndarray[np.float64] = self.combined_mesh.center_mass  # (3,)
        # Transform to world frame using the body's global pose
        com = Point3(
            x=com_local[0],
            y=com_local[1],
            z=com_local[2],
            reference_frame=self.reference_frame,
        )
        return self.world.transform(com, self.world.root)

    def copy_for_world(self, world: World) -> ShapeCollection:
        new_shapes = [s.copy_for_world(world) for s in self.shapes]
        new_reference_frame = (
            world.get_kinematic_structure_entity_by_name(self.reference_frame.name)
            if self.reference_frame
            else None
        )
        return ShapeCollection(new_shapes, new_reference_frame)


@dataclass
class BoundingBoxCollection(ShapeCollection):
    """
    Dataclass for storing a collection of bounding boxes.
    """

    shapes: List[BoundingBox]

    def __post_init__(self):
        if not self.reference_frame:
            raise ValueError("BoundingBoxCollection must have a reference frame.")
        for box in self.bounding_boxes:
            assert (
                box.origin.reference_frame == self.reference_frame
            ), "All bounding boxes must have the same reference frame."

    def __iter__(self) -> Iterator[BoundingBox]:
        return iter(self.bounding_boxes)

    @property
    def bounding_boxes(self) -> List[BoundingBox]:
        return self.shapes

    @property
    def event(self) -> Event:
        """
        :return: The bounding boxes as a random event.
        """
        return Event(*[box.simple_event for box in self.bounding_boxes])

    def merge(self, other: BoundingBoxCollection) -> BoundingBoxCollection:
        """
        Merge another bounding box collection into this one.

        :param other: The other bounding box collection.
        :return: The merged bounding box collection.
        """
        assert (
            self.reference_frame == other.reference_frame
        ), "The reference frames of the bounding box collections must be the same."
        return BoundingBoxCollection(
            reference_frame=self.reference_frame,
            shapes=self.bounding_boxes + other.bounding_boxes,
        )

    def bloat(
        self, x_amount: float = 0.0, y_amount: float = 0, z_amount: float = 0
    ) -> BoundingBoxCollection:
        """
        Enlarges all bounding boxes in the collection by a given amount in all dimensions.

        :param x_amount: The amount to adjust the x-coordinates
        :param y_amount: The amount to adjust the y-coordinates
        :param z_amount: The amount to adjust the z-coordinates

        :return: The enlarged bounding box collection
        """
        return BoundingBoxCollection(
            [box.bloat(x_amount, y_amount, z_amount) for box in self.bounding_boxes],
            self.reference_frame,
        )

    @classmethod
    def from_simple_event(
        cls,
        reference_frame: KinematicStructureEntity,
        simple_event: SimpleEvent,
        keep_surface: bool = False,
    ) -> BoundingBoxCollection:
        """
        Create a list of bounding boxes from a simple random event.

        :param reference_frame: The reference frame of the bounding boxes.
        :param simple_event: The random event.
        :param keep_surface: Whether to keep events that are infinitely thin
        :return: The list of bounding boxes.
        """
        result = []
        for x, y, z in itertools.product(
            simple_event[SpatialVariables.x.value].simple_sets,
            simple_event[SpatialVariables.y.value].simple_sets,
            simple_event[SpatialVariables.z.value].simple_sets,
        ):

            bb = BoundingBox(
                x.lower,
                y.lower,
                z.lower,
                x.upper,
                y.upper,
                z.upper,
                HomogeneousTransformationMatrix.from_point_rotation_matrix(
                    point=Point3(
                        x.upper - (x.upper - x.lower) / 2,
                        y.upper - (y.upper - y.lower) / 2,
                        z.upper - (z.upper - z.lower) / 2,
                    ),
                    reference_frame=reference_frame,
                ),
            )
            if not keep_surface and (bb.depth == 0 or bb.height == 0 or bb.width == 0):
                continue
            result.append(bb)
        return BoundingBoxCollection(result, reference_frame)

    @classmethod
    def from_event(
        cls, reference_frame: KinematicStructureEntity, event: Event
    ) -> Self:
        """
        Create a list of bounding boxes from a random event.

        :param reference_frame: The reference frame of the bounding boxes.
        :param event: The random event.
        :return: The list of bounding boxes.
        """
        return cls(
            [
                box
                for simple_event in event.simple_sets
                for box in cls.from_simple_event(reference_frame, simple_event)
            ],
            reference_frame,
        )

    @classmethod
    def from_shapes(cls, shapes: ShapeCollection) -> Self:
        """
        Create a bounding box collection from a list of shapes.

        :param shapes: The list of shapes.
        :return: The bounding box collection.
        """
        if len(shapes) == 0:
            return cls(shapes=[])
        for shape in shapes:
            assert (
                shape.origin.reference_frame == shapes[0].origin.reference_frame
            ), "All shapes must have the same reference frame."

        local_bbs = [shape.local_frame_bounding_box for shape in shapes]
        reference_frame = shapes[0].origin.reference_frame
        return cls(
            [bb.transform_to_origin(bb.origin) for bb in local_bbs],
            reference_frame,
        )

    def as_shapes(self) -> ShapeCollection:
        return ShapeCollection(
            [box.as_shape() for box in self.bounding_boxes],
            self.reference_frame,
        )

    def bounding_box(self) -> BoundingBox:
        """
        Get the 8 corners of a bounding box that contains all bounding boxes in the collection.

        :return: A list of Point3 objects representing the corners of the bounding box.
        """
        all_x = [bb.min_x for bb in self.bounding_boxes] + [
            bb.max_x for bb in self.bounding_boxes
        ]
        all_y = [bb.min_y for bb in self.bounding_boxes] + [
            bb.max_y for bb in self.bounding_boxes
        ]
        all_z = [bb.min_z for bb in self.bounding_boxes] + [
            bb.max_z for bb in self.bounding_boxes
        ]
        return BoundingBox(
            min(all_x),
            min(all_y),
            min(all_z),
            max(all_x),
            max(all_y),
            max(all_z),
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                reference_frame=self.reference_frame
            ),
        )
