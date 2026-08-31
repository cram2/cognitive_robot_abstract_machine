from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import numpy.typing as npt
from random_events.product_algebra import Event, SimpleEvent
from trimesh import Trimesh
from trimesh.util import concatenate
from typing_extensions import (
    Dict,
    Any,
    Self,
    Optional,
    List,
    Iterable,
    Iterator,
    Generic,
    Type,
    TypeVar,
)
from typing_extensions import TYPE_CHECKING

from krrood.adapters.json_serializer import SubclassJSONSerializer, to_json, from_json
from semantic_digital_twin.exceptions import MismatchingWorld
from semantic_digital_twin.world_description.geometry import (
    Shape,
    AxisAlignedBox,
    VolumetricBoundingBox,
    Color,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3

BoxT = TypeVar("BoxT", bound=AxisAlignedBox)
"""
The bounding-box type a :class:`BoundingBoxCollection` holds --
:class:`VolumetricBoundingBox` for a volumetric collection, :class:`PlanarBoundingBox`
for a planar one.
"""

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )
    from semantic_digital_twin.world import World

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

    def dye_shapes(self, color: Color):
        """
        Dye all shapes in this collection with the given color.

        :param color: The color to dye the shapes with.
        """
        for shape in self.shapes:
            shape.color = color

    def transform_all_shapes_to_own_frame(self):
        """
        Transform all shapes into this collections' frame in-place.
        """
        if self.reference_frame is None:
            return
        for shape in self.shapes:
            self._transform_shape_to_reference_frame(shape)

    def _transform_shape_to_reference_frame(self, shape: Shape) -> None:
        """
        Transform ``shape``'s origin into this collection's reference frame in-place.

        A shape without a reference frame adopts the collection's frame. Cross-frame
        transforms are logged; transforming across worlds raises
        :class:`MismatchingWorld`.
        """
        origin_reference_frame = shape.origin.reference_frame
        if origin_reference_frame is None:
            shape.origin.reference_frame = self.reference_frame
            return

        if origin_reference_frame == self.reference_frame:
            return

        if self.reference_frame is None or self.reference_frame._world is None:
            return

        if origin_reference_frame._world != self.reference_frame._world:
            raise MismatchingWorld(
                expected_world=origin_reference_frame._world,
                given_world=self.reference_frame._world,
            )

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
            self._transform_shape_to_reference_frame(shape)
        self.shapes.append(shape)

    def copy_without_reference_frame(self) -> ShapeCollection:
        """
        Creates a copy of this shape collection without the reference frame.
        """
        return ShapeCollection(
            shapes=[shape.copy_without_reference_frame() for shape in self.shapes]
        )

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
        Provides the bounding box collection for this entity given a transformation
        matrix as origin.

        :param origin: The origin to express the bounding boxes from.
        :returns: A collection of bounding boxes in world-space coordinates.
        """
        world_bboxes = []

        for shape in self.shapes:
            if shape.origin.reference_frame is None:
                continue
            local_bb: VolumetricBoundingBox = shape.local_frame_bounding_box
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
        Provides the bounding box collection for this entity in the given reference
        frame.

        :param reference_frame: The reference frame to express the bounding boxes in.
        :returns: A collection of bounding boxes in world-space coordinates.
        """
        return self.as_bounding_box_collection_at_origin(
            HomogeneousTransformationMatrix(reference_frame=reference_frame)
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            **super().to_json(),
            "shapes": [to_json(shape) for shape in self.shapes],
        }

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(shapes=[from_json(d, **kwargs) for d in data["shapes"]])

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

    @property
    def scale(self):
        return (
            self.as_bounding_box_collection_at_origin(
                HomogeneousTransformationMatrix(reference_frame=self.reference_frame)
            )
            .bounding_box()
            .scale
        )

    @property
    def min_point(self) -> Point3:
        return Point3.from_iterable(
            self.combined_mesh.bounds[0], reference_frame=self.reference_frame
        )

    @property
    def max_point(self) -> Point3:
        return Point3.from_iterable(self.combined_mesh.bounds[1], self.reference_frame)


@dataclass
class BoundingBoxCollection(Generic[BoxT], ShapeCollection):
    """
    A collection of axis-aligned bounding boxes, sharing one reference frame.

    Generic over the box type -- :class:`VolumetricBoundingBox` for a volumetric collection,
    :class:`PlanarBoundingBox` for a planar one -- since every operation here depends only
    on the shared :class:`AxisAlignedBox` interface. ``as_shapes``/``from_shapes``
    additionally need :meth:`AxisAlignedBox.as_shape`, so they are only meaningful for
    a :class:`VolumetricBoundingBox` collection.
    """

    shapes: List[BoxT] = field(default_factory=list)

    def __post_init__(self):
        if not self.reference_frame:
            raise ValueError("BoundingBoxCollection must have a reference frame.")
        for box in self.bounding_boxes:
            assert (
                box.origin.reference_frame == self.reference_frame
            ), "All bounding boxes must have the same reference frame."

    def __iter__(self) -> Iterator[BoxT]:
        return iter(self.bounding_boxes)

    @property
    def bounding_boxes(self) -> List[BoxT]:
        return self.shapes

    @property
    def event(self) -> Event:
        """
        :return: The bounding boxes as a random event.
        """
        return Event.from_simple_sets(
            *[box.simple_event for box in self.bounding_boxes]
        )

    def merge(self, other: Self) -> Self:
        """
        Merge another bounding box collection into this one.

        :param other: The other bounding box collection.
        :return: The merged bounding box collection.
        """
        assert (
            self.reference_frame == other.reference_frame
        ), "The reference frames of the bounding box collections must be the same."
        return type(self)(
            reference_frame=self.reference_frame,
            shapes=self.bounding_boxes + other.bounding_boxes,
        )

    @classmethod
    def merge_all(
        cls,
        collections: Iterable[BoundingBoxCollection[BoxT]],
        reference_frame: KinematicStructureEntity,
    ) -> BoundingBoxCollection[BoxT]:
        """
        Merge a sequence of bounding box collections into one.

        :param collections: The collections to merge, in the given ``reference_frame``.
        :param reference_frame: The reference frame of the result, and of every
            collection in ``collections``.
        :return: The merged collection, empty if ``collections`` is empty.
        """
        result = cls([], reference_frame)
        for collection in collections:
            result = result.merge(collection)
        return result

    @classmethod
    def from_simple_event(
        cls,
        box_type: Type[BoxT],
        reference_frame: KinematicStructureEntity,
        simple_event: SimpleEvent,
        keep_surface: bool = False,
    ) -> BoundingBoxCollection[BoxT]:
        """
        Create a collection of bounding boxes from a simple random event.

        :param box_type: The bounding box type to build --
            :class:`VolumetricBoundingBox` for a volumetric collection,
            :class:`PlanarBoundingBox` for a planar one.
        :param reference_frame: The reference frame of the bounding boxes.
        :param simple_event: The random event.
        :param keep_surface: Whether to keep boxes that are infinitely thin.
        :return: The bounding box collection.
        """
        result = []
        for combination in itertools.product(
            *(simple_event[axis.value].simple_sets for axis in box_type.axes())
        ):
            origin_coordinates = np.array(
                [interval.center() for interval in combination]
            )
            lower = (
                np.array([interval.lower for interval in combination])
                - origin_coordinates
            )
            upper = (
                np.array([interval.upper for interval in combination])
                - origin_coordinates
            )
            origin = HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=Point3.from_iterable(_padded_to_3d(origin_coordinates)),
                reference_frame=reference_frame,
            )
            box = box_type.from_array_bounds(lower, upper, origin)
            if not keep_surface and any(
                np.isclose(dimension, 0) for dimension in box.dimensions
            ):
                continue
            result.append(box)
        return cls(result, reference_frame)

    @classmethod
    def from_event(
        cls,
        box_type: Type[BoxT],
        reference_frame: KinematicStructureEntity,
        event: Event,
    ) -> Self:
        """
        Create a collection of bounding boxes from a random event.

        :param box_type: The bounding box type to build.
        :param reference_frame: The reference frame of the bounding boxes.
        :param event: The random event.
        :return: The bounding box collection.
        """
        return cls(
            [
                box
                for simple_event in event.simple_sets
                for box in cls.from_simple_event(
                    box_type, reference_frame, simple_event
                ).bounding_boxes
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
        return cls(
            [bb.transform_to_origin(bb.origin) for bb in local_bbs],
            shapes.reference_frame,
        )

    def as_shapes(self) -> ShapeCollection:
        return ShapeCollection(
            [box.as_shape() for box in self.bounding_boxes],
            self.reference_frame,
        )

    def bounding_box(self) -> BoxT:
        """
        :return: The box that contains every bounding box in this collection.
        """
        bounds = [box.to_array_bounds() for box in self.bounding_boxes]
        lower = np.min([bound.lower for bound in bounds], axis=0)
        upper = np.max([bound.upper for bound in bounds], axis=0)
        box_type = type(self.bounding_boxes[0])
        return box_type.from_array_bounds(
            lower,
            upper,
            HomogeneousTransformationMatrix(reference_frame=self.reference_frame),
        )


def _padded_to_3d(
    coordinates: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    :param coordinates: A 2- or 3-element coordinate array.
    :return: The same coordinates, padded with trailing zeros to 3 elements.
    """
    return np.pad(coordinates, (0, 3 - len(coordinates)))
