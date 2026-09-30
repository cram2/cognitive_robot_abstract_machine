"""
Shared numeric pose and geometry measurements for visualization bundles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy
from scipy.spatial.transform import Rotation
from semantic_digital_twin.spatial_types import Point3, Pose
from semantic_digital_twin.world_description.geometry import Scale
from typing_extensions import (
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Tuple,
    TYPE_CHECKING,
)

POSE_PRECISION = 5
"""
Decimal places a pose is rounded to before it is published or recorded.
"""

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.shape_collection import ShapeCollection
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )


@runtime_checkable
class CarriesAMeshFile(Protocol):
    """
    A shape whose geometry lives in a file of its own.

    Structural, because a world built in code writes its generated geometry out to a
    file just as a loaded mesh names the file it came from.
    """

    filename: str
    """
    The geometry file referenced by the shape.
    """


@runtime_checkable
class CarriesBodyGeometry(Protocol):
    """
    An entity shaped like a body: geometry it is drawn with and geometry it collides
    with.

    Structural, so a mimic body reads the same way a real one does.
    """

    visual: ShapeCollection
    """
    Geometry used to draw the entity.
    """

    collision: ShapeCollection
    """
    Geometry used to represent collisions.
    """


@runtime_checkable
class CarriesAreaGeometry(Protocol):
    """
    An entity shaped like a region: the area it spans is its only geometry.
    """

    area: ShapeCollection
    """
    Geometry describing the region's extent.
    """


# %% numeric publication


@dataclass(frozen=True)
class NumericPose:
    """
    A pose copied into plain numbers for recording and display.
    """

    position: tuple[float, float, float]
    """
    Translation in metres.
    """

    quaternion: tuple[float, float, float, float]
    """
    Orientation in ``x, y, z, w`` order.
    """

    @classmethod
    def of_matrix(cls, matrix: numpy.ndarray) -> NumericPose:
        """
        Read a numeric homogeneous transform.

        :param matrix: The transform to publish.
        :return: Its translation and orientation as plain numbers.
        """
        return cls(
            position=tuple(float(value) for value in matrix[:3, 3]),
            quaternion=tuple(
                float(value) for value in Rotation.from_matrix(matrix[:3, :3]).as_quat()
            ),
        )

    @classmethod
    def of_pose(cls, pose: Pose) -> NumericPose:
        """
        Read a constant native pose at the publication boundary.

        :param pose: The native pose to publish.
        :return: Its numeric coordinates.
        """
        return cls.of_matrix(pose.to_np())

    def rounded(self, precision: int = POSE_PRECISION) -> list[float]:
        """
        Return rounded position followed by quaternion coordinates.

        :param precision: Decimal places kept in each coordinate.
        :return: The seven coordinates in viewer order.
        """
        return [round(value, precision) for value in (*self.position, *self.quaternion)]

    @property
    def label(self) -> str:
        """
        Format the position and quaternion for a readable query answer.
        """
        return "(%.2f, %.2f, %.2f) q(%.2f, %.2f, %.2f, %.2f)" % (
            *self.position,
            *self.quaternion,
        )


def shape_collections_of(
    entity: KinematicStructureEntity,
) -> Tuple[ShapeCollection, ...]:
    """
    The shape collections an entity's geometry is read from, most descriptive first.

    A body draws its visual geometry and collides with its collision geometry; a region
    has only the area it spans. Publishing either to the viewer starts from the same
    question — where is its geometry — so both are answered here.

    :param entity: The body or region whose geometry is asked for.
    """
    if isinstance(entity, CarriesBodyGeometry):
        return (entity.visual, entity.collision)
    if isinstance(entity, CarriesAreaGeometry):
        return (entity.area,)
    return ()


def mesh_file_of(entity: KinematicStructureEntity) -> Optional[str]:
    """
    The file an entity's own geometry lives in, or None for one built from primitives.

    :param entity: The body or region whose geometry is inspected.
    """
    for shape_collection in shape_collections_of(entity):
        for shape in shape_collection.shapes:
            if isinstance(shape, CarriesAMeshFile) and shape.filename:
                return shape.filename
    return None


def measure_body(entity: KinematicStructureEntity) -> Optional[Scale]:
    """
    Measure an entity from the first of its shape collections that has any shapes.

    Checks :attr:`Body.visual` before :attr:`Body.collision` (a region has only its
    area), using :attr:`ShapeCollection.scale`, which measures any shape type from its
    bounding box rather than relying on a shape-specific scale attribute.

    :param entity: The body or region to measure.
    :return: The entity's size along each world axis in metres, or None when every
        collection is empty.
    """
    for shape_collection in shape_collections_of(entity):
        if not shape_collection.shapes:
            continue
        scale = shape_collection.scale
        return Scale(x=float(scale.x), y=float(scale.y), z=float(scale.z))
    return None


def rounded_scale(scale: Scale, precision: int) -> List[float]:
    """
    A scale as ``[x, y, z]``, rounded for publication.

    :param scale: The scale to round.
    :param precision: Number of decimal places to round each axis to.
    """
    return [
        round(scale.x, precision),
        round(scale.y, precision),
        round(scale.z, precision),
    ]


def rounded_pose(
    entity: KinematicStructureEntity, precision: int = POSE_PRECISION
) -> List[float]:
    """
    An entity's world pose as ``[x, y, z, qx, qy, qz, qw]``, rounded for publication.

    Reads the pose numerically, so publishing one builds no symbolic expression.

    :param entity: The body or region whose world pose is read.
    :param precision: Number of decimal places to round each value to.
    """
    world = entity._world
    return NumericPose.of_matrix(
        world.compute_forward_kinematics_np(world.root, entity)
    ).rounded(precision)


def position_label(position: Point3) -> str:
    """
    A position's coordinates, formatted to two decimal places for display.

    :class:`Point3` has no plain-value ``__repr__`` of its own (it is a CasADi-symbolic
    type), so the coordinates are read out explicitly.

    :param position: The position to format.
    """
    return "(%.2f, %.2f, %.2f)" % tuple(position.to_np().tolist()[:3])


def pose_label(pose: Pose) -> str:
    """
    A pose's position and orientation, formatted to two decimal places for display.

    Formats the position exactly as :func:`position_label` does, so an answer showing
    both a position and a pose reads consistently.

    :param pose: The pose to format.
    """
    return NumericPose.of_pose(pose).label
