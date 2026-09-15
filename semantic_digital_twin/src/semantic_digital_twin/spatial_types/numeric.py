"""
The spatial types held as plain numbers, and the conversions that read them out.

The spatial types are CasADi-backed: reading one evaluates an expression graph whose
nodes are reference counted without atomics, while CasADi releases the GIL for the
duration of a call. Geometry that is read from any thread other than the one that owns
the world therefore has to be read out into numbers first, without building or reading
any symbolic expression on the way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from typing_extensions import List, Optional, Tuple, TYPE_CHECKING

from semantic_digital_twin.datastructures.types import NpMatrix4x4

if TYPE_CHECKING:
    from semantic_digital_twin.spatial_types.spatial_types import (
        HomogeneousTransformationMatrix,
        Pose,
    )
    from semantic_digital_twin.world_description.world_entity import (
        KinematicStructureEntity,
    )


@dataclass(frozen=True)
class NumericPoint3:
    """
    A point held as plain numbers, holding nothing symbolic.
    """

    x: float
    """
    The point's coordinate along the x-axis.
    """

    y: float
    """
    The point's coordinate along the y-axis.
    """

    z: float
    """
    The point's coordinate along the z-axis.
    """

    reference_frame: Optional[KinematicStructureEntity] = None
    """
    The frame the coordinates are expressed in.
    """

    @classmethod
    def from_coordinates(
        cls,
        coordinates: npt.NDArray[np.float64],
        reference_frame: Optional[KinematicStructureEntity] = None,
    ) -> NumericPoint3:
        """
        Read a point out of the first three entries of an array.

        :param coordinates: The array holding the point's coordinates.
        :param reference_frame: The frame the coordinates are expressed in.
        """
        return cls(
            x=float(coordinates[0]),
            y=float(coordinates[1]),
            z=float(coordinates[2]),
            reference_frame=reference_frame,
        )

    def to_np(self) -> npt.NDArray[np.float64]:
        """
        :return: The point as a homogeneous 4-vector, ending in 1.
        """
        return np.array([self.x, self.y, self.z, 1.0])


@dataclass(eq=False)
class NumericTransform:
    """
    A rigid transform held as plain numbers, together with the frame it is expressed in.

    Stands in for a :class:`HomogeneousTransformationMatrix` wherever the transform is
    only ever read back as numbers.
    """

    matrix: NpMatrix4x4 = field(default_factory=lambda: np.eye(4))
    """
    The 4x4 transform itself.
    """

    reference_frame: Optional[KinematicStructureEntity] = None
    """
    The frame the transform is expressed in.
    """

    @classmethod
    def from_transformation_matrix(
        cls, reference_T_body: HomogeneousTransformationMatrix
    ) -> NumericTransform:
        """
        Read a symbolic transform out into numbers.

        :param reference_T_body: The transform to read out.
        """
        return cls(
            matrix=reference_T_body.to_np(),
            reference_frame=reference_T_body.reference_frame,
        )

    @classmethod
    def identity(
        cls, reference_frame: Optional[KinematicStructureEntity] = None
    ) -> NumericTransform:
        """
        The transform from a frame to itself.

        :param reference_frame: The frame the transform is expressed in.
        """
        return cls(matrix=np.eye(4), reference_frame=reference_frame)

    @property
    def x(self) -> float:
        """
        :return: The translation along the x-axis.
        """
        return float(self.matrix[0, 3])

    @property
    def y(self) -> float:
        """
        :return: The translation along the y-axis.
        """
        return float(self.matrix[1, 3])

    @property
    def z(self) -> float:
        """
        :return: The translation along the z-axis.
        """
        return float(self.matrix[2, 3])

    def to_np(self) -> NpMatrix4x4:
        """
        :return: The transform as a 4x4 array.
        """
        return self.matrix

    def to_position(self) -> NumericPoint3:
        """
        :return: The transform's translation.
        """
        return NumericPoint3.from_coordinates(
            self.matrix[:3, 3], reference_frame=self.reference_frame
        )


@dataclass(frozen=True)
class NumericPose:
    """
    A pose read out into plain numbers, holding nothing symbolic.

    ..warning:: Read one out on the thread that owns the world, and hand only the
       result to any other thread.
    """

    position: Tuple[float, float, float]
    """
    The pose's x, y and z coordinates.
    """

    quaternion: Tuple[float, float, float, float]
    """
    The pose's orientation, as x, y, z and w.
    """

    @classmethod
    def from_transformation_matrix(cls, root_T_body: NpMatrix4x4) -> NumericPose:
        """
        Read a transformation matrix out as a position and a quaternion.

        :param root_T_body: The transform to read out.
        """
        return cls(
            position=(
                float(root_T_body[0, 3]),
                float(root_T_body[1, 3]),
                float(root_T_body[2, 3]),
            ),
            quaternion=cls._quaternion_of(root_T_body),
        )

    @classmethod
    def of_pose(cls, pose: Pose) -> NumericPose:
        """
        Read a pose out into plain numbers.

        :param pose: The pose to read out.
        """
        return cls.from_transformation_matrix(pose.to_np())

    @staticmethod
    def _quaternion_of(
        root_T_body: NpMatrix4x4,
    ) -> Tuple[float, float, float, float]:
        """
        The orientation of a transform as a quaternion, as x, y, z and w.

        Divides by the largest diagonal entry, so a half turn stays accurate.

        :param root_T_body: The transform whose orientation is converted.
        """
        homogeneous_scale = root_T_body[3, 3]
        trace = (
            root_T_body[0, 0]
            + root_T_body[1, 1]
            + root_T_body[2, 2]
            + homogeneous_scale
        )
        if trace - homogeneous_scale > 0:
            components = [
                root_T_body[2, 1] - root_T_body[1, 2],
                root_T_body[0, 2] - root_T_body[2, 0],
                root_T_body[1, 0] - root_T_body[0, 1],
                trace,
            ]
            return NumericPose._scaled(components, trace, homogeneous_scale)

        largest = int(np.argmax(np.diagonal(root_T_body)[:3]))
        following = (largest + 1) % 3
        preceding = (largest + 2) % 3
        diagonal_difference = (
            root_T_body[largest, largest]
            - (root_T_body[following, following] + root_T_body[preceding, preceding])
            + homogeneous_scale
        )
        components = [0.0, 0.0, 0.0, 0.0]
        components[largest] = diagonal_difference
        components[following] = (
            root_T_body[largest, following] + root_T_body[following, largest]
        )
        components[preceding] = (
            root_T_body[preceding, largest] + root_T_body[largest, preceding]
        )
        components[3] = (
            root_T_body[preceding, following] - root_T_body[following, preceding]
        )
        return NumericPose._scaled(components, diagonal_difference, homogeneous_scale)

    @staticmethod
    def _scaled(
        components: List[float], divisor: float, homogeneous_scale: float
    ) -> Tuple[float, float, float, float]:
        """
        Normalise the components a branch of the conversion produced.

        :param components: The unnormalised x, y, z and w components.
        :param divisor: The quantity the branch built its components from.
        :param homogeneous_scale: The transform's bottom right entry.
        """
        scale = 0.5 / math.sqrt(divisor * homogeneous_scale)
        return (
            float(components[0] * scale),
            float(components[1] * scale),
            float(components[2] * scale),
            float(components[3] * scale),
        )

    def to_position_quaternion_list(self) -> List[float]:
        """
        :return: This pose's position and orientation, as ``[x, y, z, qx, qy, qz, qw]``.
        """
        return [*self.position, *self.quaternion]

    def euclidean_distance(self, other: NumericPose) -> float:
        """
        How far apart two poses are placed.

        :param other: The pose to measure to.
        """
        return float(np.linalg.norm(np.subtract(self.position, other.position)))

    def rotational_distance(self, other: NumericPose) -> float:
        """
        The angle a pose would have to turn through to reach another's orientation.

        Measured as the shorter of the two ways round, so the result is in ``[0, pi]``.

        :param other: The pose to measure to.
        """
        alignment = abs(float(np.dot(self.quaternion, other.quaternion)))
        return 2.0 * math.acos(min(1.0, alignment))
