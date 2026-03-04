from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np
from numpy._typing import NDArray
from typing_extensions import Self, TypeVar

from semantic_digital_twin.spatial_types import Point3, RotationMatrix


@dataclass
class NPMatrix3x3:
    data: NDArray

    def __post_init__(self):
        assert self.data.shape == (3, 3), "Matrix must be 3x3"

    def __matmul__(self, other: GenericMatrix3x3Type) -> GenericMatrix3x3Type:
        return NPMatrix3x3(data=self.data @ other.data)


GenericMatrix3x3Type = TypeVar("GenericMatrix3x3Type", bound=NPMatrix3x3)


@dataclass
class NPVector3:
    data: NDArray

    def __post_init__(self):
        assert self.data.shape == (3,), "Vector must be 3-dimensional"

    @classmethod
    def from_values(cls, x: float, y: float, z: float) -> Self:
        """Construct from scalar components (x, y, z)."""
        return cls(data=np.array([x, y, z]))

    def to_values(self) -> Tuple[float, float, float]:
        """Return the tuple (x,y,z)"""
        return self.data[0], self.data[1], self.data[2]

    def as_matrix(self) -> NPMatrix3x3:
        """Return a 3x3 matrix with the vector on the diagonal."""
        return NPMatrix3x3(data=np.diag(self.data))


@dataclass(eq=False)
class PrincipalMoments(NPVector3):
    """
    Represents the three principal moments of inertia (I1, I2, I3) about the principal axes of the body.
    A principal moment is the eigenvalue of the inertia tensor corresponding to a principal axis.
    """

    def __post_init__(self):
        assert np.all(self.data >= 0), "Moments must be non-negative"

    @classmethod
    def from_values(cls, i1: float, i2: float, i3: float) -> Self:
        """Construct from scalar components (I1, I2, I3)."""
        return cls(data=np.array([i1, i2, i3]))


@dataclass
class PrincipalAxes(NPMatrix3x3):
    """
    The principal axes of the inertia tensor is a 3x3 matrix where each column is a principal axis.
    A principal axis is an eigenvector of the inertia tensor corresponding to a principal moment of inertia.
    """

    def __post_init__(self):
        super().__post_init__()
        assert np.allclose(
            self.data.T @ self.data, np.eye(3), atol=1e-10
        ), "Principal axes must be orthonormal"
        assert np.isclose(
            np.linalg.det(self.data), 1.0, atol=1e-10
        ), "Principal axes must form a right-handed coordinate system"

    def T(self) -> Self:
        return PrincipalAxes(data=self.data.T)

    @classmethod
    def from_rotation_matrix(cls, rotation_matrix: RotationMatrix):
        """Construct PrincipalAxes from a RotationMatrix."""
        data = rotation_matrix.to_np()[:3, :3]
        return cls(data=data)

    def to_rotation_matrix(self) -> RotationMatrix:
        """Convert PrincipalAxes to a RotationMatrix."""
        return RotationMatrix(self.data)


@dataclass(eq=False)
class InertiaTensor(NPMatrix3x3):
    """
    Represents the inertia tensor of a body in a given coordinate frame.
    The inertia tensor is a symmetric positive semi-definite 3x3 matrix.
    https://en.wikipedia.org/wiki/Moment_of_inertia#Definition_2
    """

    def __post_init__(self):
        super().__post_init__()
        diag = np.diag(self.data)
        assert np.all(
            diag >= 0.0
        ), "Diagonal elements of inertia tensor must be non-negative"
        assert np.allclose(
            self.data, self.data.T, atol=1e-10
        ), "Inertia tensor must be symmetric"

    @classmethod
    def from_values(
        cls, ixx: float, iyy: float, izz: float, ixy: float, ixz: float, iyz: float
    ) -> Self:
        """
        Construct inertia tensor from individual components.
        :param ixx: Moment of inertia about x-axis
        :param iyy: Moment of inertia about y-axis
        :param izz: Moment of inertia about z-axis
        :param ixy: Product of inertia xy
        :param ixz: Product of inertia xz
        :param iyz: Product of inertia yz
        :return: InertiaTensor
        """
        data = np.array(
            [
                [ixx, ixy, ixz],
                [ixy, iyy, iyz],
                [ixz, iyz, izz],
            ]
        )
        return InertiaTensor(data=data)

    def to_values(self) -> Tuple[float, float, float, float, float, float]:
        """
        Return the individual components of the inertia tensor.
        :return: (ixx, iyy, izz, ixy, ixz, iyz)
        """
        ixx = self.data[0, 0]
        iyy = self.data[1, 1]
        izz = self.data[2, 2]
        ixy = self.data[0, 1]
        ixz = self.data[0, 2]
        iyz = self.data[1, 2]
        return ixx, iyy, izz, ixy, ixz, iyz

    @classmethod
    def from_principal_moments_and_axes(
        cls,
        moments: PrincipalMoments,
        axes: PrincipalAxes = PrincipalAxes(data=np.eye(3)),
    ) -> Self:
        """
        Construct from principal representation:
        R * I_diag * R^T
        """
        data = axes @ moments.as_matrix() @ axes.T()
        return InertiaTensor(data=data.data)

    def to_principal_moments_and_axes(
        self, sorted_array: Optional[NDArray] = None
    ) -> Tuple[PrincipalMoments, PrincipalAxes]:
        """
        Decompose inertia tensor into principal moments and axes.
        The principal moments will be sorted if sorted_array is provided.
        :param sorted_array: Optional array to sort the principal moments and axes (e.g., [2, 1, 0], [4.5, 6.4, 1.2]).
        :return: (PrincipalMoments, PrincipalAxes)
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.data)
        if sorted_array is not None:
            sorted_indices = np.argsort(np.argsort(sorted_array))
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors @ np.eye(3)[sorted_indices].T
        if np.linalg.det(eigenvectors) < 0:
            eigenvectors[:, 2] *= -1.0
        moments = PrincipalMoments.from_values(*eigenvalues)
        axes = PrincipalAxes(data=eigenvectors)
        return moments, axes


@dataclass
class Inertial:
    """
    Represents the inertial properties of a body. https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-inertial
    """

    mass: float = 1.0
    """
    The mass of the body in kilograms.
    """

    center_of_mass: Point3 = field(default_factory=Point3)
    """
    The center of mass of the body. If a force acts through the COM, the body experiences pure translation, no torque
    """

    inertia: InertiaTensor = field(
        default_factory=lambda: InertiaTensor.from_values(1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    )
    """
    The inertia tensor of the body about its center of mass, expressed in the body's local coordinate frame.
    """
