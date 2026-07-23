from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Self
from uuid import UUID

import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import (
    Connection,
    KinematicStructureEntity,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World


@dataclass(eq=False)
class PiecewiseConstantCurvatureConnection(Connection):
    """
    A continuum connection based on the Piecewise Constant Curvature (PCC) model.

    Models the segment as a perfect circular arc using a closed-form geometric
    mapping. Assumes constant curvature throughout the segment length.
    It is suitable for modeling soft robots where bending is approximately uniform.
    """

    kappa_dof_id: UUID = field(kw_only=True)
    """UUID of the Degree of Freedom representing curvature (kappa = 1/radius)."""

    phi_dof_id: UUID = field(kw_only=True)
    """UUID of the Degree of Freedom representing the plane of bending."""

    segment_length: float = field(kw_only=True)
    """The physical arc length of this specific segment."""

    @classmethod
    def create_with_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: Optional[PrefixedName] = None,
        *args,
        **kwargs,
    ) -> Self:
        """
        Factory method to instantiate a PCC connection in the world.
        """
        return cls(parent=parent, child=child, name=name, **kwargs)

    def add_to_world(self, world: World):
        """
        Defines the symbolic transformation matrix for the PCC segment.
        """
        super().add_to_world(world)

        # Access DOFs through the world using the stored UUIDs
        kappa_dof = world.get_degree_of_freedom_by_id(self.kappa_dof_id)
        phi_dof = world.get_degree_of_freedom_by_id(self.phi_dof_id)

        kappa = kappa_dof.variables.position
        phi = phi_dof.variables.position
        length = self.segment_length
        theta = kappa * length

        # Singular handling for a straight segment (kappa close to zero)
        it_is_straight = sm.abs(kappa) < 1e-8

        # Position formulas (px, py, pz)
        position_x = sm.if_else(
            it_is_straight, 0, (sm.cos(phi) * (1 - sm.cos(theta))) / (kappa + 1e-12)
        )
        position_y = sm.if_else(
            it_is_straight, 0, (sm.sin(phi) * (1 - sm.cos(theta))) / (kappa + 1e-12)
        )
        position_z = sm.if_else(it_is_straight, length, sm.sin(theta) / (kappa + 1e-12))

        # Rotation matrix components
        cos_phi = sm.cos(phi)
        sin_phi = sm.sin(phi)
        cos_theta = sm.cos(theta)
        sin_theta = sm.sin(theta)

        # fmt: off
        rotation_11 = cos_phi**2 * (cos_theta - 1) + 1;     rotation_12 = sin_phi * cos_phi * (cos_theta - 1);       rotation_13 = cos_phi * sin_theta
        rotation_21 = sin_phi * cos_phi * (cos_theta - 1);  rotation_22 = cos_phi**2 * (1 - cos_theta) + cos_theta;  rotation_23 = sin_phi * sin_theta
        rotation_31 = -cos_phi * sin_theta;                 rotation_32 = -sin_phi * sin_theta;                      rotation_33 = cos_theta
        # fmt: on

        # Build 4x4 matrix
        matrix = sm.vstack(
            [
                sm.hstack([rotation_11, rotation_12, rotation_13, position_x]),
                sm.hstack([rotation_21, rotation_22, rotation_23, position_y]),
                sm.hstack([rotation_31, rotation_32, rotation_33, position_z]),
                sm.hstack([0, 0, 0, 1]),
            ]
        )

        self._kinematics = HomogeneousTransformationMatrix(matrix.casadi_sx)
        self._kinematics.child_frame = self.child

    @property
    def active_dofs(self):
        """Returns the list of degrees of freedom controlling this PCC segment."""
        return [
            self._world.get_degree_of_freedom_by_id(self.kappa_dof_id),
            self._world.get_degree_of_freedom_by_id(self.phi_dof_id),
        ]


@dataclass(eq=False)
class CosseratRodConnection(Connection):
    """
    A connection implementing Cosserat Rod Theory.

    Treats the soft segment as an elastic rod that can bend, twist, and extend.
    Unlike PCC, this model also supports torsion and non-circular bending shapes.
    Kinematics are computed using 4th-order Runge-Kutta (RK4) numerical integration.
    """

    bending_x_dof_id: UUID = field(kw_only=True)
    """UUID of the Degree of Freedom for the bending rate around the X-axis (ux)."""

    bending_y_dof_id: UUID = field(kw_only=True)
    """UUID of the Degree of Freedom for the bending rate around the Y-axis (uy)."""

    torsion_dof_id: UUID = field(kw_only=True)
    """UUID of the Degree of Freedom for the twisting rate around the Z-axis (uz)."""

    extension_dof_id: UUID = field(kw_only=True)
    """UUID of the Degree of Freedom for the linear stretching rate along the Z-axis (vz)."""

    segment_length: float = field(kw_only=True)
    """The intrinsic rest length of the rod segment."""

    @classmethod
    def create_with_dofs(
        cls,
        world: World,
        parent: KinematicStructureEntity,
        child: KinematicStructureEntity,
        name: Optional[PrefixedName] = None,
        *args,
        **kwargs,
    ) -> Self:
        """
        Factory method to instantiate a Cosserat connection in the world.
        """
        return cls(parent=parent, child=child, name=name, **kwargs)

    def add_to_world(self, world: World):
        """
        Integrates the Cosserat differential equations along the length of the
        segment to compute the child's pose relative to the parent.
        """
        super().add_to_world(world)

        bending_x_dof = world.get_degree_of_freedom_by_id(self.bending_x_dof_id)
        bending_y_dof = world.get_degree_of_freedom_by_id(self.bending_y_dof_id)
        torsion_dof = world.get_degree_of_freedom_by_id(self.torsion_dof_id)
        extension_dof = world.get_degree_of_freedom_by_id(self.extension_dof_id)

        bending_x = bending_x_dof.variables.position
        bending_y = bending_y_dof.variables.position
        torsion = torsion_dof.variables.position
        extension = extension_dof.variables.position

        # xi vector: [bending_x, bending_y, torsion, shear_x, shear_y, extension]
        xi = sm.Vector([bending_x, bending_y, torsion, 0, 0, extension])

        def hat_operator(strain_vector: sm.Vector) -> sm.Matrix:
            """
            Maps a 6D strain vector to a 4x4 se(3) Lie Algebra matrix.
            This matrix represents the local derivative of the transformation
            along the rod's length.
            """
            angular_strain = strain_vector[:3]
            linear_strain = strain_vector[3:]
            return sm.vstack(
                [
                    sm.hstack(
                        [0, -angular_strain[2], angular_strain[1], linear_strain[0]]
                    ),
                    sm.hstack(
                        [angular_strain[2], 0, -angular_strain[0], linear_strain[1]]
                    ),
                    sm.hstack(
                        [-angular_strain[1], angular_strain[0], 0, linear_strain[2]]
                    ),
                    sm.hstack([0, 0, 0, 0]),
                ]
            )

        # RK4 Integration along the length
        accumulated_transform = sm.Matrix.eye(4)
        num_steps = 10
        step_length = self.segment_length / num_steps

        for _ in range(num_steps):
            k1 = accumulated_transform @ hat_operator(xi)
            k2 = (accumulated_transform + (step_length / 2) * k1) @ hat_operator(xi)
            k3 = (accumulated_transform + (step_length / 2) * k2) @ hat_operator(xi)
            k4 = (accumulated_transform + step_length * k3) @ hat_operator(xi)
            accumulated_transform = accumulated_transform + (step_length / 6) * (
                k1 + 2 * k2 + 2 * k3 + k4
            )

        self._kinematics = HomogeneousTransformationMatrix(
            accumulated_transform.casadi_sx
        )
        self._kinematics.child_frame = self.child

    @property
    def active_dofs(self):
        """Returns the list of degrees of freedom controlling this rod segment."""
        return [
            self._world.get_degree_of_freedom_by_id(self.bending_x_dof_id),
            self._world.get_degree_of_freedom_by_id(self.bending_y_dof_id),
            self._world.get_degree_of_freedom_by_id(self.torsion_dof_id),
            self._world.get_degree_of_freedom_by_id(self.extension_dof_id),
        ]
