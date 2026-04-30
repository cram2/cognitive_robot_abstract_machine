from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

import krrood.symbolic_math.symbolic_math as sm
from semantic_digital_twin.world_description.world_entity import Connection
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

    def add_to_world(self, world: World):
        """
        Defines the symbolic transformation matrix for the PCC segment.
        """
        super().add_to_world(world)

        # Access DOFs through the world using the stored UUIDs
        kappa_dof = world.get_degree_of_freedom_by_id(self.kappa_dof_id)
        phi_dof = world.get_degree_of_freedom_by_id(self.phi_dof_id)

        # Use symbolic_math wrappers for CasADi expressions
        kappa = sm.Scalar.from_casadi_sx(kappa_dof.variables.position.casadi_sx)
        phi = sm.Scalar.from_casadi_sx(phi_dof.variables.position.casadi_sx)
        length = self.segment_length
        theta = kappa * length

        # Singular handling for a straight segment (kappa close to zero)
        it_is_straight = sm.abs(kappa) < 1e-8

        # Position formulas (px, py, pz)
        px = sm.if_else(
            it_is_straight, 0, (sm.cos(phi) * (1 - sm.cos(theta))) / (kappa + 1e-12)
        )
        py = sm.if_else(
            it_is_straight, 0, (sm.sin(phi) * (1 - sm.cos(theta))) / (kappa + 1e-12)
        )
        pz = sm.if_else(it_is_straight, length, sm.sin(theta) / (kappa + 1e-12))

        # Rotation matrix components
        c_p = sm.cos(phi)
        s_p = sm.sin(phi)
        c_t = sm.cos(theta)
        s_t = sm.sin(theta)
        # fmt: off
        r11 = c_p**2 * (c_t - 1) + 1;  r12 = s_p * c_p * (c_t - 1);     r13 = c_p * s_t
        r21 = s_p * c_p * (c_t - 1);   r22 = c_p**2 * (1 - c_t) + c_t;  r23 = s_p * s_t
        r31 = -c_p * s_t;              r32 = -s_p * s_t;                r33 = c_t
        # fmt: on

        # Build 4x4 matrix
        matrix = sm.vstack(
            [
                sm.hstack([r11, r12, r13, px]),
                sm.hstack([r21, r22, r23, py]),
                sm.hstack([r31, r32, r33, pz]),
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

    def add_to_world(self, world: World):
        """
        Integrates the Cosserat differential equations along the length of the
        segment to compute the child's pose relative to the parent.
        """
        super().add_to_world(world)

        bx = world.get_degree_of_freedom_by_id(self.bending_x_dof_id)
        by = world.get_degree_of_freedom_by_id(self.bending_y_dof_id)
        tor = world.get_degree_of_freedom_by_id(self.torsion_dof_id)
        ext = world.get_degree_of_freedom_by_id(self.extension_dof_id)

        ux = sm.Scalar.from_casadi_sx(bx.variables.position.casadi_sx)
        uy = sm.Scalar.from_casadi_sx(by.variables.position.casadi_sx)
        uz = sm.Scalar.from_casadi_sx(tor.variables.position.casadi_sx)
        vz = sm.Scalar.from_casadi_sx(ext.variables.position.casadi_sx)

        # xi vector: [bending_x, bending_y, torsion, shear_x, shear_y, extension]
        xi = sm.Vector([ux, uy, uz, 0, 0, vz])

        def hat_operator(strain_vector: sm.Vector) -> sm.Matrix:
            """
            Maps a 6D strain vector to a 4x4 se(3) Lie Algebra matrix.
            This matrix represents the local derivative of the transformation
            along the rod's length.
            """
            u = strain_vector[:3]
            v = strain_vector[3:]
            return sm.vstack(
                [
                    sm.hstack([0, -u[2], u[1], v[0]]),
                    sm.hstack([u[2], 0, -u[0], v[1]]),
                    sm.hstack([-u[1], u[0], 0, v[2]]),
                    sm.hstack([0, 0, 0, 0]),
                ]
            )

        # RK4 Integration along the length
        accumulated_transform = sm.Matrix.eye(4)
        num_steps = 10
        ds = self.segment_length / num_steps

        for _ in range(num_steps):
            k1 = accumulated_transform @ hat_operator(xi)
            k2 = (accumulated_transform + (ds / 2) * k1) @ hat_operator(xi)
            k3 = (accumulated_transform + (ds / 2) * k2) @ hat_operator(xi)
            k4 = (accumulated_transform + ds * k3) @ hat_operator(xi)
            accumulated_transform = accumulated_transform + (ds / 6) * (
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
