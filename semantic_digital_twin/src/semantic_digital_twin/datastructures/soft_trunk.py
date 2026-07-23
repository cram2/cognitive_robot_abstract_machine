from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

from semantic_digital_twin.robots.robot_parts import (
    Arm,
    EndEffector,
)

from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.soft_connections import (
    PiecewiseConstantCurvatureConnection,
    CosseratRodConnection,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Cylinder, Color
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Quaternion,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass
class SoftTrunkSection:
    """
    Encapsulates the physical and visual properties of a single soft section.
    """

    length: float
    """The rest length of the section in meters."""

    radius: float
    """The radius of the cylinder representing the section's volume."""

    resolution: int
    """The number of discrete rigid segments used to approximate the continuous curve."""


@dataclass(eq=False, kw_only=True)
class SoftEndEffector(EndEffector):
    """Concrete implementation of EndEffector for soft robots."""

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(cls, robot_root):
        pass

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> list:
        return []


@dataclass(eq=False, kw_only=True)
class SoftArm(Arm):
    """Concrete implementation of Arm for soft robots."""

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(cls, robot_root):
        pass

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> list:
        return []


@dataclass(eq=False, kw_only=True)
class SoftTrunk(SemanticAnnotation):
    """
    A mathematical and semantic representation of a soft continuum manipulator.

    This class enables the construction of soft robotic structures that do not have
    discrete rigid joints. Instead, it uses mathematical continuum models to
    simulate bending and twisting behavior. Two models are supported:

    Piecewise Constant Curvature (PCC): A geometric model assuming sections bend
    into perfect circular arcs.

    Cosserat Rod Theory: A differential model that supports internal twisting
    (torsion) and stretching (extension).
    """

    name: PrefixedName
    """The unique prefixed name assigned to this soft trunk instance."""

    root: Body
    """The base body representing the physical root of the trunk."""

    _world: World
    """Reference to the parent world containing this robot."""

    kappa_dofs: list[DegreeOfFreedom] = field(default_factory=list)
    """List of curvature DOFs (1/radius) ordered from base to tip."""

    phi_dofs: list[DegreeOfFreedom] = field(default_factory=list)
    """List of bending plane DOFs ordered from base to tip."""

    bending_x_dofs: list[DegreeOfFreedom] = field(default_factory=list)
    """List of bending DOFs around the local X-axis ordered from base to tip."""

    bending_y_dofs: list[DegreeOfFreedom] = field(default_factory=list)
    """List of bending DOFs around the local Y-axis ordered from base to tip."""

    torsion_dofs: list[DegreeOfFreedom] = field(default_factory=list)
    """List of axial torsion (twisting) DOFs ordered from base to tip."""

    extension_dofs: list[DegreeOfFreedom] = field(default_factory=list)
    """List of longitudinal extension (stretching) DOFs ordered from base to tip."""

    arms: list[Arm] = field(default_factory=list)
    """List of semantic Arm structures associated with this trunk."""

    def __post_init__(self):
        super().__post_init__()

    @property
    def piecewise_constant_curvature_sections(
        self,
    ) -> list[tuple[DegreeOfFreedom, DegreeOfFreedom]]:
        """Returns a list of (kappa_dof, phi_dof) pairs, ordered from base to tip."""
        return list(zip(self.kappa_dofs, self.phi_dofs))

    @property
    def cosserat_sections(self) -> list[tuple[DegreeOfFreedom, ...]]:
        """Returns a list of (bx, by, torsion, extension) tuples, ordered from base to tip."""
        return list(
            zip(
                self.bending_x_dofs,
                self.bending_y_dofs,
                self.torsion_dofs,
                self.extension_dofs,
            )
        )

    @classmethod
    def build_piecewise_constant_curvature(
        cls,
        world: World,
        sections: list[SoftTrunkSection],
    ) -> SoftTrunk:
        """
        Builds a soft robot using the Piecewise Constant Curvature (PCC) model.

        PCC is a geometric approximation that assumes soft segments deform into perfect circular arcs.

        Ref: Robert J Webster III and Bryan A Jones,
            “Design and kinematic modeling of constant curvature continuum robots: A review,”
            The International Journal of Robotics Research, vol. 29, no. 13, pp. 1661-1683,
            2010.

        :param world: The world from which to create the robot view.

        :param sections: A list of section configurations defining the robot's morphology.

        :return: A SoftTrunk robot view.
        """

        prefix = "piecewise_constant_curvature"
        with world.modify_world():
            root_body = Body(name=PrefixedName(name="base", prefix=prefix))
            world.add_body(root_body)

            trunk = cls(
                name=PrefixedName(name="robot", prefix=prefix),
                root=root_body,
                _world=world,
            )

            prev_body = root_body
            limits = DegreeOfFreedomLimits(
                lower=DerivativeMap(position=-10.0, velocity=-10.0),
                upper=DerivativeMap(position=10.0, velocity=10.0),
            )

            for section_index, section in enumerate(sections):
                kappa = DegreeOfFreedom(
                    name=PrefixedName(f"kappa_{section_index}", prefix), limits=limits
                )
                phi = DegreeOfFreedom(
                    name=PrefixedName(f"phi_{section_index}", prefix), limits=limits
                )
                world.add_degree_of_freedom(kappa)
                world.add_degree_of_freedom(phi)

                # Store references to preserve order
                trunk.kappa_dofs.append(kappa)
                trunk.phi_dofs.append(phi)

                segment_length = section.length / section.resolution
                for segment_index in range(section.resolution):
                    color_val = (section_index + 1) / len(sections)
                    visual_shape = Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=-segment_length / 2
                        ),
                        width=section.radius * 2,
                        height=segment_length,
                        color=Color(0.0, color_val, 1.0 - color_val, 1.0),
                    )

                    curr_body = Body(
                        name=PrefixedName(
                            f"sec{section_index}_seg{segment_index}", prefix
                        ),
                        visual=ShapeCollection([visual_shape]),
                        collision=ShapeCollection([visual_shape]),
                    )
                    world.add_body(curr_body)

                    connection = PiecewiseConstantCurvatureConnection(
                        parent=prev_body,
                        child=curr_body,
                        kappa_dof_id=kappa.id,
                        phi_dof_id=phi.id,
                        segment_length=segment_length,
                    )
                    world.add_connection(connection)
                    prev_body = curr_body

            effector = SoftEndEffector(
                name=PrefixedName("effector", prefix),
                root=prev_body,
                tool_frame=prev_body,
                front_facing_orientation=Quaternion(w=1.0),
                _world=world,
            )
            world.add_semantic_annotation(effector)

            arm = SoftArm(
                name=PrefixedName("arm", prefix),
                root=root_body,
                tip=prev_body,
                _world=world,
                end_effector=effector,
            )
            trunk.arms.append(arm)
            world.add_semantic_annotation(trunk)

        return trunk

    @classmethod
    def build_cosserat(
        cls,
        world: World,
        sections: list[SoftTrunkSection],
    ) -> SoftTrunk:
        """
        Builds a soft robot using the differential Cosserat Rod Theory.

        This is a differential model that solves the robot's shape by integrating local
        strain rates. Unlike PCC, it can represent axial torsion (twisting) and
        longitudinal extension (stretching).

        Ref: Kelan Luo, “Modeling of continuum robots: A review,”
            Journal of Physics: Conference Series, vol. 2634, pp.012029,
            Nov. 2023.

        :param world: The world from which to create the robot view.

        :param sections: A list of section configurations defining the morphology.

        :return: A SoftTrunk robot view
        """

        prefix = "cosserat"
        with world.modify_world():
            root_body = Body(name=PrefixedName(name="base", prefix=prefix))
            world.add_body(root_body)

            trunk = cls(
                name=PrefixedName(name="robot", prefix=prefix),
                root=root_body,
                _world=world,
            )

            prev_body = root_body
            strain_limits = DegreeOfFreedomLimits(
                lower=DerivativeMap(position=-10.0, velocity=-10.0),
                upper=DerivativeMap(position=10.0, velocity=10.0),
            )
            extension_limits = DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.1, velocity=-10.0),
                upper=DerivativeMap(position=3.0, velocity=10.0),
            )

            for section_index, section in enumerate(sections):
                bending_x = DegreeOfFreedom(
                    name=PrefixedName(f"bending_x_{section_index}", prefix),
                    limits=strain_limits,
                )
                bending_y = DegreeOfFreedom(
                    name=PrefixedName(f"bending_y_{section_index}", prefix),
                    limits=strain_limits,
                )
                torsion = DegreeOfFreedom(
                    name=PrefixedName(f"torsion_{section_index}", prefix),
                    limits=strain_limits,
                )
                extension = DegreeOfFreedom(
                    name=PrefixedName(f"extension_{section_index}", prefix),
                    limits=extension_limits,
                )

                for dof in [bending_x, bending_y, torsion, extension]:
                    world.add_degree_of_freedom(dof)

                world.state[extension.id].position = 1.0

                # Store references to preserve order
                trunk.bending_x_dofs.append(bending_x)
                trunk.bending_y_dofs.append(bending_y)
                trunk.torsion_dofs.append(torsion)
                trunk.extension_dofs.append(extension)

                segment_length = section.length / section.resolution
                for segment_index in range(section.resolution):
                    color_val = (section_index + 1) / len(sections)
                    visual_shape = Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=-segment_length / 2
                        ),
                        width=section.radius * 2,
                        height=segment_length,
                        color=Color(color_val, 0.2, 1.0 - color_val, 1.0),
                    )

                    curr_body = Body(
                        name=PrefixedName(
                            f"sec{section_index}_seg{segment_index}", prefix
                        ),
                        visual=ShapeCollection([visual_shape]),
                        collision=ShapeCollection([visual_shape]),
                    )
                    world.add_body(curr_body)

                    connection = CosseratRodConnection(
                        parent=prev_body,
                        child=curr_body,
                        bending_x_dof_id=bending_x.id,
                        bending_y_dof_id=bending_y.id,
                        torsion_dof_id=torsion.id,
                        extension_dof_id=extension.id,
                        segment_length=segment_length,
                    )
                    world.add_connection(connection)
                    prev_body = curr_body

            effector = SoftEndEffector(
                name=PrefixedName("effector", prefix),
                root=prev_body,
                tool_frame=prev_body,
                front_facing_orientation=Quaternion(w=1.0),
                _world=world,
            )
            world.add_semantic_annotation(effector)

            arm = SoftArm(
                name=PrefixedName("arm", prefix),
                root=root_body,
                tip=prev_body,
                _world=world,
                end_effector=effector,
            )
            trunk.arms.append(arm)
            world.add_semantic_annotation(trunk)

        return trunk
