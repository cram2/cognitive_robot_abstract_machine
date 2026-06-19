from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

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
    Vector3,
    Quaternion,
)

if TYPE_CHECKING:
    from semantic_digital_twin.world import World


@dataclass
class SoftTrunkSection:
    """
    Encapsulates the physical and visual properties of a single soft section.

    :length: The rest length of the section in meters.

    :radius: The radius of the cylinder representing the section's volume.

    :resolution: The number of discrete rigid segments used to approximate the
                continuous curve for visualization and collision checking.
    """

    length: float
    radius: float
    resolution: int


@dataclass(eq=False, kw_only=True)
class SoftEndEffector(EndEffector):
    """Concrete implementation of EndEffector for soft robots."""

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(cls, robot_root):
        pass

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List:
        return []


@dataclass(eq=False, kw_only=True)
class SoftArm(Arm):
    """Concrete implementation of Arm for soft robots."""

    @classmethod
    def setup_default_configuration_in_world_below_robot_root(cls, robot_root):
        pass

    def setup_hardware_interfaces(self):
        pass

    def setup_joint_states(self) -> List:
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
    root: Body
    _world: World

    # Base to Tip ordering in lists
    kappa_dofs: List[DegreeOfFreedom] = field(default_factory=list)
    phi_dofs: List[DegreeOfFreedom] = field(default_factory=list)
    bending_x_dofs: List[DegreeOfFreedom] = field(default_factory=list)
    bending_y_dofs: List[DegreeOfFreedom] = field(default_factory=list)
    torsion_dofs: List[DegreeOfFreedom] = field(default_factory=list)
    extension_dofs: List[DegreeOfFreedom] = field(default_factory=list)

    # Semantic components
    arms: List[Arm] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()

    @property
    def piecewise_constant_curvature_sections(
        self,
    ) -> List[Tuple[DegreeOfFreedom, DegreeOfFreedom]]:
        """Returns a list of (kappa_dof, phi_dof) pairs, ordered from base to tip."""
        return list(zip(self.kappa_dofs, self.phi_dofs))

    @property
    def cosserat_sections(self) -> List[Tuple[DegreeOfFreedom, ...]]:
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
        sections: List[SoftTrunkSection],
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

            for s, section in enumerate(sections):
                kappa = DegreeOfFreedom(
                    name=PrefixedName(f"kappa_{s}", prefix), limits=limits
                )
                phi = DegreeOfFreedom(
                    name=PrefixedName(f"phi_{s}", prefix), limits=limits
                )
                world.add_degree_of_freedom(kappa)
                world.add_degree_of_freedom(phi)

                # Store references to preserve order
                trunk.kappa_dofs.append(kappa)
                trunk.phi_dofs.append(phi)

                segment_length = section.length / section.resolution
                for i in range(section.resolution):
                    color_val = (s + 1) / len(sections)
                    visual_shape = Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=-segment_length / 2
                        ),
                        width=section.radius * 2,
                        height=segment_length,
                        color=Color(0.0, color_val, 1.0 - color_val, 1.0),
                    )

                    curr_body = Body(
                        name=PrefixedName(f"sec{s}_seg{i}", prefix),
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
        sections: List[SoftTrunkSection],
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

            for s, section in enumerate(sections):
                bx = DegreeOfFreedom(
                    name=PrefixedName(f"bending_x_{s}", prefix), limits=strain_limits
                )
                by = DegreeOfFreedom(
                    name=PrefixedName(f"bending_y_{s}", prefix), limits=strain_limits
                )
                tor = DegreeOfFreedom(
                    name=PrefixedName(f"torsion_{s}", prefix), limits=strain_limits
                )
                ext = DegreeOfFreedom(
                    name=PrefixedName(f"extension_{s}", prefix), limits=extension_limits
                )

                for dof in [bx, by, tor, ext]:
                    world.add_degree_of_freedom(dof)

                world.state[ext.id].position = 1.0

                # Store references to preserve order
                trunk.bending_x_dofs.append(bx)
                trunk.bending_y_dofs.append(by)
                trunk.torsion_dofs.append(tor)
                trunk.extension_dofs.append(ext)

                segment_length = section.length / section.resolution
                for i in range(section.resolution):
                    color_val = (s + 1) / len(sections)
                    visual_shape = Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=-segment_length / 2
                        ),
                        width=section.radius * 2,
                        height=segment_length,
                        color=Color(color_val, 0.2, 1.0 - color_val, 1.0),
                    )

                    curr_body = Body(
                        name=PrefixedName(f"sec{s}_seg{i}", prefix),
                        visual=ShapeCollection([visual_shape]),
                        collision=ShapeCollection([visual_shape]),
                    )
                    world.add_body(curr_body)

                    connection = CosseratRodConnection(
                        parent=prev_body,
                        child=curr_body,
                        bending_x_dof_id=bx.id,
                        bending_y_dof_id=by.id,
                        torsion_dof_id=tor.id,
                        extension_dof_id=ext.id,
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
