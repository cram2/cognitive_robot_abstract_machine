from __future__ import annotations
from dataclasses import dataclass
from typing import Self, TYPE_CHECKING, List, Tuple

from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Arm, Manipulator
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
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


@dataclass(eq=False)
class SoftTrunk(AbstractRobot):
    """
    A robot representation for continuum soft manipulators.

    This class enables the construction of soft robotic arms that do not have
    discrete rigid joints. Instead, it uses mathematical continuum models to
    simulate bending and twisting behavior. Two models are supported:

    Piecewise Constant Curvature (PCC): A geometric model assuming sections bend
    into perfect circular arcs.

    Cosserat Rod Theory: A differential model that supports internal twisting
    (torsion) and stretching (extension).
    """

    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(
            name=PrefixedName(name="soft_trunk", prefix="robot"),
            root=Body(name=PrefixedName(name="base", prefix="soft_trunk")),
            _world=world,
        )

    # Internal setup methods
    def _setup_semantic_annotations(self):
        pass

    def _setup_collision_rules(self):
        pass

    def _setup_velocity_limits(self):
        pass

    def _setup_hardware_interfaces(self):
        pass

    def _setup_joint_states(self):
        pass

    @classmethod
    def build_piecewise_constant_curvature(
        cls,
        world: World,
        num_sections: int = 3,
        segments_per_section: int = 6,
        total_length: float = 0.9,
        radius: float = 0.02,
    ) -> Tuple[SoftTrunk, List[DegreeOfFreedom], List[DegreeOfFreedom]]:
        """
        Builds a uniform soft robot using the Piecewise Constant Curvature (PCC) model.

        :param world: The world from which to create the robot view.
        :param num_sections: The number of independently controlled bending sections.
        :param segments_per_section: The number of visual segments per section (defines the smoothness).
        :param total_length: The total physical length of the robot in meters.
        :param radius: The uniform radius of the robot's cross-section.

        :return: A tuple containing:
                - The initialized SoftTrunk robot instance.
                - A list of kappa Degrees of Freedom (curvature) for each section.
                - A list of phi Degrees of Freedom (bending plane) for each section.
        """
        section_lengths = [total_length / num_sections] * num_sections
        section_radii = [radius] * num_sections
        section_resolutions = [segments_per_section] * num_sections

        return cls.build_custom_piecewise_constant_curvature(
            world, section_lengths, section_radii, section_resolutions
        )

    @classmethod
    def build_custom_piecewise_constant_curvature(
        cls,
        world: World,
        section_lengths: List[float],
        section_radii: List[float],
        section_resolutions: List[int],
    ) -> Tuple[SoftTrunk, List[DegreeOfFreedom], List[DegreeOfFreedom]]:
        """
        Builds a heterogeneous PCC robot with unique dimensions for every section.

        :param world: The world from which to create the robot view.
        :param section_lengths: A list containing the length of each section in meters.
        :param section_radii: A list containing the radius of each section.
        :param section_resolutions: A list containing the number of visual segments for each section.

        :return: A tuple containing:
                - The initialized SoftTrunk robot instance.
                - A list of kappa Degrees of Freedom (curvature) for each section.
                - A list of phi Degrees of Freedom (bending plane) for each section.
        """
        num_sections = len(section_lengths)
        prefix = "pcc"

        with world.modify_world():
            root_body = Body(name=PrefixedName(name="base", prefix=prefix))
            world.add_body(root_body)
            robot = cls(
                name=PrefixedName(name="robot", prefix=prefix),
                root=root_body,
                _world=world,
            )

            prev_body = root_body
            kappas, phis = [], []

            for s in range(num_sections):
                # Create unique DOFs for this section
                kappa = DegreeOfFreedom(
                    name=PrefixedName(name=f"kappa_{s}", prefix=prefix)
                )
                phi = DegreeOfFreedom(name=PrefixedName(name=f"phi_{s}", prefix=prefix))
                world.add_degree_of_freedom(kappa)
                world.add_degree_of_freedom(phi)
                kappas.append(kappa)
                phis.append(phi)

                # Section Dimensions
                current_section_length = section_lengths[s]
                current_radius = section_radii[s]
                resolution = section_resolutions[s]
                segment_length = current_section_length / resolution

                # Build the segments for this section
                for i in range(resolution):
                    color_val = (s + 1) / num_sections
                    visual = Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=-segment_length / 2
                        ),
                        width=current_radius * 2,
                        height=segment_length,
                        color=Color(0.0, color_val, 1.0 - color_val, 1.0),
                    )

                    curr_body = Body(
                        name=PrefixedName(name=f"sec{s}_seg{i}", prefix=prefix),
                        visual=ShapeCollection([visual]),
                    )
                    world.add_body(curr_body)

                    # Connect using UUIDs
                    connection = PiecewiseConstantCurvatureConnection(
                        parent=prev_body,
                        child=curr_body,
                        kappa_dof_id=kappa.id,
                        phi_dof_id=phi.id,
                        segment_length=segment_length,
                    )
                    world.add_connection(connection)
                    prev_body = curr_body

            tip_manipulator = Manipulator(
                name=PrefixedName(name="effector", prefix=prefix),
                root=prev_body,
                tool_frame=prev_body.name,
                front_facing_axis=Vector3(z=1.0),
                front_facing_orientation=Quaternion(w=1.0),
                _world=world,
            )
            world.add_semantic_annotation(tip_manipulator)

            arm_chain = Arm(
                name=PrefixedName(name="arm", prefix=prefix),
                root=root_body,
                tip=prev_body,
                _world=world,
                manipulator=tip_manipulator,
            )
            robot.add_kinematic_chain(arm_chain)
            world.add_semantic_annotation(robot)
        return robot, kappas, phis

    @classmethod
    def build_cosserat(
        cls,
        world: World,
        num_sections: int = 3,
        segments_per_section: int = 10,
        total_length: float = 0.9,
        radius: float = 0.02,
    ) -> Tuple[
        SoftTrunk,
        List[DegreeOfFreedom],
        List[DegreeOfFreedom],
        List[DegreeOfFreedom],
        List[DegreeOfFreedom],
    ]:
        """
        Builds a uniform soft robot based on Cosserat Rod Theory.

        :param world: The world from which to create the robot view.
        :param num_sections: Number of control sections.
        :param segments_per_section: Integration steps per section (defines accuracy and smoothness).
        :param total_length: Total length in meters.
        :param radius: Uniform radius of the rod.

        :return: A tuple containing:
                - The SoftTrunk instance.
                - List of Degrees of Freedom for bending around the X-axis (bending_x).
                - List of Degrees of Freedom for bending around the Y-axis (bending_y).
                - List of Degrees of Freedom for twisting (torsion).
                - List of Degrees of Freedom for stretching (extension).
        """
        section_lengths = [total_length / num_sections] * num_sections
        section_radii = [radius] * num_sections
        section_resolutions = [segments_per_section] * num_sections

        return cls.build_custom_cosserat(
            world, section_lengths, section_radii, section_resolutions
        )

    @classmethod
    def build_custom_cosserat(
        cls,
        world: World,
        section_lengths: List[float],
        section_radii: List[float],
        section_resolutions: List[int],
    ) -> Tuple[
        SoftTrunk,
        List[DegreeOfFreedom],
        List[DegreeOfFreedom],
        List[DegreeOfFreedom],
        List[DegreeOfFreedom],
    ]:
        """
        Builds a heterogeneous Cosserat Rod robot supporting complex torsion and extension.

        :param world: The world from which to create the robot view.
        :param section_lengths: List of lengths for each section.
        :param section_radii: List of radii for each section.
        :param section_resolutions: List of integration resolutions for each section.

        :return: A tuple containing:
                - The SoftTrunk instance.
                - List of Degrees of Freedom for bending around the X-axis (bending_x).
                - List of Degrees of Freedom for bending around the Y-axis (bending_y).
                - List of Degrees of Freedom for twisting (torsion).
                - List of Degrees of Freedom for stretching (extension).
        """
        num_sections = len(section_lengths)
        prefix = "cosserat"

        with world.modify_world():
            root_body = Body(name=PrefixedName(name="base", prefix=prefix))
            world.add_body(root_body)
            robot = cls(
                name=PrefixedName(name="robot", prefix=prefix),
                root=root_body,
                _world=world,
            )

            prev_body = root_body
            all_bx, all_by, all_tor, all_ext = [], [], [], []

            for s in range(num_sections):
                # Create unique DOFs (Strains)
                bx = DegreeOfFreedom(
                    name=PrefixedName(name=f"bending_x_{s}", prefix=prefix)
                )
                by = DegreeOfFreedom(
                    name=PrefixedName(name=f"bending_y_{s}", prefix=prefix)
                )
                tor = DegreeOfFreedom(
                    name=PrefixedName(name=f"torsion_{s}", prefix=prefix)
                )
                ext = DegreeOfFreedom(
                    name=PrefixedName(name=f"extension_{s}", prefix=prefix)
                )

                for dof in [bx, by, tor, ext]:
                    world.add_degree_of_freedom(dof)

                # Initialize extension to rest length
                world.state[ext.id].position = 1.0

                all_bx.append(bx)
                all_by.append(by)
                all_tor.append(tor)
                all_ext.append(ext)

                # Section Dimensions
                current_section_length = section_lengths[s]
                current_radius = section_radii[s]
                resolution = section_resolutions[s]
                segment_length = current_section_length / resolution

                # Build Segments for this section
                for i in range(resolution):
                    color_val = (s + 1) / num_sections
                    visual = Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=-segment_length / 2
                        ),
                        width=current_radius * 2,
                        height=segment_length,
                        color=Color(color_val, 0.2, 1.0 - color_val, 1.0),
                    )

                    curr_body = Body(
                        name=PrefixedName(name=f"sec{s}_seg{i}", prefix=prefix),
                        visual=ShapeCollection([visual]),
                    )
                    world.add_body(curr_body)

                    # Connect using UUIDs
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

            # Semantic Registration
            tip_manipulator = Manipulator(
                name=PrefixedName(name="effector", prefix=prefix),
                root=prev_body,
                tool_frame=prev_body.name,
                front_facing_axis=Vector3(z=1.0),
                front_facing_orientation=Quaternion(w=1.0),
                _world=world,
            )
            world.add_semantic_annotation(tip_manipulator)

            arm_chain = Arm(
                name=PrefixedName(name="arm", prefix=prefix),
                root=root_body,
                tip=prev_body,
                _world=world,
                manipulator=tip_manipulator,
            )
            robot.add_kinematic_chain(arm_chain)
            world.add_semantic_annotation(robot)

        return robot, all_bx, all_by, all_tor, all_ext
