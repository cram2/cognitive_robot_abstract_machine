from __future__ import annotations
from dataclasses import dataclass, field
from typing import Self, TYPE_CHECKING, List, Tuple

from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Arm, Manipulator
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
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
    """

    length: float
    radius: float
    resolution: int


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

    @classmethod
    def from_world(
        cls, world: World, name: str = "robot", prefix: str = "soft_trunk"
    ) -> Self:
        """
        Wraps an existing soft robot structure in the world into a SoftTrunk view.

        :param world: The world instance containing the soft robot entities.

        :param name: The semantic name to assign to the robot.

        :param prefix: The prefix used by the robot's bodies in the world.

        :return: A SoftTrunk robot view.
        """

        with world.modify_world():
            root_body = world.get_body_by_name(PrefixedName("base", prefix))

            soft_trunk = cls(
                name=PrefixedName(name, prefix),
                root=root_body,
                _world=world,
            )
            world.add_semantic_annotation(soft_trunk)

        return soft_trunk

    def _get_sorted_dofs_by_type(self, substring: str) -> List[DegreeOfFreedom]:
        """
        Finds DOFs by traversing the robot's own kinematic chains.
        """
        found_dofs = set()
        for chain in self.manipulator_chains:
            connections = self._world.compute_chain_of_connections(
                chain.root, chain.tip
            )
            for conn in connections:
                for dof in conn.active_dofs:
                    if substring in str(dof.name):
                        found_dofs.add(dof)

        return sorted(list(found_dofs), key=lambda x: str(x.name))

    @property
    def kappa_dofs(self) -> List[DegreeOfFreedom]:
        """Returns all curvature DOFs for a PCC robot."""
        return self._get_sorted_dofs_by_type("kappa")

    @property
    def phi_dofs(self) -> List[DegreeOfFreedom]:
        """Returns all bending plane DOFs for a PCC robot."""
        return self._get_sorted_dofs_by_type("phi")

    @property
    def bending_x_dofs(self) -> List[DegreeOfFreedom]:
        """Returns all bending-x (ux) DOFs for a Cosserat robot."""
        return self._get_sorted_dofs_by_type("bending_x")

    @property
    def bending_y_dofs(self) -> List[DegreeOfFreedom]:
        """Returns all bending-y (uy) DOFs for a Cosserat robot."""
        return self._get_sorted_dofs_by_type("bending_y")

    @property
    def torsion_dofs(self) -> List[DegreeOfFreedom]:
        """Returns all axial torsion (uz) DOFs for a Cosserat robot."""
        return self._get_sorted_dofs_by_type("torsion")

    @property
    def extension_dofs(self) -> List[DegreeOfFreedom]:
        """Returns all linear extension (vz) DOFs for a Cosserat robot."""
        return self._get_sorted_dofs_by_type("extension")

    @property
    def pcc_sections(self) -> List[Tuple[DegreeOfFreedom, DegreeOfFreedom]]:
        """Returns a list of (kappa_dof, phi_dof) tuples, one per section."""
        return list(zip(self.kappa_dofs, self.phi_dofs))

    @property
    def cosserat_sections(
        self,
    ) -> List[
        Tuple[DegreeOfFreedom, DegreeOfFreedom, DegreeOfFreedom, DegreeOfFreedom]
    ]:
        """Returns a list of (bx, by, torsion, extension) tuples, one per section."""
        return list(
            zip(
                self.bending_x_dofs,
                self.bending_y_dofs,
                self.torsion_dofs,
                self.extension_dofs,
            )
        )

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
        sections: List[SoftTrunkSection],
    ) -> SoftTrunk:
        """
        Builds a Piecewise Constant Curvature (PCC) robot from a list of sections.

        :param world: The world from which to create the robot view.

        :param sections: A list of section configurations defining the robot's morphology.

        :return: A SoftTrunk robot view.
        """

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
            limits = DegreeOfFreedomLimits(
                lower=DerivativeMap(position=-10.0, velocity=-10.0),
                upper=DerivativeMap(position=10.0, velocity=10.0),
            )

            for s, section in enumerate(sections):
                kappa = DegreeOfFreedom(
                    name=PrefixedName(f"kappa_{s}", prefix),
                    limits=limits,
                )
                phi = DegreeOfFreedom(
                    name=PrefixedName(f"phi_{s}", prefix),
                    limits=limits,
                )
                world.add_degree_of_freedom(kappa)
                world.add_degree_of_freedom(phi)
                kappas.append(kappa)
                phis.append(phi)

                segment_length = section.length / section.resolution
                for i in range(section.resolution):
                    color_val = (s + 1) / len(sections)
                    visual = Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=-segment_length / 2
                        ),
                        width=section.radius * 2,
                        height=segment_length,
                        color=Color(0.0, color_val, 1.0 - color_val, 1.0),
                    )
                    curr_body = Body(
                        name=PrefixedName(f"sec{s}_seg{i}", prefix),
                        visual=ShapeCollection([visual]),
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

            tip_manipulator = Manipulator(
                name=PrefixedName("effector", prefix),
                root=prev_body,
                tool_frame=prev_body.name,
                front_facing_axis=Vector3(z=1.0),
                front_facing_orientation=Quaternion(w=1.0),
                _world=world,
            )
            world.add_semantic_annotation(tip_manipulator)

            arm_chain = Arm(
                name=PrefixedName("arm", prefix),
                root=root_body,
                tip=prev_body,
                _world=world,
                manipulator=tip_manipulator,
            )
            robot.add_kinematic_chain(arm_chain)
            world.add_semantic_annotation(robot)

        return robot

    @classmethod
    def build_cosserat(
        cls,
        world: World,
        sections: List[SoftTrunkSection],
    ) -> SoftTrunk:
        """
        Builds a Cosserat Rod robot.

        :param world: The world from which to create the robot view.

        :param sections: A list of section configurations defining the morphology.

        :return: A SoftTrunk robot view
        """

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
                all_bx.append(bx)
                all_by.append(by)
                all_tor.append(tor)
                all_ext.append(ext)

                segment_length = section.length / section.resolution
                for i in range(section.resolution):
                    color_val = (s + 1) / len(sections)
                    visual = Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(
                            z=-segment_length / 2
                        ),
                        width=section.radius * 2,
                        height=segment_length,
                        color=Color(color_val, 0.2, 1.0 - color_val, 1.0),
                    )

                    curr_body = Body(
                        name=PrefixedName(f"sec{s}_seg{i}", prefix),
                        visual=ShapeCollection([visual]),
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

            # Semantic registration
            tip_manipulator = Manipulator(
                name=PrefixedName("effector", prefix),
                root=prev_body,
                tool_frame=prev_body.name,
                front_facing_axis=Vector3(z=1.0),
                front_facing_orientation=Quaternion(w=1.0),
                _world=world,
            )
            world.add_semantic_annotation(tip_manipulator)

            arm_chain = Arm(
                name=PrefixedName("arm", prefix),
                root=root_body,
                tip=prev_body,
                _world=world,
                manipulator=tip_manipulator,
            )
            robot.add_kinematic_chain(arm_chain)
            world.add_semantic_annotation(robot)

        return robot
