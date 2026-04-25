from __future__ import annotations
from dataclasses import dataclass
from typing import Self

from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Arm
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.soft_connections import SoftPCCConnection
from semantic_digital_twin.world_description.soft_connections import CosseratRodConnection
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Cylinder, Color
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix

@dataclass(eq=False)
class SoftTrunk(AbstractRobot):
    @classmethod
    def _init_empty_robot(cls, world: World) -> Self:
        return cls(name=PrefixedName(name="soft_trunk", prefix="robot"), 
                   root=Body(name=PrefixedName(name="base", prefix="soft_trunk")), 
                   _world=world)

    def _setup_semantic_annotations(self): pass
    def _setup_collision_rules(self): pass
    def _setup_velocity_limits(self): pass
    def _setup_hardware_interfaces(self): pass
    def _setup_joint_states(self): pass
    
    @classmethod
    def build_pcc(cls, world: World, 
                  num_sections: int = 3, 
                  segs_per_section: int = 6, 
                  total_length: float = 0.9, 
                  radius: float = 0.02) -> tuple:
        """
        Method to build a uniform PCC robot.
        """
        section_lengths = [total_length / num_sections] * num_sections
        section_radii = [radius] * num_sections
        section_resolutions = [segs_per_section] * num_sections
        
        return cls.build_custom_pcc(world, section_lengths, section_radii, section_resolutions)

    @classmethod
    def build_custom_pcc(cls, world: World, 
                        section_lengths: list[float], 
                        section_radii: list[float], 
                        section_resolutions: list[int]) -> tuple:
        """
        The core logic for building PCC robots (Heterogeneous or Homogeneous).
        """
        num_sections = len(section_lengths)
        prefix = "pcc"
        
        with world.modify_world():
            root_body = Body(name=PrefixedName(name="base", prefix=prefix))
            world.add_body(root_body)
            robot = cls(name=PrefixedName(name="robot", prefix=prefix), root=root_body, _world=world)
            
            prev_body = root_body
            kappas, phis = [], []

            for s in range(num_sections):
                # Create DOFs
                kappa = DegreeOfFreedom(name=PrefixedName(name=f"kappa_{s}", prefix=prefix))
                phi = DegreeOfFreedom(name=PrefixedName(name=f"phi_{s}", prefix=prefix))
                world.add_degree_of_freedom(kappa); world.add_degree_of_freedom(phi)
                kappas.append(kappa); phis.append(phi)

                # Section Dimensions
                L_total = section_lengths[s]
                radius = section_radii[s]
                res = section_resolutions[s]
                seg_len = L_total / res

                # Build Segments
                for i in range(res):
                    color_val = (s + 1) / num_sections
                    seg_visual = Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=-seg_len/2), 
                        width=radius * 2, height=seg_len, 
                        color=Color(0.0, color_val, 1.0 - color_val, 1.0)
                    )
                    
                    curr_body = Body(name=PrefixedName(name=f"sec{s}_seg{i}", prefix=prefix), 
                                    visual=ShapeCollection([seg_visual]))
                    world.add_body(curr_body)
                    
                    # Connection
                    conn = SoftPCCConnection(prev_body, curr_body, kappa, phi, length=seg_len)
                    world.add_connection(conn)
                    prev_body = curr_body

            # Semantic Registration
            arm_chain = Arm(name=PrefixedName(name="arm", prefix=prefix), 
                            root=root_body, tip=prev_body, _world=world)
            robot.add_kinematic_chain(arm_chain)
            world.add_semantic_annotation(robot)
            
        return robot, kappas, phis
    
    @classmethod
    def build_cosserat(cls, world: World, 
                       num_sections: int = 3, 
                       segs_per_section: int = 10, 
                       total_length: float = 0.9, 
                       radius: float = 0.02) -> tuple:
        """
        Method to build a uniform Cosserat Rod.
        """
        section_lengths = [total_length / num_sections] * num_sections
        section_radii = [radius] * num_sections
        section_resolutions = [segs_per_section] * num_sections
        
        return cls.build_custom_cosserat(world, section_lengths, section_radii, section_resolutions)

    @classmethod
    def build_custom_cosserat(cls, world: World, 
                             section_lengths: list[float], 
                             section_radii: list[float], 
                             section_resolutions: list[int]) -> tuple:
        """
        The core logic for building heterogeneous Cosserat Rods.
        """
        num_sections = len(section_lengths)
        prefix = "cosserat"
        
        with world.modify_world():
            root_body = Body(name=PrefixedName(name="base", prefix=prefix))
            world.add_body(root_body)
            robot = cls(name=PrefixedName(name="robot", prefix=prefix), root=root_body, _world=world)
            
            prev_body = root_body
            all_ux, all_uy, all_uz, all_vz = [], [], [], []

            for s in range(num_sections):
                # Create DOFs for this section
                ux = DegreeOfFreedom(name=PrefixedName(name=f"ux_{s}", prefix=prefix))
                uy = DegreeOfFreedom(name=PrefixedName(name=f"uy_{s}", prefix=prefix))
                uz = DegreeOfFreedom(name=PrefixedName(name=f"uz_{s}", prefix=prefix))
                vz = DegreeOfFreedom(name=PrefixedName(name=f"vz_{s}", prefix=prefix))
                
                world.add_degree_of_freedom(ux); world.add_degree_of_freedom(uy)
                world.add_degree_of_freedom(uz); world.add_degree_of_freedom(vz)
                
                # Initialize extension to 1.0 (rest length)
                world.state[vz.id].position = 1.0
                
                all_ux.append(ux); all_uy.append(uy)
                all_uz.append(uz); all_vz.append(vz)

                # Section Dimensions
                L_total = section_lengths[s]
                radius = section_radii[s]
                res = section_resolutions[s]
                seg_len = L_total / res

                # Build the noodle segments
                for i in range(res):
                    color_val = (s + 1) / num_sections
                    seg_visual = Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=-seg_len/2), 
                        width=radius * 2, height=seg_len, 
                        color=Color(color_val, 0.2, 1.0 - color_val, 1.0)
                    )
                    
                    curr_body = Body(name=PrefixedName(name=f"sec{s}_seg{i}", prefix=prefix), 
                                    visual=ShapeCollection([seg_visual]))
                    world.add_body(curr_body)
                    
                    # Connect using the integrated solver
                    conn = CosseratRodConnection(prev_body, curr_body, ux, uy, uz, vz, length=seg_len)
                    world.add_connection(conn)
                    prev_body = curr_body

            # Semantic Registration
            arm_chain = Arm(name=PrefixedName(name="arm", prefix=prefix), 
                            root=root_body, tip=prev_body, _world=world)
            robot.add_kinematic_chain(arm_chain)
            world.add_semantic_annotation(robot)

        return robot, all_ux, all_uy, all_uz, all_vz