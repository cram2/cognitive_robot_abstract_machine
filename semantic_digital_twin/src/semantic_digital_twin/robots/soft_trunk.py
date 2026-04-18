from __future__ import annotations
from dataclasses import dataclass
from typing import Self, TYPE_CHECKING

from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Arm
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.soft_connections import SoftPCCConnection
from semantic_digital_twin.world_description.soft_connections import CosseratRodConnection
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Cylinder, Color
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix

if TYPE_CHECKING:
    from semantic_digital_twin.world import World

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
    def build_pcc(cls, world: World, num_sections: int = 3, segs_per_section: int = 6, total_robot_length: float = 0.9) -> SoftTrunk:
        section_length = total_robot_length / num_sections
        seg_length = section_length / segs_per_section
        
        with world.modify_world():
            root_body = Body(name=PrefixedName(name="base", prefix="snake"))
            world.add_body(root_body)
            robot = cls(name=PrefixedName(name="soft_snake", prefix="robot"), root=root_body, _world=world)
            
            prev_body = root_body

            kappas = []
            phis = []

            for s in range(num_sections):
                # unique dofs for section
                kappa = DegreeOfFreedom(name=PrefixedName(name=f"kappa_{s}", prefix="snake"))
                phi = DegreeOfFreedom(name=PrefixedName(name=f"phi_{s}", prefix="snake"))
                world.add_degree_of_freedom(kappa)
                world.add_degree_of_freedom(phi)
                kappas.append(kappa)
                phis.append(phi)

                # Build the noodle for this section
                for i in range(segs_per_section):
                    # Color-code sections
                    color_val = (s + 1) / num_sections
                    seg_visual = Cylinder(
                        origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=-seg_length/2), 
                        width=0.04, height=seg_length, 
                        color=Color(0.0, color_val, 1.0 - color_val, 1.0)
                    )
                    
                    curr_body = Body(name=PrefixedName(name=f"sec{s}_seg{i}", prefix="snake"), 
                                    visual=ShapeCollection([seg_visual]))
                    world.add_body(curr_body)
                    
                    # Connect using the DOFs for this specific section
                    conn = SoftPCCConnection(parent=prev_body, child=curr_body, 
                                            kappa_dof=kappa, phi_dof=phi, length=seg_length)
                    world.add_connection(conn)
                    prev_body = curr_body

            # Register the whole thing as one arm
            arm_chain = Arm(name=PrefixedName(name="snake_arm", prefix="snake"), 
                            root=root_body, tip=prev_body, _world=world)
            robot.add_kinematic_chain(arm_chain)
            world.add_semantic_annotation(robot)
            
        return robot, kappas, phis
    
    @classmethod
    def build_cosserat_trunk(cls, world: World) -> tuple:
        with world.modify_world():
            root_body = Body(name=PrefixedName(name="base", prefix="cosserat"))
            tip_body = Body(name=PrefixedName(name="tip", prefix="cosserat"), 
                            visual=ShapeCollection([Cylinder(height=0.5, width=0.04)]))
            
            world.add_body(root_body)
            world.add_body(tip_body)
            
            # 3 DOFs for bending and twisting
            ux = DegreeOfFreedom(name=PrefixedName(name="ux", prefix="cosserat"))
            uy = DegreeOfFreedom(name=PrefixedName(name="uy", prefix="cosserat"))
            uz = DegreeOfFreedom(name=PrefixedName(name="uz", prefix="cosserat"))
            world.add_degree_of_freedom(ux)
            world.add_degree_of_freedom(uy)
            world.add_degree_of_freedom(uz)
            
            conn = CosseratRodConnection(root_body, tip_body, ux, uy, uz, length=0.5)
            world.add_connection(conn)
            
            robot = cls(name=PrefixedName(name="cosserat_bot", prefix="robot"), root=root_body, _world=world)
            return robot, ux, uy, uz