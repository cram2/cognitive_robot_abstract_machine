from __future__ import annotations
from dataclasses import dataclass
from typing import Self, TYPE_CHECKING

from semantic_digital_twin.robots.abstract_robot import AbstractRobot, Arm
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.soft_connections import SoftPCCConnection
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
    def build_pcc(cls, world: World, num_segments: int = 10) -> tuple[SoftTrunk, DegreeOfFreedom, DegreeOfFreedom]:
        total_length = 0.5
        seg_length = total_length / num_segments
        
        with world.modify_world():
            # Create the global DOFs (shared by all segments)
            kappa = DegreeOfFreedom(name=PrefixedName(name="kappa", prefix="soft_trunk"))
            phi = DegreeOfFreedom(name=PrefixedName(name="phi", prefix="soft_trunk"))
            world.add_degree_of_freedom(kappa)
            world.add_degree_of_freedom(phi)

            # Create the Root
            root_body = Body(name=PrefixedName(name="base", prefix="soft_trunk"))
            world.add_body(root_body)
            
            robot = cls(name=PrefixedName(name="soft_trunk", prefix="robot"), root=root_body, _world=world)
            
            prev_body = root_body
            for i in range(num_segments):
                # Create a small cylinder for each segment
                # center the cylinder so it looks like a continuous rod
                seg_visual = Cylinder(
                    origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=-seg_length/2), 
                    width=0.05, 
                    height=seg_length, 
                    color=Color(0.0, 0.5, 1.0, 1.0)
                )
                
                curr_body = Body(
                    name=PrefixedName(name=f"seg_{i}", prefix="soft_trunk"), 
                    visual=ShapeCollection([seg_visual])
                )
                world.add_body(curr_body)
                
                # Every segment uses the same kappa and phi
                conn = SoftPCCConnection(
                    parent=prev_body, 
                    child=curr_body, 
                    kappa_dof=kappa, 
                    phi_dof=phi, 
                    length=seg_length
                )
                world.add_connection(conn)
                prev_body = curr_body

            # Tip is the last body created
            arm_chain = Arm(name=PrefixedName(name="soft_arm", prefix="soft_trunk"), 
                            root=root_body, tip=prev_body, _world=world)
            robot.add_kinematic_chain(arm_chain)
            world.add_semantic_annotation(robot)
            
        return robot, kappa, phi