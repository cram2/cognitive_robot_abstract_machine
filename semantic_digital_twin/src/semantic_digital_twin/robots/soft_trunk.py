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
    def build_pcc(cls, world: World) -> tuple[SoftTrunk, DegreeOfFreedom, DegreeOfFreedom]:
        with world.modify_world():
            # Create and add bodies
            root_body = Body(name=PrefixedName(name="base", prefix="soft_trunk"))
            arm_visual = Cylinder(origin=HomogeneousTransformationMatrix.from_xyz_rpy(z=-0.25), 
                                 width=0.05, height=0.5, color=Color(0.0, 0.0, 1.0, 1.0))
            tip_body = Body(name=PrefixedName(name="tip", prefix="soft_trunk"), visual=ShapeCollection([arm_visual]))
            
            world.add_body(root_body)
            world.add_body(tip_body)
            
            # Create and add DOFs
            kappa = DegreeOfFreedom(name=PrefixedName(name="kappa", prefix="soft_trunk"))
            phi = DegreeOfFreedom(name=PrefixedName(name="phi", prefix="soft_trunk"))
            world.add_degree_of_freedom(kappa)
            world.add_degree_of_freedom(phi)
            
            # Create the soft connection
            conn = SoftPCCConnection(parent=root_body, child=tip_body, kappa_dof=kappa, phi_dof=phi, length=0.5)
            world.add_connection(conn)
            
            # Semantic Registration
            robot = cls(name=PrefixedName(name="soft_trunk", prefix="robot"), root=root_body, _world=world)
            arm_chain = Arm(name=PrefixedName(name="soft_arm", prefix="soft_trunk"), 
                            root=root_body, tip=tip_body, _world=world)
            robot.add_kinematic_chain(arm_chain)
            world.add_semantic_annotation(robot)
            
        return robot, kappa, phi