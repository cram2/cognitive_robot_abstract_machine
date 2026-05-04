from dataclasses import dataclass, field

from pycram.plans.plan_node import PlanNode
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class AttachNode(PlanNode):

    body: Body = field(kw_only=True)

    new_parent: Body = field(kw_only=True)

    def _perform(self):
        # Attach the object to the end effector
        with self.plan.world.modify_world():
            self.plan.world.move_branch_with_fixed_connection(
                self.body, self.new_parent
            )


@dataclass
class DetachNode(PlanNode):
    body: Body = field(kw_only=True)

    new_parent: Body = field(kw_only=True)

    def _perform(self):
        # Detaches the object from the robot
        obj_transform = self.plan.world.compute_forward_kinematics(
            self.new_parent, self.body
        )
        with self.plan.world.modify_world():
            self.plan.world.remove_connection(self.body.parent_connection)
            connection = Connection6DoF.create_with_dofs(
                parent=self.new_parent, child=self.body, world=self.plan.world
            )
            self.plan.world.add_connection(connection)
            connection.origin = obj_transform
