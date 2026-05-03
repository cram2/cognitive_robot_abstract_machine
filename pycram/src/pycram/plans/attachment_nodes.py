from dataclasses import dataclass, field

from pycram.plans.plan_node import PlanNode
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class ModelChangeNode(PlanNode):

    body: Body = field(kw_only=True)

    new_parent: Body = field(kw_only=True)

    def _perform(self):
        # Attach the object to the end effector
        with self.plan.world.modify_world():
            self.plan.world.move_branch_with_fixed_connection(
                self.body, self.new_parent
            )
