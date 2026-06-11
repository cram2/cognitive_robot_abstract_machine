from dataclasses import dataclass, field

from pycram.plans.executables import Executable, ModelChangeExecutable
from pycram.plans.plan_node import PlanNode
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class ModelChangeNode(PlanNode):
    body: Body = field(kw_only=True)

    new_parent: Body = field(kw_only=True)

    def notify(self):
        pass

    def parse(self) -> ModelChangeExecutable:
        return ModelChangeExecutable(
            context=self.plan.context, body=self.body, new_parent=self.new_parent
        )


@dataclass
class AttachNode(ModelChangeNode): ...


@dataclass
class DetachNode(ModelChangeNode): ...
