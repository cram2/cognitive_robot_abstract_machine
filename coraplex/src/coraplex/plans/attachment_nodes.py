from dataclasses import dataclass, field

from coraplex.plans.executables import Executable, ModelChangeExecutable
from coraplex.plans.plan_node import PlanNode
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class ModelChangeNode(PlanNode):
    """
    Node that represents a change in the world model of the semantic digital twin.

    new_parent is the point to which the body should be attached to. If no parent is
    provided the world root is used. Intended as a convenient use for detect. This is
    just the representation the actual change lies in the executable in
    pycram.plan.executables
    """

    body: Body = field(kw_only=True)
    """
    Body that should be moved in the world model.
    """

    new_parent: Body = field(kw_only=True, default=None)
    """
    New parent to which the body should be attached to.
    """

    def __post_init__(self):
        self.new_parent = self.new_parent or self.body._world.root

    def notify(self):
        pass

    def parse(self) -> ModelChangeExecutable:
        return ModelChangeExecutable(
            context=self.plan.context, body=self.body, new_parent=self.new_parent
        )


@dataclass
class AttachNode(ModelChangeNode):
    """
    Model change that attaches a body to another body (e.g. an object to the gripper
    after grasping).

    Kept as a distinct type so it can be located by type in the plan.
    """


@dataclass
class DetachNode(ModelChangeNode):
    """
    Model change that detaches a body from its current parent and re-attaches it to the
    world root (e.g. after placing an object).

    Kept as a distinct type so it can be located by type in the plan.
    """
