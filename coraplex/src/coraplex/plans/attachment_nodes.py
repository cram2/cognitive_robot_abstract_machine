from dataclasses import dataclass, field

from coraplex.plans.executables import (
    MoveBranchExecutable,
)
from coraplex.plans.plan_node import ExecutionBoundaryNode
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)


@dataclass
class ReAttachNode(ExecutionBoundaryNode):
    """
    Node that represents a change in the world model of the semantic digital twin.

    new_parent is the point to which the body should be attached to. If no parent is
    provided the world root is used. Intended as a convenient use for detect. This is
    just the representation the actual change lies in the executable in
    pycram.plan.executables
    """

    body: KinematicStructureEntity = field(kw_only=True)
    """
    Body that should be moved in the world model.
    """

    new_parent: KinematicStructureEntity = field(kw_only=True, default=None)
    """
    New parent to which the body should be attached to.
    """

    def __post_init__(self):
        self.new_parent = self.new_parent or self.body._world.root

    def notify(self):
        pass

    def parse(self) -> MoveBranchExecutable:
        return MoveBranchExecutable(
            context=self.context, body=self.body, new_parent=self.new_parent
        )
