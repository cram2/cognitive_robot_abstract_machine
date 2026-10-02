from dataclasses import dataclass, field

from typing_extensions import Optional

from coraplex.plans.executables import (
    MoveBranchExecutable,
)
from coraplex.plans.plan_node import ExecutionBoundaryNode
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
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

    parent_T_connection_expression: Optional[HomogeneousTransformationMatrix] = field(
        default=None, kw_only=True
    )
    """
    Explicit transform from ``new_parent`` to the body.

    When ``None`` (default), the transform is computed to preserve the body's current
    global pose. When provided, it is used directly as the transform from
    ``new_parent`` to the body.
    """

    def __post_init__(self):
        self.new_parent = self.new_parent or self.body._world.root

    def notify(self):
        pass

    def parse(self) -> MoveBranchExecutable:
        return MoveBranchExecutable(
            context=self.context,
            body=self.body,
            new_parent=self.new_parent,
            parent_T_connection_expression=self.parent_T_connection_expression,
            execution_scope=self.execution_scope,
        )
