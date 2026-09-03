from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass

from typing_extensions import Callable, ClassVar, Dict, Generic, List, Type, TypeVar

from coraplex.datastructures.enums import InsertionPosition
from coraplex.plans.factories import make_node
from coraplex.plans.plan import Plan
from coraplex.plans.plan_node import ActionLike, ActionNode, PlanNode
from coraplex.robot_plans.actions.base import ActionDescription
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import get_generic_type_parameters

NodeType = TypeVar("NodeType", bound=PlanNode)
ActionType = TypeVar("ActionType", bound=ActionDescription)


# %% matching


@dataclass
class PlanTransformation(Generic[NodeType], SubClassSafeGeneric, ABC):
    """
    Rewrites the part of a plan that a node expanded into.

    A transformation is applied to every node it applies to, right after that node has
    been expanded and before the nodes below it are expanded in turn.
    """

    @property
    def node_type(self) -> Type[PlanNode]:
        """
        :return: The type of node this transformation rewrites.
        """
        return get_generic_type_parameters(
            type(self), PlanTransformation, include_root_generic_base=False
        )[0]

    def applies_to(self, plan_node: PlanNode) -> bool:
        """
        :param plan_node: The node that was just expanded
        :return: Whether this transformation rewrites the given node.
        """
        return isinstance(plan_node, self.node_type)

    @abstractmethod
    def apply(self, plan_node: NodeType) -> None:
        """
        Rewrites the plan around the given node.

        :param plan_node: The node this transformation applies to
        """


@dataclass
class ActionTransformation(
    PlanTransformation[ActionNode], Generic[ActionType], SubClassSafeGeneric, ABC
):
    """
    Rewrites the plan of actions of the bound action type.
    """

    @property
    def node_type(self) -> Type[PlanNode]:
        # Concrete transformations of this family bind the action type, so the node
        # type is read from what the family itself binds.
        return get_generic_type_parameters(
            ActionTransformation,
            PlanTransformation,
            include_root_generic_base=False,
        )[0]

    @property
    def action_type(self) -> Type[ActionDescription]:
        """
        :return: The type of action this transformation rewrites the plan of.
        """
        return get_generic_type_parameters(
            type(self), ActionTransformation, include_root_generic_base=False
        )[0]

    def applies_to(self, plan_node: PlanNode) -> bool:
        return super().applies_to(plan_node) and isinstance(
            plan_node.designator, self.action_type
        )


# %% rewriting


@dataclass
class InsertionTransformation(ABC):
    """
    Rewrites a plan by inserting freshly built nodes next to an anchor node.

    The nodes are built anew on every application, since a node belongs to the one plan
    it was inserted into.
    """

    position: InsertionPosition = InsertionPosition.BEFORE
    """
    Where the inserted nodes are placed relative to the anchor node.
    """

    _insertion_methods: ClassVar[Dict[InsertionPosition, Callable[..., None]]] = {
        InsertionPosition.BEFORE: Plan.insert_before,
        InsertionPosition.AFTER: Plan.insert_after,
        InsertionPosition.BELOW: Plan.insert_below,
    }
    """
    The way each position inserts a node into the plan.
    """

    @abstractmethod
    def anchor(self, plan_node: PlanNode) -> PlanNode:
        """
        :param plan_node: The node this transformation applies to
        :return: The node the new nodes are inserted next to.
        """

    @abstractmethod
    def nodes_to_insert(self, plan_node: PlanNode) -> List[ActionLike]:
        """
        :param plan_node: The node this transformation applies to
        :return: The actions, motions or nodes to insert, in the order they take.
        """

    def apply(self, plan_node: PlanNode) -> None:
        anchor = self.anchor(plan_node)
        insert = self._insertion_methods[self.position]
        for action_like in self.nodes_to_insert(plan_node):
            node = make_node(action_like)
            insert(plan_node.plan, anchor, node)
            if self.position is InsertionPosition.AFTER:
                # each further node goes behind the one before it, keeping their order
                anchor = node
