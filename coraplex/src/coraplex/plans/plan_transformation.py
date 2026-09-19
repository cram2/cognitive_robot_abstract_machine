from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass

from typing_extensions import (
    Generic,
    List,
    Type,
    TypeVar,
)

from coraplex.datastructures.enums import InsertionPosition
from coraplex.exceptions import CannotMatchOnType
from coraplex.plans.designator import Designator
from coraplex.plans.factories import make_node
from coraplex.plans.plan_node import ActionLike, DesignatorNode, PlanNode
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

MatchedType = TypeVar("MatchedType", bound=PlanNode | Designator)


# %% transformations


@dataclass
class PlanTransformation(Generic[MatchedType], SubClassSafeGeneric, ABC):
    """
    Rewrites the part of a plan that a node expanded into.

    The bound type says which nodes it rewrites: a node type selects the nodes of that
    type, a designator type the nodes carrying such a designator. A transformation is
    applied to every node it matches, right after that node has been expanded and before
    the nodes below it are expanded in turn.
    """

    @property
    def matched_type(self) -> Type[MatchedType]:
        """
        :return: The type this selects its nodes by.
        """
        return type(self).get_type_of_generic_parameter(MatchedType)

    def matches_node(self, plan_node: PlanNode) -> bool:
        """
        :param plan_node: The node that was just expanded
        :return: Whether the given node is one this rewrites.
        :raises CannotMatchOnType: If the bound type is neither a node nor a designator
        """
        if issubclass(self.matched_type, PlanNode):
            return isinstance(plan_node, self.matched_type)
        if issubclass(self.matched_type, Designator):
            return isinstance(plan_node, DesignatorNode) and isinstance(
                plan_node.designator, self.matched_type
            )
        raise CannotMatchOnType(type(self), self.matched_type)

    @abstractmethod
    def is_applicable(self, plan_node: PlanNode) -> bool:
        """
        Reports whether the case the node describes needs this transformation.

        It is asked only about nodes :meth:`matches_node` selected, so the node can be
        read as the type this is bound to.

        :param plan_node: A node this matches
        :return: Whether the transformation is needed here.
        """

    @abstractmethod
    def apply(self, plan_node: PlanNode) -> None:
        """
        Rewrites the plan around the given node.

        :param plan_node: The node this transformation is applied to
        """


# %% inserting


@dataclass
class InsertionTransformation(
    PlanTransformation[MatchedType], Generic[MatchedType], SubClassSafeGeneric, ABC
):
    """
    Rewrites a plan by inserting freshly built nodes next to an anchor node.

    The nodes are built anew on every application, since a node belongs to the one plan
    it was inserted into.
    """

    @property
    @abstractmethod
    def position(self) -> InsertionPosition:
        """
        :return: Where the inserted nodes are placed relative to the anchor node.
        """

    @abstractmethod
    def anchor(self, plan_node: PlanNode) -> PlanNode:
        """
        :param plan_node: The node this transformation is applied to
        :return: The node the new nodes are inserted next to.
        """

    @abstractmethod
    def nodes_to_insert(self, plan_node: PlanNode) -> List[ActionLike]:
        """
        :param plan_node: The node this transformation is applied to
        :return: The actions, motions or nodes to insert, in the order they take.
        """

    def apply(self, plan_node: PlanNode) -> None:
        anchor = self.anchor(plan_node)
        for action_like in self.nodes_to_insert(plan_node):
            node = make_node(action_like)
            self.position.insert(plan_node.plan, anchor, node)
            if self.position is InsertionPosition.AFTER:
                # each further node goes behind the one before it, keeping their order
                anchor = node
