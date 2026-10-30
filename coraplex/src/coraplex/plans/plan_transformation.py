from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from dataclasses import dataclass, field

from typing_extensions import Any, Generic, TypeVar

from coraplex.datastructures.enums import InsertionPosition
from coraplex.exceptions import CannotMatchOnType
from cramph.composites import ControlStructureNode, Parallel, Sequence
from cramph.context import ContextExtension
from cramph.node import StatechartNode
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

logger = logging.getLogger(__name__)

MatchedType = TypeVar("MatchedType", bound=StatechartNode)


# %% transformations


@dataclass
class PlanTransformation(Generic[MatchedType], SubClassSafeGeneric, ABC):
    """
    Rewrites the part of a plan around a node, before the statechart running the plan is
    compiled.

    The bound type says which nodes it rewrites: the nodes of that type. Every node of
    a plan is offered to a transformation once, after it has been expanded, see
    :class:`PlanRewriting`.
    """

    @property
    def matched_type(self) -> type[MatchedType]:
        """
        :return: The type this selects its nodes by.
        """
        return type(self).get_type_of_generic_parameter(MatchedType)

    def matches_node(self, plan_node: StatechartNode) -> bool:
        """
        :param plan_node: The node that was just expanded
        :return: Whether the given node is one this rewrites.
        :raises CannotMatchOnType: If the bound type is not a statechart node type
        """
        if not issubclass(self.matched_type, StatechartNode):
            raise CannotMatchOnType(type(self), self.matched_type)
        return isinstance(plan_node, self.matched_type)

    @abstractmethod
    def is_applicable(self, plan_node: MatchedType) -> bool:
        """
        Reports whether the case the node describes needs this transformation.

        It is asked only about nodes :meth:`matches_node` selected, so the node can be
        read as the type this is bound to.

        :param plan_node: A node this matches
        :return: Whether the transformation is needed here.
        """

    @abstractmethod
    def apply(self, plan_node: MatchedType) -> None:
        """
        Rewrites the plan around the given node.

        :param plan_node: The node this transformation is applied to
        """


# %% configuring a node before it expands


@dataclass
class ConfigurationTransformation(
    PlanTransformation[MatchedType], Generic[MatchedType], SubClassSafeGeneric, ABC
):
    """
    Rewrites a plan by changing the settings of a node before it expands, so that what
    it expands into already reflects them.

    Offered every node right before it expands, see :meth:`PlanRewriting.configure`, and
    never after.
    """


# %% inserting


@dataclass
class InsertionTransformation(
    PlanTransformation[MatchedType], Generic[MatchedType], SubClassSafeGeneric, ABC
):
    """
    Rewrites a plan by inserting freshly built nodes next to an anchor node.

    The nodes are built anew on every application, since a node belongs to the one
    statechart it was inserted into.
    """

    @abstractmethod
    def anchor(self, plan_node: MatchedType) -> StatechartNode:
        """
        :param plan_node: The node this transformation is applied to
        :return: The node the new nodes are inserted next to.
        """

    @abstractmethod
    def nodes_to_insert(self, plan_node: MatchedType) -> list[StatechartNode]:
        """
        :param plan_node: The node this transformation is applied to
        :return: The nodes to insert, in the order they take.
        """

    def apply(self, plan_node: MatchedType) -> None:
        anchor = self.anchor(plan_node)
        for node in self.nodes_to_insert(plan_node):
            anchor = self._insert(anchor, node)

    @abstractmethod
    def _insert(self, anchor: StatechartNode, node: StatechartNode) -> StatechartNode:
        """
        Inserts a node next to the anchor node.

        :param anchor: The node the given node is placed next to.
        :param node: The node to insert.
        :return: The node the next node is placed next to, which keeps the inserted
            nodes in their order.
        """


@dataclass
class SequentialInsertion(
    InsertionTransformation[MatchedType], Generic[MatchedType], SubClassSafeGeneric, ABC
):
    """
    Inserts nodes that run right before or right after the anchor node: as steps of the
    sequence that runs the anchor, or, when no sequence runs it, of a new sequence that
    takes the place of the step holding the anchor.
    """

    @property
    @abstractmethod
    def position(self) -> InsertionPosition:
        """
        :return: Whether the inserted nodes run before or after the anchor node.
        """

    def _insert(self, anchor: StatechartNode, node: StatechartNode) -> StatechartNode:
        step = ControlStructureNode.step_holding(anchor)
        if not isinstance(step.parent_node, Sequence):
            step.statechart.replace(step, Sequence(nodes=[step]))
        match self.position:
            case InsertionPosition.BEFORE:
                step.parent_node.insert_before(step, node)
                return anchor
            case InsertionPosition.AFTER:
                step.parent_node.insert_after(step, node)
                return node


@dataclass
class ParallelInsertion(
    InsertionTransformation[MatchedType], Generic[MatchedType], SubClassSafeGeneric, ABC
):
    """
    Inserts nodes that run alongside the anchor node: in the parallel that runs the
    anchor, or, when no parallel runs it, in a new parallel that takes the place of the
    step holding the anchor.
    """

    def _insert(self, anchor: StatechartNode, node: StatechartNode) -> StatechartNode:
        step = ControlStructureNode.step_holding(anchor)
        if not isinstance(step.parent_node, Parallel):
            step.statechart.replace(step, Parallel(nodes=[step]))
        step.parent_node.add_node(node)
        return anchor


# %% rewriting a plan


@dataclass
class PlanRewriting(ContextExtension):
    """
    Offers the nodes of a plan to the transformations that may rewrite it.

    A node is offered to the :class:`ConfigurationTransformation` instances right before
    it expands, see :meth:`configure`, and to the other transformations once it and
    everything below it has been expanded, in the order the plan runs its nodes; the
    nodes a transformation inserts are offered in turn. Carried in the context of the
    statechart running the plan, so that a part of the plan that joins it later can be
    rewritten too.
    """

    transformations: list[PlanTransformation] = field(default_factory=list)
    """
    The transformations the nodes are offered to.
    """

    _offered: set[StatechartNode] = field(default_factory=set, init=False, repr=False)
    """
    The nodes offered already, so that rewriting a part of the plan again offers only
    what is new.
    """

    def __deepcopy__(self, memo: dict[int, Any]) -> PlanRewriting:
        """
        :return: A rewriting by the same transformations that has offered nothing yet,
            since the nodes offered so far belong to the statechart this one rewrites.
        """
        return PlanRewriting(transformations=list(self.transformations))

    def copy_with_preceding_transformations(
        self, transformations: list[PlanTransformation]
    ) -> PlanRewriting:
        """
        :param transformations: The transformations to offer every node to first.
        :return: A new rewriting by `transformations` and then this one's, which has
            offered nothing yet.
        """
        return PlanRewriting(transformations=[*transformations, *self.transformations])

    def configure(self, node: StatechartNode) -> None:
        """
        Applies every :class:`ConfigurationTransformation` that matches and applies to
        `node`, which has not expanded yet.

        :param node: The node about to expand.
        """
        for transformation in self.transformations:
            if (
                isinstance(transformation, ConfigurationTransformation)
                and transformation.matches_node(node)
                and transformation.is_applicable(node)
            ):
                transformation.apply(node)

    def rewrite(self, root: StatechartNode) -> StatechartNode:
        """
        Applies every transformation to every node below and including `root` that it
        matches and applies to, and to every node a transformation puts around `root`.

        :param root: The part of the plan to rewrite.
        :return: The node that runs in the place of `root` once rewritten: `root`
            itself, or the node a transformation put in its place around it.
        """
        if not self.transformations:
            return root
        former_parent = root.parent_node
        rewritten = root
        while (node := self._next_node_to_offer(rewritten)) is not None:
            self._offered.add(node)
            self._apply_to(node)
            while rewritten.parent_node is not former_parent:
                rewritten = rewritten.parent_node
        return rewritten

    def _next_node_to_offer(self, root: StatechartNode) -> StatechartNode | None:
        """
        :param root: The part of the plan being rewritten.
        :return: The first node of `root` in the order the plan runs its nodes that has
            not been offered yet, or None if there is none.
        """
        for node in [root, *root.descendants]:
            if node not in self._offered:
                return node
        return None

    def _apply_to(self, node: StatechartNode) -> None:
        """
        Rewrites the plan with every transformation that applies to `node`.

        Each of them rewrites what the ones before it left, so more than one of them on
        the same node is reported.

        :param node: The node offered to the transformations.
        """
        transformations = [
            transformation
            for transformation in self.transformations
            if not isinstance(transformation, ConfigurationTransformation)
            and transformation.matches_node(node)
            and transformation.is_applicable(node)
        ]
        if len(transformations) > 1:
            logger.warning(
                f"{len(transformations)} plan transformations are applied to {node}: "
                f"{transformations}"
            )
        for transformation in transformations:
            transformation.apply(node)
