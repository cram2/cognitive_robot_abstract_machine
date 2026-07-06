from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import field, dataclass

import rustworkx as rx
import rustworkx.visualization
from typing_extensions import (
    Optional,
    Any,
    List,
    Iterable,
    TYPE_CHECKING,
    Tuple,
    TypeVar,
    Type,
)

from coraplex.plans.plan_entity import PlanEntity
from coraplex.plans.plan_node import (
    PlanNode,
    ActionNode,
    DesignatorNode,
)
from coraplex.visualization import plot_rustworkx_interactive, create_ordered_graph
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World

if TYPE_CHECKING:
    from coraplex.plans.plan_callbacks import PlanCallback
    from coraplex.datastructures.dataclasses import Context
    from coraplex.plans.designator import Designator


logger = logging.getLogger(__name__)


T = TypeVar("T")


@dataclass
class Plan:
    """
    Represents a plan structure, typically a tree, which can be changed at any point in time. Performing the plan will
    traverse the plan structure in depth first order and perform each PlanNode
    """

    context: Optional[Context] = None
    """
    The context where the plan can extract information from.
    """

    initial_world: Optional[World] = field(repr=False, kw_only=True, default=None)
    """
    A deepcopy of the world before the first perform was called.
    """

    node_callbacks: List[PlanCallback] = field(default_factory=list)
    """
    A list of callbacks that are called when a node is started or ended.
    """

    plan_graph: rx.PyDiGraph[PlanNode] = field(
        default_factory=lambda: rx.PyDiGraph(multigraph=False), init=False, repr=False
    )
    """
    A directed graph representation of the plan structure.
    """

    def __post_init__(self):
        if self.context is not None:
            self.add_plan_entity(self.context)

    def validate(self):
        """
        Check that the plan as constructed so far is valid.
        A plan is valid if it is a tree.
        """
        if not (
            len(self.plan_graph.edges()) == len(self.plan_graph.nodes()) - 1
            and self.root
        ):
            raise ValueError("Plan is not a tree")

    @property
    def root(self):
        [result] = [node for node in self.all_nodes if node.parent is None]
        return result

    @property
    def world(self) -> World:
        return self.context.world

    @property
    def robot(self) -> AbstractRobot:
        return self.context.robot

    @property
    def nodes(self) -> List[PlanNode]:
        """
        All nodes of the plan in depth first order.

        .. info::
            This will only return nodes that have a path from the root node. Nodes that are part of the plan but do not
            have a path from the root node will not be returned. In that case use all_nodes

        :return: All nodes under the root node in depth first order
        """
        return [self.root] + self.root.descendants

    @property
    def all_nodes(self) -> List[PlanNode]:
        """
        All nodes that are part of this plan
        """
        return self.plan_graph.nodes()

    @property
    def edges(self):
        return self.plan_graph.edges()

    def add_plan_entity(self, entity: PlanEntity):
        entity.plan = self

    def remove_plan_entity(self, entity: PlanEntity):
        entity.plan = None

    def merge_nodes(self, node1: PlanNode, node2: PlanNode):
        """
        Merges two nodes into one. The node2 will be removed and all its children will be added to node1.

        :param node1: Node which will remain in the plan
        :param node2: Node which will be removed from the plan
        """
        for node in node2.children:
            self.add_edge(node1, node)
        self.remove_node(node2)

    def remove_node(self, node_for_removal: PlanNode):
        """
        Removes a node from the plan. If the node is not in the plan, it will be ignored.

        :param node_for_removal: Node to be removed
        """
        if node_for_removal.plan is self:
            self.plan_graph.remove_node(node_for_removal.index)
            node_for_removal.index = None
            node_for_removal.layer_index = None
            node_for_removal.plan = None
            node_for_removal.world = None

    def add_node(self, node: PlanNode):
        """
        Adds a node to the plan. The node will not be connected to any other node of the plan.

        :param node: Node to be added
        """
        if node.plan is self:
            return
        self.add_plan_entity(node)
        node.index = self.plan_graph.add_node(node)
        if len(self.plan_graph.nodes()) == 1:
            node.layer_index = 0

    def add_edge(
        self, source: PlanNode, target: PlanNode, target_index: Optional[int] = None
    ):
        """
        Adds an edge to the plan. Nodes that are not in the plan will be added to the plan.

        :param source: Origin node of the edge
        :param target: Target node of the edge
        :param target_index: The index of the target node in the source nodes children.
        If not target_index is given, the target node will be appended to the source's children.
        If the target_index is given, the target node will be inserted at the given index in the source's children and
        the later children are shifted to the right.
        """

        if source.plan is not self:
            self.add_node(source)
        if target.plan is not self:
            self.add_node(target)

        self.plan_graph.add_edge(
            source.index,
            target.index,
            (source, target),
        )

        # if no target index is given, the target is appended to the source's children'
        if target_index is None:
            number_of_children = len(self.plan_graph.successors(source.index))
            target.layer_index = number_of_children - 1
            return

        # shift all children that should come after target to the right
        target.layer_index = target_index
        for child in source.children:
            if child is target:
                continue
            if child.layer_index >= target_index:
                child.layer_index += 1

    def add_edges_from(
        self,
        edges: Iterable[Tuple[PlanNode, PlanNode]],
    ):
        """
        Adds edges to the plan from an iterable of tuples. If one or both nodes are not in the plan, they will be added to the plan.

        :param edges: Iterable of tuples of nodes to be added
        """
        for u, v in edges:
            self.add_edge(u, v)

    def add_nodes_from(self, nodes_for_adding: Iterable[PlanNode]):
        """
        Adds nodes from an Iterable of nodes.

        :param nodes_for_adding: The iterable of nodes
        """
        for node in nodes_for_adding:
            self.add_node(node)

    def insert_below(self, insert_node: PlanNode, insert_below: PlanNode):
        """
        Inserts a node below the given node.

        :param insert_node: The node to be inserted
        :param insert_below: A node of the plan below which the given node should be added
        """
        self.add_edge(insert_below, insert_node)

    def perform(self) -> Any:
        """
        Performs the root node of this plan.

        :return: The return value of the root node
        """
        self.initial_world = deepcopy(self.world)
        result = self.root.perform()
        return result

    def re_perform(self):
        for child in self.root.descendants:
            if child.is_leaf:
                child.perform()

    @property
    def actions(self) -> List[ActionNode]:
        return [node for node in self.nodes if isinstance(node, ActionNode)]

    @property
    def layers(self) -> List[List[PlanNode]]:
        """
        Returns the nodes of the plan layer by layer starting from the root node.

        :return: A list of lists where each list represents a layer
        """
        layers = rx.layers(self.plan_graph, [self.root.index], index_output=False)
        return [
            sorted(layer, key=lambda node: node.layer_index) for layer in layers
        ]

    def _migrate_nodes_from_plan(self, other: Plan) -> PlanNode:
        """
        Steal all nodes from another plan and add them to this plan.
        After this the other plan will be empty.

        :param other: The plan to steal nodes from
        :return: The root node of the other plan mounted in this plan
        """
        root_ref = other.root

        for layer in reversed(other.layers):
            for node in layer:
                self.add_edges_from([(node, child) for child in node.children])

        return root_ref

    def get_nodes_by_designator_type(self, *args: Type[Designator]):
        return [
            node
            for node in self.nodes
            if isinstance(node, DesignatorNode) and isinstance(node.designator, args)
        ]

    def simplify(self):
        """
        Simplifies the plan by merging language nodes that are semantically equivalent.
        This modifies the plan in-place.
        """
        for _, successors in reversed(
            rustworkx.bfs_successors(self.plan_graph, self.root.index)
        ):
            for node in successors:
                node.simplify()

        self.root.simplify()

    def plot(self, layout: str = "bfs"):
        """
        Plots the plan in an interactive browser window

        :param layout: The layout of the plot
        """
        graph, mapping = create_ordered_graph(self)
        plot_rustworkx_interactive(
            graph,
            graph_source=lambda: create_ordered_graph(self)[0],
            layout=layout,
            start=mapping[self.root.index],
        )

    def __repr__(self):
        return f"Plan with {len(self.all_nodes)} nodes"

    def prepare_for_replay(self):
        """
        Prepare the worlds context for a replay.
        Sets the worlds context to the initial world and update its robot.
        """
        self.context.world = deepcopy(self.initial_world)
        self.context.robot = self.context.world.get_semantic_annotation_by_name(
            self.context.robot.name.name
        )

    def replay(self):
        """
        Replay all leaf motions.
        """
        for node in self.nodes:
            if node.is_leaf:
                node.perform()
