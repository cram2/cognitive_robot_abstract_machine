from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, List, Type, TYPE_CHECKING, Iterable

from typing_extensions import Union

from coraplex.plans.designator import Designator
from giskardpy.motion_statechart.graph_node import Goal
from krrood.entity_query_language.query.match import Match
from coraplex.datastructures.enums import TaskStatus
from coraplex.datastructures.execution_data import ExecutionData
from coraplex.plans.executables import (
    Executable,
    GiskardExecutable,
)
from coraplex.plans.failures import PlanFailure
from coraplex.plans.motion_state_chart_building import BuildsMotionStateChart
from coraplex.plans.plan_entity import PlanEntity

if TYPE_CHECKING:
    from giskardpy.motion_statechart.graph_node import Task
    from coraplex.datastructures.dataclasses import Context
    from coraplex.robot_plans.actions.base import ActionDescription
    from coraplex.robot_plans.motions.base import BaseMotion


logger = logging.getLogger(__name__)


def sort_by_layer_index(nodes: Iterable[PlanNode]) -> Iterable[PlanNode]:
    """
    :param nodes: The nodes to sort
    :return: An iterator of the sorted nodes by layer index
    """
    return sorted(nodes, key=lambda node: node.layer_index)


@dataclass(eq=False)
class PlanNode(PlanEntity):
    """
    A node in the plan.
    """

    status: TaskStatus = TaskStatus.CREATED
    """
    The status of the node from the TaskStatus enum.
    """

    start_time: Optional[datetime] = field(default_factory=datetime.now)
    """
    The starting time of the function, optional.
    """

    end_time: Optional[datetime] = None
    """
    The ending time of the function, optional.
    """

    reason: Optional[PlanFailure] = None
    """
    The reason of failure if the action failed.
    """

    result: Optional[Any] = None
    """
    Result from the execution of this node.
    """

    index: Optional[int] = field(default=None, init=False, repr=False)
    """
    The index of this node in `self.plan.plan_graph`.
    """

    layer_index: Optional[int] = field(default=None, init=False, repr=False)
    """
    The position of this node in its children.

    The children of a node are interpreted as a list of nodes that have order. rustworkx
    doesn't have order in the children, hence this attribute makes it possible.
    """

    @property
    def context(self) -> Context:
        """
        :return: The context of the plan this node belongs to.
        """
        return self.plan.context

    @property
    def parent(self) -> Optional[PlanNode]:
        """
        The parent node of this node, None if this is the root node.

        :return: The parent node
        """
        return (
            self.plan.plan_graph.predecessors(self.index)[0]
            if self.plan.plan_graph.predecessors(self.index)
            else None
        )

    @property
    def children(self) -> List[PlanNode]:
        """
        All children nodes of this node.

        :return: A list of child nodes
        """
        children = self.plan.plan_graph.successors(self.index)
        return list(sort_by_layer_index(children))

    @property
    def descendants(self) -> List[PlanNode]:
        """
        :return: A list of all descendants in breadth-first order.
        """
        result = []
        queue = deque(self.children)

        while queue:
            node = queue.popleft()
            result.append(node)
            queue.extend(node.children)

        return result

    @property
    def path(self) -> List[PlanNode]:
        """
        :return: The ancestors of this node, ordered from the immediate parent
            up to and including the root node. Empty for the root node.

        The plan is a tree, so the path is found by walking parent links rather
        than by a shortest-path search. This avoids depending on contiguous
        rustworkx node indices, which no longer hold once nodes are removed.
        """
        ancestors = []
        node = self.parent
        while node is not None:
            ancestors.append(node)
            node = node.parent
        return ancestors

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def is_leaf(self) -> bool:
        """
        Returns True if this node is a leaf node.

        :return: True if this node is a leaf node
        """
        return self.children == []

    @property
    def siblings(self) -> List[PlanNode]:
        """
        :return: All siblings of this node.
        """
        if self.parent is None:
            return []
        return list(
            sort_by_layer_index(
                child for child in self.parent.children if child is not self
            )
        )

    @property
    def left_siblings(self) -> List[PlanNode]:
        return [
            sibling
            for sibling in self.siblings
            if sibling.layer_index < self.layer_index
        ]

    @property
    def right_siblings(self) -> List[PlanNode]:
        return [
            sibling
            for sibling in self.siblings
            if sibling.layer_index > self.layer_index
        ]

    @property
    def left_neighbour(self) -> Optional[PlanNode]:
        """
        :return: The closest sibling to the left, or None if this is the leftmost.
        """
        left_siblings = self.left_siblings
        return left_siblings[-1] if left_siblings else None

    @property
    def right_neighbour(self) -> Optional[PlanNode]:
        """
        :return: The closest sibling to the right, or None if this is the rightmost.
        """
        right_siblings = self.right_siblings
        return right_siblings[0] if right_siblings else None

    @property
    def previous_nodes(self) -> List[PlanNode]:
        """
        Gets the previous nodes to the given node.

        Previous meaning the nodes that are before the given one in depth first order of
        nodes.

        :return: The previous nodes as a list of nodes
        """
        previous_nodes = []
        for search_node in self.plan.nodes:
            if search_node is self:
                break
            previous_nodes.append(search_node)
        return previous_nodes

    def get_previous_node_by_designator_type(
        self, *type_: Type[Designator]
    ) -> Optional[DesignatorNode]:
        """
        :param type_: The types of the designator to search for.
        :return: The previous node with a designator of the specified type, or None if not found.
        """
        for sibling in reversed(self.previous_nodes):
            if isinstance(sibling, DesignatorNode) and isinstance(
                sibling.designator, type_
            ):
                return sibling
        return None

    def __hash__(self):
        return id(self)

    def __repr__(self, *args, **kwargs):
        return f"{type(self).__name__}"

    def interrupt(self):
        """
        Interrupts the execution of this node and all nodes below.
        """
        self.status = TaskStatus.INTERRUPTED
        logger.info(f"Interrupted node: {str(self)}")
        # TODO: cancel giskard execution

    def resume(self):
        """
        Resumes the execution of this node and all nodes below.
        """
        self.status = TaskStatus.RUNNING

    def pause(self):
        """
        Suspends the execution of this node and all nodes below.
        """
        self.status = TaskStatus.PAUSE

    def add_child(self, child: PlanNode):
        self.plan.add_edge(self, child)

    @property
    def is_interrupted(self) -> bool:
        return any(
            parent.status == TaskStatus.INTERRUPTED for parent in [self] + self.path
        )

    @property
    def is_paused(self) -> bool:
        return any(parent.status == TaskStatus.PAUSE for parent in [self] + self.path)

    def perform(self):
        """
        Perform the node and update the fields of this node.
        """
        for parent in self.path:
            if parent.status == TaskStatus.INTERRUPTED:
                self.status = TaskStatus.INTERRUPTED
                return

        self.status = TaskStatus.RUNNING
        try:
            self.notify()
            self.result = self.parse().execute()
        except PlanFailure as e:
            self.status = TaskStatus.FAILED
            self.reason = e
            raise e
        finally:
            self.end_time = datetime.now()
        self.status = TaskStatus.SUCCEEDED

    def mount_subplan(self, root: PlanNode):
        """
        Mount an entire plan as a child of to this node.

        :param root: The root node of the plan to be mounted
        """
        self.plan._migrate_nodes_from_plan(root.plan)
        self.add_child(root)

    def simplify(self):
        """
        Simplifies the plan by merging nodes that are semantically equivalent.

        This modifies the plan in-place. Only implement this if it makes sense for your
        class to have this ability.
        """
        pass

    def merge(self, other: PlanNode):
        """
        Merges this node with another, this will mount the children of the other node
        under this one and remove the other node from the plan.

        :param other: The other node to merge
        """
        for grand_child in other.children:
            grand_child.redirect_node_reference(other, self)
            self.plan.add_edge(
                self, grand_child, other.layer_index + grand_child.layer_index
            )
        self.plan.plan_graph.remove_edge(self.index, other.index)
        self.plan.remove_node(other)

    def redirect_node_reference(
        self, replaced_node: PlanNode, replacement_node: PlanNode
    ) -> None:
        """
        Update references this node holds to ``replaced_node`` so they point to
        ``replacement_node`` instead.

        Called when ``replaced_node`` is merged into ``replacement_node`` and removed
        from the plan. Subclasses that reference other plan nodes override this to avoid
        dangling references to the removed node.

        :param replaced_node: The node being removed from the plan.
        :param replacement_node: The node that takes its place.
        """

    @abstractmethod
    def notify(self):
        """
        Perform the node without managing the fields of this node.
        """

    def parse(self) -> Executable: ...

    @property
    def has_motions(self) -> bool:
        """
        Whether this subtree contributes any node to a motion state chart.

        Used to skip nodes that would otherwise produce an empty goal.
        """
        return any(child.has_motions for child in self.children)

    @property
    def contains_execution_boundary(self) -> bool:
        """
        Whether this node or any of its descendants splits the plan into separate motion
        state charts.
        """
        return any(
            isinstance(node, ExecutionBoundaryNode)
            for node in [self] + self.descendants
        )

    def __node_info__(self):
        return [
            f"status: {self.status.name}",
            f"start: {self.start_time}",
            f"end: {self.end_time}",
            f"result: {self.result}",
            f"reason: {self.reason}",
        ]

    def __node_label__(self):
        return f"{self.__class__.__name__}"


@dataclass(eq=False, repr=False)
class ExecutionBoundaryNode(ABC, PlanNode):
    """
    A PlanNode that interrupts the merging of surrounding motions into one chart.
    """


@dataclass(eq=True, repr=False)
class DesignatorNode(PlanNode, ABC):
    """
    Abstract base class for all nodes that represent a designator.
    """

    designator: Designator = field(kw_only=True)
    """
    The designator that is managed by this node.
    """

    def __post_init__(self):
        self.designator.plan_node = self

    def __repr__(self):
        return f"{type(self.designator).__name__}"

    def simplify(self):
        """
        Merges this designator node with a child if they are of the same type and carry
        the same parameters.
        """
        for child in list(self.children):
            if not isinstance(child, DesignatorNode):
                continue
            if type(self.designator) is not type(child.designator):
                continue
            if (
                self.designator.designator_parameter
                != child.designator.designator_parameter
            ):
                continue
            self.merge(child)

    def __hash__(self):
        return id(self)

    def __node_info__(self):
        parent_infos = super().__node_info__()
        designator_field = [
            f"{field.name}: {getattr(self.designator, field.name)}"
            for field in self.designator.fields
        ]
        parent_infos.append(
            "---------------- Designator Parameter --------------------"
        )
        parent_infos.extend(
            [
                f"Designator Type: {self.designator.__class__.__name__}",
                *designator_field,
            ]
        )
        return parent_infos

    def __node_label__(self):
        return f"{self.designator.__class__.__name__}"


@dataclass(eq=False, repr=False)
class ActionNode(DesignatorNode, BuildsMotionStateChart):
    """
    A node representing a fully specified action.
    """

    execution_data: Optional[ExecutionData] = None
    """
    Additional data that is collected before and after the execution of the action.
    """

    _last_world_modification_block_pre_perform_index: Optional[int] = None
    """
    Index of the last model modification block before the execution of this node.

    Used to check if the model has changed during execution.
    """

    @property
    def action(self) -> ActionDescription:
        return self.designator

    def create_execution_data_pre_perform(self):
        """
        Create the ExecutionData and logs additional information about the execution of
        this node.

        .. note: With the current implementation, the exact recording of execution data is not possible. So this is
        not called at the moment.
        """
        robot_pose = self.plan.robot.root.global_pose
        exec_data = ExecutionData(robot_pose, self.plan.world.state._data)
        self.execution_data = exec_data
        self._last_world_modification_block_pre_perform_index = len(
            self.plan.world._model_manager.model_modification_blocks
        )

    def update_execution_data_post_perform(self):
        """
        Update the ExecutionData with additional information to the ExecutionData object
        after performing this node.
        """
        self.execution_data.execution_end_pose = self.plan.robot.root.global_pose

        self.execution_data.execution_end_world_state = self.plan.world.state._data
        self.execution_data.added_world_modifications = (
            self.plan.world._model_manager.model_modification_blocks[
                self._last_world_modification_block_pre_perform_index :
            ]
        )

    @property
    def parent_action_node(self) -> Optional[ActionNode]:
        """
        Returns the next action node in the plan above this node, None if this is the
        outermost action.
        """
        for node in self.path:
            if isinstance(node, ActionNode):
                return node
        return None

    def notify(self):
        if not self.children:
            self.action.expand()

        # recursively expand nested actions, conditions are only evaluated during execution
        for child in self.children:
            child.notify()

    @property
    def body_children(self) -> List[PlanNode]:
        """
        :return: The children forming the action body, without its pre- and
            post-condition.
        """
        return self.children[1:-1]

    def add_to_motion_state_chart(
        self, parent_goal: Goal, executable: GiskardExecutable
    ) -> Goal:
        """
        Add this action's body as its own goal below `parent_goal`.

        .. note:: A nested action's conditions are not evaluated inside the surrounding
            motion state chart; only the conditions of the action a chart is built for
            are, see :meth:`parse`.
        """
        goal = self.create_goal()
        parent_goal.add_node(goal)
        self.add_children_to_motion_state_chart(goal, self.body_children, executable)
        return goal

    def parse(self) -> Executable:
        """
        Parse the action body into an executable, gating it with the action's
        conditions.

        The pre-condition gates the first motion state chart of the body and the post-
        condition the last one.
        """
        children = self.children
        pre_condition_node = children[0]
        post_condition_node = children[-1]

        executable = self.parse_children(self.body_children)
        giskard_executables = executable.giskard_executables
        if not giskard_executables:
            return executable
        giskard_executables[0].pre_condition_node = pre_condition_node
        giskard_executables[-1].post_condition_node = post_condition_node
        return executable

    def execute(self):
        self.parse().execute()


@dataclass(eq=False, repr=False)
class MotionNode(DesignatorNode, BuildsMotionStateChart):
    """
    A node in the plan representing a fully specified motion.

    Motions are not directly performed. Motions get merged with their siblings into one
    motion state chart which then is executed.
    """

    designator: BaseMotion = field(kw_only=True)
    """
    Reference to the motion designator which is linked to this node.
    """

    @property
    def motion(self) -> BaseMotion:
        return self.designator

    def notify(self):
        """
        Performs this node by performing the respective MotionDesignator.

        Additionally, checks if one of the parents has the status INTERRUPTED and aborts
        the perform if that is the case.

        :return: The return value of the Motion Designator
        """
        pass
        # return self.motion.perform()

    @property
    def parent_action_node(self) -> Optional[ActionNode]:
        """
        Returns the next resolved action node in the plan above this motion node.
        """
        for node in self.path:
            if isinstance(node, ActionNode):
                return node
        return None

    @property
    def has_motions(self) -> bool:
        return True

    def add_to_motion_state_chart(
        self, parent_goal: Goal, executable: GiskardExecutable
    ) -> Task:
        """
        Add this motion's giskard task below `parent_goal` and record it on
        `executable`.
        """
        task = self.motion.motion_chart
        parent_goal.add_node(task)
        executable.motion_mappings[self] = task
        return task

    def parse(self) -> Executable:
        return self.create_giskard_executable([self])


ActionLike = Union[Match, Designator, PlanNode]
