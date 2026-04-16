from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, List, Type, TYPE_CHECKING, Iterable, Iterator

import rustworkx as rx
from typing_extensions import Union

from giskardpy.motion_statechart.graph_node import Task
from krrood.entity_query_language.query.match import Match

from pycram.datastructures.enums import TaskStatus
from pycram.plans.failures import PlanFailure
from pycram.motion_executor import MotionExecutor

from pycram.plans.plan_entity import PlanEntity
from pycram.datastructures.execution_data import ExecutionData
from pycram.plans.designator import Designator

if TYPE_CHECKING:
    from pycram.robot_plans import ActionDescription, BaseMotion


logger = logging.getLogger("pycram")


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
    The starting time of the function, optional
    """

    end_time: Optional[datetime] = None
    """
    The ending time of the function, optional
    """

    reason: Optional[PlanFailure] = None
    """
    The reason of failure if the action failed.
    """

    result: Optional[Any] = None
    """
    Result from the execution of this node
    """

    index: Optional[int] = field(default=None, init=False, repr=False)
    """
    The index of this node in `self.plan.plan_graph`.
    """

    layer_index: Optional[int] = field(default=None, init=False, repr=False)
    """
    The position of this node in its children.
    The children of a node are interpreted as a list of nodes that have order.
    rustworkx doesn't have order in the children, hence this attribute makes it possible.
    """

    @property
    def parent(self) -> Optional[PlanNode]:
        """
        The parent node of this node, None if this is the root node

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
        All children nodes of this node

        :return:  A list of child nodes
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
        :return: The path from the root node to this node
        """

        paths = rx.all_shortest_paths(
            self.plan.plan_graph, self.index, self.plan.root.index, as_undirected=True
        )
        return [self.plan.plan_graph[i] for i in paths[0][1:]] if len(paths) > 0 else []

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def is_leaf(self) -> bool:
        """
        Returns True if this node is a leaf node

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
        return [
            sibling
            for sibling in self.siblings
            if sibling.layer_index < self.layer_index
        ][-1]

    @property
    def right_neighbour(self) -> Optional[PlanNode]:
        return [
            sibling
            for sibling in self.siblings
            if sibling.layer_index > self.layer_index
        ][0]

    @property
    def previous_nodes(self) -> List[PlanNode]:
        """
        Gets the previous nodes to the given node. Previous meaning the nodes that are before the given one in
        depth first order of nodes.

        :return: The previous nodes as a list of nodes
        """
        previous_nodes = []
        for search_node in self.plan.nodes:
            if search_node == self:
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
        Interrupts the execution of this node and all nodes below
        """
        self.status = TaskStatus.INTERRUPTED
        logger.info(f"Interrupted node: {str(self)}")
        # TODO: cancel giskard execution

    def resume(self):
        """
        Resumes the execution of this node and all nodes below
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
        logger.debug(f"Performing node: {str(self)}")
        for parent in self.path:
            if parent.status == TaskStatus.INTERRUPTED:
                self.status = TaskStatus.INTERRUPTED
                return

        self.status = TaskStatus.RUNNING
        try:
            self.result = self._perform()
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
        This modifies the plan in-place.
        Only implement this if it makes sense for your class to have this ability.
        """
        pass

    @abstractmethod
    def _perform(self):
        """
        Perform the node without managing the fields of this node.
        """


@dataclass(eq=False, repr=False)
class UnderspecifiedNode(PlanNode):
    """
    An action or language expression that is described by an `underspecified(...)` statement.
    This node is used to generate fully specified actions  or language expressions.
    The semantics are: try until it succeeds or fails if the underspecified action is exhausted.
    If you want to limit the number of attempts, add a limit clause to the underspecified action.
    """

    underspecified_action: Match = field(kw_only=True)
    """
    The underspecified statement that can be used to generate actions.
    """

    _action_iterator: Optional[Iterator[ActionDescription]] = field(
        default=None, kw_only=True
    )
    """
    The iterator that is used to generate the actions.
    Only available after the first call to _perform.
    """

    @property
    def designator_type(self) -> Type:
        return self.underspecified_action.type

    def _perform(self):
        if self._action_iterator is None:
            self._action_iterator = self.plan.context.query_backend.evaluate(
                self.underspecified_action
            )

        for grounded_action in self._action_iterator:
            new_child = ActionNode(designator=grounded_action)
            self.add_child(new_child)
            try:
                new_child.perform()
            except PlanFailure:
                continue
            return

    def __repr__(self):
        return f"{self.designator_type.__name__}"


@dataclass
class DesignatorNode(PlanNode, ABC):
    """
    Abstract base class for all nodes that represent a designator.
    """

    designator: Designator = field(kw_only=True)
    """
    The designator that is managed by this node.
    """

    motion_executor: MotionExecutor = None
    """
    Instance of the MotionExecutor used to execute the motion chart of the sub-motions of this action.
    """

    def __post_init__(self):
        self.designator.plan_node = self

    def __repr__(self):
        return f"{type(self.designator).__name__}"

    def collect_motions(self) -> List[Task]:
        """
        Collects all child motions of this action. A motion is considered if it is a direct child of this action node,
        i.e. there is no other action node between this action node and the motion.
        """
        return [
            motion_node.motion.motion_chart
            for motion_node in self.children + [self]
            if isinstance(motion_node, MotionNode)
            # and self is motion_node.parent_action_node
        ]

    def construct_motion_state_chart(self):
        """
        Builds a giskard Motion State Chart from the collected motions of this action node.
        """
        self.motion_executor = MotionExecutor(
            self.collect_motions(),
            self.plan.world,
            ros_node=self.plan.context.ros_node,
            plan_node=self,
        )
        self.motion_executor.construct_msc()

    def execute_motion_state_chart(self):
        """
        Executes the constructed Motion State Chart of this action node.
        """
        self.construct_motion_state_chart()
        self.motion_executor.execute()


@dataclass(eq=False, repr=False)
class ActionNode(DesignatorNode):
    """
    A node representing a fully specified action.
    """

    execution_data: ExecutionData = None
    """
    Additional data that  is collected before and after the execution of the action.
    """

    _world_modification_block_length_pre_perform: Optional[int] = None
    """
    The last model modification block before the execution of this node. 
    Used to check if the model has changed during execution.
    """

    @property
    def action(self) -> ActionDescription:
        return self.designator

    def create_execution_data_pre_perform(self):
        """
        Create the ExecutionData and logs additional information about the execution of this node.
        """
        robot_pose = self.plan.robot.root.global_pose
        exec_data = ExecutionData(robot_pose, self.plan.world.state._data)
        self.execution_data = exec_data
        self._last_world_modification_block_pre_perform_index = len(
            self.plan.world._model_manager.model_modification_blocks
        )

    def update_execution_data_post_perform(self):
        """
        Update the ExecutionData with additional information to the ExecutionData object after performing this node.
        """
        self.execution_data.execution_end_pose = self.plan.robot.root.global_pose

        self.execution_data.execution_end_world_state = self.plan.world.state._data
        self.execution_data.added_world_modifications = (
            self.plan.world._model_manager.model_modification_blocks[
                self._last_world_modification_block_pre_perform_index :
            ]
        )

    def _perform(self):
        logger.debug(f"Performing node: {str(self)} with params: {[f for f in self.action.fields]}")
        self.create_execution_data_pre_perform()

        result = self.action.perform()

        self.execute_motion_state_chart()

        self.update_execution_data_post_perform()

        return result


@dataclass(eq=False, repr=False)
class MotionNode(DesignatorNode):
    """
    A node in the plan representing a fully specified motion.
    Motions are not directly performed. Motions get merged with their siblings into one motion state chart which then is
    executed.
    """

    @property
    def motion(self) -> BaseMotion:
        return self.designator

    def _perform(self):
        """
        Performs this node by performing the respective MotionDesignator. Additionally, checks if one of the parents has
        the status INTERRUPTED and aborts the perform if that is the case.

        :return: The return value of the Motion Designator
        """
        if not any([isinstance(node, ActionNode) for node in self.path]):
            return self.execute_motion_state_chart()
        return self.motion.perform()

    @property
    def parent_action_node(self) -> Optional[ActionNode]:
        """
        Returns the next resolved action node in the plan above this motion node.
        """
        for node in self.path:
            if isinstance(node, ActionNode):
                return node
        return None


ActionLike = Union[Match, Designator, PlanNode]
