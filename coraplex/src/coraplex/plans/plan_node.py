from __future__ import annotations

import logging
from abc import abstractmethod, ABC
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any, List, Dict, Type, TYPE_CHECKING, Iterable, Iterator

from typing_extensions import Union

from coraplex.plans.designator import Designator
from krrood.entity_query_language.query.match import Match
from coraplex.datastructures.enums import TaskStatus
from coraplex.datastructures.execution_data import ExecutionData
from coraplex.plans.executables import (
    Executable,
    GiskardExecutable,
    UnderspecifiedExecutable,
)
from coraplex.plans.failures import PlanFailure
from coraplex.plans.plan_entity import PlanEntity
from coraplex.utils import split_list_by_type

if TYPE_CHECKING:
    from giskardpy.motion_statechart.graph_node import Task
    from coraplex.robot_plans import ActionDescription, BaseMotion


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
        Gets the previous nodes to the given node. Previous meaning the nodes that are before the given one in
        depth first order of nodes.

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
        This modifies the plan in-place.
        Only implement this if it makes sense for your class to have this ability.
        """
        pass

    def merge(self, other: PlanNode):
        """
        Merges this node with another, this will mount the children of the other node under this one and remove the other
        node from the plan.

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
        from the plan. Subclasses that reference other plan nodes override this to
        avoid dangling references to the removed node.

        :param replaced_node: The node being removed from the plan.
        :param replacement_node: The node that takes its place.
        """

    @abstractmethod
    def notify(self):
        """
        Perform the node without managing the fields of this node.
        """

    def parse(self) -> Executable: ...

    def merge_motion_executables(
        self, executables: List[Executable]
    ) -> List[Executable]:
        """
        Merge consecutive giskard executables into a single one while leaving the
        other executables untouched and in their original order.
        """
        result = []
        for group in split_list_by_type(executables, GiskardExecutable):
            if not isinstance(group[0], GiskardExecutable):
                result.extend(group)
                continue
            result.append(
                GiskardExecutable(
                    motion_mappings=self.merge_motion_mappings(group),
                    context=self.plan.context,
                )
            )
        return result

    def merge_motion_mappings(
        self, motions: List[GiskardExecutable]
    ) -> Dict[MotionNode, Task]:
        """
        Combine the motion mappings of several giskard executables into one mapping.
        """
        new_mappings = {}
        for motion in motions:
            new_mappings.update(motion.motion_mappings)
        return new_mappings


@dataclass(eq=False, repr=False)
class UnderspecifiedNode(PlanNode):
    """
    An action or language expression that is described by an underspecified `an(...)` match statement.
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
    Only available after the first call to notify.
    """

    current_candidate: Optional[ActionNode] = field(
        default=None, init=False, repr=False
    )
    """
    The action candidate this node currently resolves to, set by `advance` at
    execution time. On failure, `advance` replaces it with the next candidate.
    """

    @property
    def designator_type(self) -> Type:
        return self.underspecified_action.type

    def _next_candidate(self) -> Optional[ActionNode]:
        """
        Pull the next grounded action from the iterator and make it the current
        candidate.

        :return: The new candidate node, or None if the iterator is exhausted.
        """
        if self._action_iterator is None:
            self._action_iterator = self.plan.context.query_backend.evaluate(
                self.underspecified_action
            )

        grounded_action = next(self._action_iterator, None)
        if grounded_action is None:
            self._action_iterator = None
            return None

        candidate = ActionNode(designator=grounded_action)
        self.add_child(candidate)
        self.current_candidate = candidate
        return candidate

    def stop_grounding(self) -> None:
        """
        Release the action iterator once no further candidate will be requested from it.

        Between candidates the iterator is left suspended (rather than exhausted) so a later
        retry can resume the search instead of restarting it; a suspended generator keeps every
        value its frame holds alive, including resources a candidate generator only builds to
        validate against (for example a location's deep-copied test world). Once a candidate is
        accepted and no retry will happen, closing the iterator here releases those resources
        immediately instead of retaining them for this node's whole lifetime.
        """
        if self._action_iterator is not None:
            self._action_iterator.close()
            self._action_iterator = None

    def notify(self):
        # Resolution is deferred to execution time: the underspecified statement can
        # only be grounded once the preceding actions have run and mutated the world
        # (e.g. the torso is raised, the object is in the gripper). The grounding
        # happens in UnderspecifiedExecutable, so expansion does nothing here.
        pass

    def advance(self) -> bool:
        """
        Resolve the next candidate and expand it against the current world state.
        Driven by :class:`~pycram.plans.executables.UnderspecifiedExecutable` to
        ground the action at execution time, and reused by failure handling to retry
        with a freshly generated action.

        :return: True if a new candidate was generated, False if the iterator is exhausted.
        """
        if self._next_candidate() is None:
            return False
        self.current_candidate.notify()
        return True

    def parse(self) -> Executable:
        # Defer resolution to execution: the returned executable grounds the action
        # when it is reached, against the world state produced by the preceding nodes.
        return UnderspecifiedExecutable(node=self, context=self.plan.context)

    def __repr__(self):
        return f"{self.designator_type.__name__}"


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
        Merges this designator node with a child if they are of the same type and
        carry the same parameters.
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


@dataclass(eq=False, repr=False)
class ActionNode(DesignatorNode):
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

    @property
    def parent_action_node(self) -> Optional[ActionNode]:
        """
        Returns the next action node in the plan above this node, None if this is the outermost action.
        """
        for node in self.path:
            if isinstance(node, ActionNode):
                return node
        return None

    def notify(self):

        self.create_execution_data_pre_perform()

        if not self.children:
            self.action.expand()

        # recursively expand nested actions, conditions are only evaluated during execution
        for child in self.children:
            child.notify()

        # TODO: This can't stay here
        self.update_execution_data_post_perform()

    def parse(self) -> Executable:
        children = self.children
        pre_condition_node = children.pop(0)
        post_condition_node = children.pop(-1)

        child_execs = [child.parse() for child in children]
        merged = self.merge_motion_executables(child_execs)

        motion_execs = [
            executable
            for executable in merged
            if isinstance(executable, GiskardExecutable)
        ]
        if len(motion_execs) == 1:
            # The action body is a single motion state chart, so the conditions
            # can be evaluated inside it (gating start/end, aborting on failure).
            motion_exec = motion_execs[0]
            motion_exec.pre_condition_node = pre_condition_node
            motion_exec.post_condition_node = post_condition_node
            return motion_exec

        giskard_child_execs = [
            executable
            for executable in child_execs[0].execution_list
            if isinstance(executable, GiskardExecutable)
        ]
        giskard_child_execs[0].pre_condition_node = pre_condition_node
        giskard_child_execs[-1].post_condition_node = post_condition_node
        return child_execs[0]

    def execute(self):
        self.parse().execute()


@dataclass(eq=False, repr=False)
class MotionNode(DesignatorNode):
    """
    A node in the plan representing a fully specified motion.
    Motions are not directly performed. Motions get merged with their siblings into one motion state chart which then is
    executed.
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
        Performs this node by performing the respective MotionDesignator. Additionally, checks if one of the parents has
        the status INTERRUPTED and aborts the perform if that is the case.

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

    def parse(self) -> Executable:
        task = self.motion.motion_chart

        return GiskardExecutable(
            motion_mappings={self: task}, context=self.plan.context
        )


ActionLike = Union[Match, Designator, PlanNode]
