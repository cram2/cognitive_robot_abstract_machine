"""
The changes to the structure of a statechart and their recording, so that another copy
of the statechart can repeat them without expanding any node again.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum

from typing_extensions import TYPE_CHECKING, Any, Self

from krrood.adapters.json_serializer import SubclassJSONSerializer, from_json, to_json
from krrood.patterns.classproperty import classproperty
from krrood.symbolic_math.symbolic_math import Scalar
from cramph.composites import CompositeNodeChoosingItsChild
from cramph.data_types import TransitionKind
from cramph.node import (
    CompositeNode,
    DeserializedNodeTracker,
    StatechartNode,
    TransitionCondition,
)
from cramph.structure_change_recorder import StructureChangeRecorder
from semantic_digital_twin.world_description.world_entity import (
    WorldEntityReferenceWriter,
)

if TYPE_CHECKING:
    from cramph.statechart import Statechart


# %% serialization keys


class StatechartChangeJSONKey(StrEnum):
    """
    Keys a serialized change of a statechart carries.
    """

    NODE = "node"
    """
    The node the change is about: the whole node where it joins the statechart, its id
    otherwise.
    """

    CHILDREN = "children"
    """
    The ids of the children, in order.
    """

    CONDITION = "condition"
    """
    The transition condition, as :class:`~cramph.node.TransitionCondition` writes it.
    """

    CHANGES = "changes"
    """
    The changes of a block, in the order they are applied.
    """


class ChangeApplicationOrder(IntEnum):
    """
    The order in which the kinds of change of one block are applied.

    Nodes join first, so that children and conditions can name them, and leave only once
    nothing names them any more. Changes of the same order keep the order they were
    recorded in.
    """

    NODE_ADDITION = 0
    """
    A node joins the statechart.
    """

    CHILDREN_CHANGE = 1
    """
    A composite node gets its children.
    """

    CONDITION_CHANGE = 2
    """
    A transition gets its condition.
    """

    CHOSEN_CHILDREN_CHANGE = 3
    """
    A node choosing its child gets the children it chose.
    """

    NODE_REMOVAL_OR_REQUEST = 4
    """
    A node leaves the statechart, or is to be reset on the next settled tick or built
    again on the next compile.
    """


# %% changes


@dataclass(kw_only=True)
class StatechartChange(SubclassJSONSerializer, ABC):
    """
    One change to the structure of a statechart, naming its nodes by their id when it is
    serialized.
    """

    @classproperty
    @abstractmethod
    def application_order(cls) -> ChangeApplicationOrder:
        """
        :return: When a change of this kind is applied, relative to the other changes of
            its block.
        """

    @abstractmethod
    def apply(self, *, statechart: Statechart) -> None:
        """
        Makes the same change to `statechart`, without expanding any node.

        :param statechart: The statechart to change, which holds every node the change
            names, apart from the node a :class:`NodeAddition` adds.
        """


@dataclass(kw_only=True)
class ChangeOfNode(StatechartChange, ABC):
    """
    A change about one node of the statechart, which it names by its id.
    """

    node: StatechartNode
    """
    The node the change is about.
    """

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        Writes the node by its id.
        """
        return {
            **super().to_json(**kwargs),
            StatechartChangeJSONKey.NODE: self.node._node_id,
        }

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        Resolves the node through the :class:`~cramph.node.DeserializedNodeTracker` in
        `kwargs`.
        """
        nodes = DeserializedNodeTracker.from_kwargs(from_json_kwargs=kwargs)
        return cls(node=nodes.get(key=data[StatechartChangeJSONKey.NODE]))


@dataclass(kw_only=True)
class NodeAddition(StatechartChange):
    """
    A node joins the statechart, already expanded.
    """

    node: StatechartNode
    """
    The node that joins.
    """

    @classproperty
    def application_order(cls) -> ChangeApplicationOrder:
        return ChangeApplicationOrder.NODE_ADDITION

    def apply(self, *, statechart: Statechart) -> None:
        statechart._register_node(node=self.node)

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        Writes the whole node, so that a statechart that does not hold it can add it.
        """
        return {
            **super().to_json(**kwargs),
            StatechartChangeJSONKey.NODE: to_json(self.node, **kwargs),
        }

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        Reads the whole node, which joins the tracker of the document.
        """
        return cls(node=from_json(data[StatechartChangeJSONKey.NODE], **kwargs))


@dataclass(kw_only=True)
class NodeRemoval(ChangeOfNode):
    """
    A node leaves the statechart, together with everything below it.
    """

    @classproperty
    def application_order(cls) -> ChangeApplicationOrder:
        return ChangeApplicationOrder.NODE_REMOVAL_OR_REQUEST

    def apply(self, *, statechart: Statechart) -> None:
        statechart.remove_node(node=self.node)


@dataclass(kw_only=True)
class ResetRequest(ChangeOfNode):
    """
    A node and everything below it are to be reset on the next settled tick, see
    :meth:`~cramph.statechart.Statechart.request_reset`.
    """

    @classproperty
    def application_order(cls) -> ChangeApplicationOrder:
        return ChangeApplicationOrder.NODE_REMOVAL_OR_REQUEST

    def apply(self, *, statechart: Statechart) -> None:
        statechart.request_reset(node=self.node)


@dataclass(kw_only=True)
class RebuildRequest(ChangeOfNode):
    """
    A node is to be built again on the next compile, see
    :meth:`~cramph.statechart.Statechart.request_rebuild`.
    """

    @classproperty
    def application_order(cls) -> ChangeApplicationOrder:
        return ChangeApplicationOrder.NODE_REMOVAL_OR_REQUEST

    def apply(self, *, statechart: Statechart) -> None:
        statechart.request_rebuild(node=self.node)


@dataclass(kw_only=True)
class ChangeNamingChildren(ChangeOfNode, ABC):
    """
    A change about a composite node and some of its children, in order.
    """

    node: CompositeNode
    """
    The node whose children the change is about.
    """

    children: list[StatechartNode] = field(default_factory=list)
    """
    Children of :attr:`node`, in order.
    """

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        Writes the children by their ids.
        """
        return {
            **super().to_json(**kwargs),
            StatechartChangeJSONKey.CHILDREN: [
                child._node_id for child in self.children
            ],
        }

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        Resolves the node and its children through the
        :class:`~cramph.node.DeserializedNodeTracker` in `kwargs`.
        """
        nodes = DeserializedNodeTracker.from_kwargs(from_json_kwargs=kwargs)
        return cls(
            node=nodes.get(key=data[StatechartChangeJSONKey.NODE]),
            children=[
                nodes.get(key=child_id)
                for child_id in data[StatechartChangeJSONKey.CHILDREN]
            ],
        )


@dataclass(kw_only=True)
class ChildrenChange(ChangeNamingChildren):
    """
    A composite node holds :attr:`children` as its children from now on.
    """

    @classproperty
    def application_order(cls) -> ChangeApplicationOrder:
        return ChangeApplicationOrder.CHILDREN_CHANGE

    def apply(self, *, statechart: Statechart) -> None:
        self.node.set_children(children=self.children)
        for child in self.children:
            child.parent_node = self.node


@dataclass(kw_only=True)
class ChosenChildrenChange(ChangeNamingChildren):
    """
    A node choosing its child has chosen :attr:`children`, in that order.
    """

    node: CompositeNodeChoosingItsChild
    """
    The node whose chosen children changed.
    """

    @classproperty
    def application_order(cls) -> ChangeApplicationOrder:
        return ChangeApplicationOrder.CHOSEN_CHILDREN_CHANGE

    def apply(self, *, statechart: Statechart) -> None:
        self.node.restore_chosen_children(children=self.children)


@dataclass(kw_only=True)
class ConditionChange(StatechartChange):
    """
    A transition of a node has a new condition.
    """

    owner: StatechartNode
    """
    The node the transition belongs to.
    """

    kind: TransitionKind
    """
    The kind of the transition.
    """

    expression: Scalar
    """
    The condition the transition has now.
    """

    @classproperty
    def application_order(cls) -> ChangeApplicationOrder:
        return ChangeApplicationOrder.CONDITION_CHANGE

    def apply(self, *, statechart: Statechart) -> None:
        self.owner.set_condition(transition_kind=self.kind, expression=self.expression)

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        Writes the condition as the transition it belongs to writes it.
        """
        transition = TransitionCondition(
            kind=self.kind, expression=self.expression, owner=self.owner
        )
        return {
            **super().to_json(**kwargs),
            StatechartChangeJSONKey.CONDITION: transition.to_json(**kwargs),
        }

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        Builds the change from the transition the condition was written as.
        """
        transition = TransitionCondition.from_json(
            data=data[StatechartChangeJSONKey.CONDITION], **kwargs
        )
        return cls(
            owner=transition.owner,
            kind=transition.kind,
            expression=transition.expression,
        )


# %% changes waiting for the nodes they name


@dataclass(kw_only=True, eq=False)
class PendingChange(ABC):
    """
    A change recorded but not written into a block yet, because it is written with the
    value it has once every node it names has joined the statechart.
    """

    node: StatechartNode
    """
    The node the change is about.
    """

    @property
    @abstractmethod
    def named_nodes(self) -> list[StatechartNode]:
        """
        :return: The nodes the change names as it would be written now, which have to
            join the statechart before it is written.
        """

    @abstractmethod
    def create_change(self) -> StatechartChange:
        """
        :return: The change, with the value it has now.
        """

    def is_same_change_as(self, *, other: PendingChange) -> bool:
        """
        :return: Whether `other` is the same kind of change about :attr:`node` itself,
            whatever other node it compares equal to, so it is written only once.
        """
        return type(other) is type(self) and other.node is self.node


@dataclass(kw_only=True, eq=False)
class PendingChildrenChange(PendingChange):
    """
    The children of a composite node changed.
    """

    node: CompositeNode
    """
    The node whose children changed.
    """

    @property
    def named_nodes(self) -> list[StatechartNode]:
        return self.node.nodes

    def create_change(self) -> ChildrenChange:
        return ChildrenChange(node=self.node, children=list(self.node.nodes))


@dataclass(kw_only=True, eq=False)
class PendingChosenChildrenChange(PendingChange):
    """
    A node choosing its child chose another child.
    """

    node: CompositeNodeChoosingItsChild
    """
    The node whose chosen children changed.
    """

    @property
    def named_nodes(self) -> list[StatechartNode]:
        return self.node.chosen_children

    def create_change(self) -> ChosenChildrenChange:
        return ChosenChildrenChange(node=self.node, children=self.node.chosen_children)


@dataclass(kw_only=True, eq=False)
class PendingConditionChange(PendingChange):
    """
    The condition of a transition of a node changed.
    """

    transition_kind: TransitionKind
    """
    The kind of the transition.
    """

    @property
    def named_nodes(self) -> list[StatechartNode]:
        return self.node._get_transition(
            transition_kind=self.transition_kind
        ).node_dependencies

    def create_change(self) -> ConditionChange:
        return ConditionChange(
            owner=self.node,
            kind=self.transition_kind,
            expression=self.node._get_transition(
                transition_kind=self.transition_kind
            ).expression,
        )

    def is_same_change_as(self, *, other: PendingChange) -> bool:
        """
        Is also about the same transition, see :meth:`PendingChange.is_same_change_as`.
        """
        return (
            super().is_same_change_as(other=other)
            and other.transition_kind == self.transition_kind
        )


# %% blocks of changes


@dataclass(kw_only=True)
class StatechartChangeBlock(SubclassJSONSerializer):
    """
    The changes the outermost :meth:`~cramph.statechart.Statechart.modify` block made
    to a statechart, in the order they are applied.
    """

    changes: list[StatechartChange] = field(default_factory=list)
    """
    The changes, ordered by :attr:`StatechartChange.application_order` and then by the
    order they were recorded in.
    """

    def apply(self, *, statechart: Statechart) -> None:
        """
        Makes the same changes to `statechart`, in one
        :meth:`~cramph.statechart.Statechart.modify` block, without expanding any node.

        :param statechart: A statechart holding the same nodes as the one the changes
            were made to, before they were made.
        """
        with statechart.modify():
            for change in self.changes:
                change.apply(statechart=statechart)

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        World entities are written as references, because whoever applies the block
        resolves them against its own world, which has the same entities.
        """
        kwargs = {**kwargs, **WorldEntityReferenceWriter().create_kwargs()}
        return {
            **super().to_json(**kwargs),
            StatechartChangeJSONKey.CHANGES: [
                change.to_json(**kwargs) for change in self.changes
            ],
        }

    @classmethod
    def _from_json(cls, *, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        Resolves the nodes the changes name through the
        :class:`~cramph.node.DeserializedNodeTracker` in `kwargs`, which also tracks
        the nodes the block adds, so that later changes of the block find them.
        """
        DeserializedNodeTracker.from_kwargs(from_json_kwargs=kwargs)
        return cls(
            changes=[
                from_json(change, **kwargs)
                for change in data[StatechartChangeJSONKey.CHANGES]
            ]
        )


# %% recording the changes of a statechart


@dataclass(kw_only=True, eq=False)
class StructureChangeRecorderInStatechart(StructureChangeRecorder):
    """
    Records the changes to the structure of a statechart and closes them into a
    :class:`StatechartChangeBlock` when the outermost
    :meth:`~cramph.statechart.Statechart.modify` block ends, or right away outside one.
    """

    statechart: Statechart = field(repr=False)
    """
    The statechart whose changes are recorded.
    """

    change_blocks: list[StatechartChangeBlock] = field(
        default_factory=list, init=False, repr=False
    )
    """
    The changes made to the structure of :attr:`statechart`, one block per outermost
    :meth:`~cramph.statechart.Statechart.modify` block, oldest first. Applying them in
    order to an empty statechart gives a statechart of the same structure.
    """

    recorded_changes: list[StatechartChange] = field(
        default_factory=list, init=False, repr=False
    )
    """
    The nodes added and removed and the resets and rebuilds requested since the latest
    block closed, in order.
    """

    _pending_changes: list[PendingChange] = field(
        default_factory=list, init=False, repr=False
    )
    """
    The changes to children, chosen children and conditions recorded but not written
    into a block yet, in the order they were first recorded.
    """

    def record_change(self, *, change: StatechartChange) -> None:
        """
        Records `change` for the next block.
        """
        self.recorded_changes.append(change)
        self._close_change_block_outside_modify()

    def record_joined_node(self, *, node: StatechartNode) -> None:
        """
        Records that the children, the chosen children and every condition of `node`,
        which just joined, are to be written into a block once the nodes they name
        joined too.
        """
        if isinstance(node, CompositeNode):
            self._record_changed_children(node=node)
        if isinstance(node, CompositeNodeChoosingItsChild):
            self._record_changed_chosen_children(node=node)
        for transition_kind in TransitionKind:
            self._record_changed_condition(node=node, transition_kind=transition_kind)

    def _record_changed_children(self, *, node: CompositeNode) -> None:
        self._record_pending_change(pending=PendingChildrenChange(node=node))

    def _record_changed_chosen_children(
        self, *, node: CompositeNodeChoosingItsChild
    ) -> None:
        self._record_pending_change(pending=PendingChosenChildrenChange(node=node))

    def _record_changed_condition(
        self, *, node: StatechartNode, transition_kind: TransitionKind
    ) -> None:
        self._record_pending_change(
            pending=PendingConditionChange(node=node, transition_kind=transition_kind)
        )

    def _record_pending_change(self, *, pending: PendingChange) -> None:
        """
        Records `pending`, to be written into a block with the value it has once every
        node it names joined :attr:`statechart`. A pending change that is the same
        change, see :meth:`PendingChange.is_same_change_as`, stays the only one.
        """
        if not any(
            recorded.is_same_change_as(other=pending)
            for recorded in self._pending_changes
        ):
            self._pending_changes.append(pending)
        self._close_change_block_outside_modify()

    def _close_change_block_outside_modify(self) -> None:
        """
        Closes the changes made so far into a block, unless a
        :meth:`~cramph.statechart.Statechart.modify` block is open, which closes them
        once the outermost one ends.
        """
        if self.statechart._open_modifications == 0:
            self.close_change_block()

    def close_change_block(self) -> None:
        """
        Appends the changes made since the latest block to :attr:`change_blocks`, as one
        block, unless there are none.

        Children and conditions are written as they are now, once every node they name
        belongs to :attr:`statechart`; until then they wait for a later block.

        :raises StatechartBrokenByFailedModificationError: If a
            :meth:`~cramph.statechart.Statechart.modify` block of :attr:`statechart`
            raised.
        """
        self.statechart.check_not_broken()
        changes = [*self.recorded_changes, *self._take_ready_changes()]
        self.recorded_changes = []
        if not changes:
            return
        changes.sort(key=lambda change: change.application_order)
        self.change_blocks.append(StatechartChangeBlock(changes=changes))

    def _take_ready_changes(self) -> list[StatechartChange]:
        """
        Forgets the pending changes about nodes that left :attr:`statechart`.

        :return: The pending changes whose nodes all joined :attr:`statechart`, with the
            value they have now, which no longer wait then.
        """
        ready_changes = []
        still_pending = []
        for pending in self._pending_changes:
            if pending.node._statechart is not self.statechart:
                continue
            if all(
                named._statechart is self.statechart for named in pending.named_nodes
            ):
                ready_changes.append(pending.create_change())
            else:
                still_pending.append(pending)
        self._pending_changes = still_pending
        return ready_changes
