"""
The changes to the structure of a statechart, recorded so that another copy of the
statechart can repeat them without expanding any node again.
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
    nothing names them any more.
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

    NODE_REMOVAL = 4
    """
    A node leaves the statechart.
    """

    RESET_REQUEST = 5
    """
    A node is to be reset on the next settled tick.
    """

    REBUILD_REQUEST = 6
    """
    A node is to be built again on the next compile.
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
        return ChangeApplicationOrder.NODE_REMOVAL

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
        return ChangeApplicationOrder.RESET_REQUEST

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
        return ChangeApplicationOrder.REBUILD_REQUEST

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


# %% blocks of changes


@dataclass(kw_only=True)
class StatechartChangeBlock(SubclassJSONSerializer):
    """
    The changes the outermost :meth:`~cramph.statechart.Statechart.modify` block made
    to a statechart, in the order they are applied.
    """

    changes: list[StatechartChange] = field(default_factory=list)
    """
    The changes, ordered by :attr:`StatechartChange.application_order`.
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
