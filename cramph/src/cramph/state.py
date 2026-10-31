from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from typing_extensions import (
    TYPE_CHECKING,
    Any,
    Iterator,
    MutableMapping,
    Self,
)

from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.ormatic.utils import classproperty
from cramph.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from cramph.node import (
    LastObservationVariable,
    LifeCycleVariable,
    ObservationVariable,
    StatechartNode,
)

if TYPE_CHECKING:
    from cramph.statechart import Statechart

# %% the state of every node of a statechart


@dataclass(repr=False, eq=False)
class State(MutableMapping[StatechartNode, float], SubclassJSONSerializer, ABC):
    """
    Maps every node of a statechart to a scalar value, backed by a single contiguous
    array indexed by :attr:`~StatechartNode.index`.
    """

    statechart: Statechart
    """
    The statechart whose nodes are the keys of this mapping.
    """

    data: NDArray[np.float64] = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    """
    One entry per node, ordered by :attr:`~StatechartNode.index`.
    """

    @classproperty
    @abstractmethod
    def default_value(cls) -> float:
        """
        :return: The value a node has before anything changed it.
        """

    def append_default_value(self) -> None:
        """
        Appends :attr:`default_value` to :attr:`data`, keeping it in sync with a newly
        added node.
        """
        self.data = np.append(self.data, self.default_value)

    def life_cycle_symbols(self) -> list[LifeCycleVariable]:
        """
        :return: The life cycle variable of every node, in node order.
        """
        return [node.life_cycle_variable for node in self.statechart.nodes]

    def observation_symbols(self) -> list[ObservationVariable]:
        """
        :return: The observation variable of every node, in node order.
        """
        return [node.observation_variable for node in self.statechart.nodes]

    def last_observation_symbols(self) -> list[LastObservationVariable]:
        """
        :return: The last observation variable of every node, in node order.
        """
        return [node.last_observation for node in self.statechart.nodes]

    def __getitem__(self, node: StatechartNode) -> float:
        """
        :param node: The node to look up.
        :return: The value stored for `node`, read from :attr:`data` at :attr:`~StatechartNode.index`.
        """
        return float(self.data[node.index])

    def __setitem__(self, node: StatechartNode, value: float) -> None:
        """
        Writes `value` into :attr:`data` at `node`'s :attr:`~StatechartNode.index`.

        :param node: The node to write the value for.
        :param value: The value to store.
        """
        self.data[node.index] = value

    def __delitem__(self, node: StatechartNode) -> None:
        """
        Removes the entry for `node` from :attr:`data`.

        .. warning:: This shifts the indices of all nodes after `node`, but does not update
            their :attr:`~StatechartNode.index`, so the state and the nodes fall out of sync.

        :param node: The node whose entry to remove.
        """
        self.data = np.delete(self.data, node.index)

    def __iter__(self) -> Iterator[StatechartNode]:
        return iter(self.statechart.nodes)

    def __len__(self) -> int:
        return self.data.shape[0]

    def keys(self) -> list[StatechartNode]:
        """
        :return: All nodes of the statechart, i.e. the keys of this mapping.
        """
        return self.statechart.nodes

    def items(self) -> list[tuple[StatechartNode, float]]:
        """
        :return: (node, value) pairs for every node of the statechart.
        """
        return [(node, self[node]) for node in self.statechart.nodes]

    def values(self) -> list[float]:
        """
        :return: The value of every node, in node order.
        """
        return [self[node] for node in self.keys()]

    def __contains__(self, node: StatechartNode) -> bool:
        return node in self.statechart.nodes

    def snapshot(self) -> Self:
        """
        :return: A state over the same nodes of the same :attr:`statechart`, holding a
            copy of the current values that later changes to this state leave alone.
        """
        return self.__class__(
            statechart=self.statechart,
            data=self.data.copy(),
        )

    def to_json(self, **kwargs: Any) -> dict[str, Any]:
        """
        :return: The JSON representation of the base class, extended with the raw :attr:`data` array.
        """
        return {**super().to_json(**kwargs), "data": self.data.tolist()}

    @classmethod
    def _from_json(cls, data: dict[str, Any], **kwargs: Any) -> Self:
        """
        Reconstruct a state from its JSON representation.

        :param data: The JSON dict, as produced by :meth:`to_json`.
        :param kwargs: Must contain the owning `statechart`.
        :return: The deserialized state.
        """
        statechart = kwargs["statechart"]
        return cls(
            statechart=statechart,
            data=np.array(data["data"], dtype=np.float64),
        )

    def __str__(self) -> str:
        return str({str(symbol.name): value for symbol, value in self.items()})

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: Self) -> bool:
        """
        :param other: The object to compare against.
        :return: True if `other` is a :class:`State` with the same :attr:`data`.
        .. note:: The owning :attr:`statechart` is not compared.
        """
        if not isinstance(other, State):
            return NotImplemented
        return np.array_equal(self.data, other.data)


@dataclass(repr=False, eq=False)
class LifeCycleState(State):
    """
    The life cycle state of every node in a statechart, see
    :class:`~cramph.statechart.Statechart`.
    """

    @classproperty
    def default_value(cls) -> LifeCycleValues:
        """
        :return: Not started, which every node starts out as.
        """
        return LifeCycleValues.NOT_STARTED

    def __getitem__(self, node: StatechartNode) -> LifeCycleValues:
        """
        :param node: The node to look up.
        :return: The life cycle state of `node`, as a :class:`LifeCycleValues` member.
        """
        return LifeCycleValues(super().__getitem__(node))

    def __str__(self) -> str:
        return str(
            {
                str(symbol.name): LifeCycleValues(value).name
                for symbol, value in self.items()
            }
        )


@dataclass(repr=False, eq=False)
class ObservationState(State):
    """
    The observation state of every node in a statechart, see
    :class:`~cramph.statechart.Statechart`.
    """

    @classproperty
    def default_value(cls) -> ObservationStateValues:
        """
        :return: Unknown, because a node that is not running is not observing.
        """
        return ObservationStateValues.UNKNOWN

    def __getitem__(self, node: StatechartNode) -> ObservationStateValues:
        """
        :param node: The node to look up.
        :return: What `node` observes, as an :class:`ObservationStateValues` member.
        """
        return ObservationStateValues(super().__getitem__(node))


@dataclass(repr=False, eq=False)
class LastObservationState(State):
    """
    The observation every node of a statechart took most recently.

    .. seealso:: :attr:`~cramph.node.StatechartNode.last_observation`
    """

    @classproperty
    def default_value(cls) -> ObservationStateValues:
        """
        :return: Unknown, because a node that has not started has not observed anything.
        """
        return ObservationStateValues.UNKNOWN

    def __getitem__(self, node: StatechartNode) -> ObservationStateValues:
        """
        :param node: The node to look up.
        :return: What `node` observed most recently, as an
            :class:`ObservationStateValues` member.
        """
        return ObservationStateValues(super().__getitem__(node))
