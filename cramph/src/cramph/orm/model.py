"""
The mappings ORMatic cannot derive from cramph's classes on its own: a statechart keeps
its nodes, the conditions of their transitions and the history of its state in fields it
builds itself rather than takes as arguments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from typing_extensions import Self

from cramph.context import StatechartContext
from cramph.state import LifeCycleState, ObservationState
from cramph.state_history import StateHistoryItem
from cramph.statechart import Statechart
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.world import World

# %% the history of a statechart's state


@dataclass
class NodeStatesAtTick:
    """
    The life cycle and observation value of every node of a statechart at one tick, as a
    :class:`~cramph.state_history.StateHistoryItem` records them.
    """

    tick_count: int
    """
    The number of ticks run when the values were recorded.
    """

    life_cycle_values: list[float]
    """
    The life cycle value of every node that had joined the statechart by then, in index
    order.
    """

    observation_values: list[float]
    """
    The observation value of every node that had joined the statechart by then, in index
    order.
    """

    cancelling_node_index: int | None = None
    """
    The index of the node whose cancellation ended the statechart with these values,
    None if they did not end it.
    """

    cancellation_error: Exception | None = None
    """
    The exception the cancellation ended the statechart with, None if the values did not
    end it.
    """

    @classmethod
    def from_item(cls, item: StateHistoryItem) -> Self:
        """
        :param item: A snapshot of a statechart's state.
        :return: The values the snapshot holds.
        """
        return cls(
            tick_count=item.tick_count,
            life_cycle_values=item.life_cycle_state.data.tolist(),
            observation_values=item.observation_state.data.tolist(),
            cancelling_node_index=(
                None if item.cancelled_by is None else item.cancelled_by.index
            ),
            cancellation_error=item.cancellation_error,
        )

    def item_of(self, statechart: Statechart) -> StateHistoryItem:
        """
        :param statechart: The statechart whose nodes the values belong to.
        :return: The snapshot holding the values.
        """
        return StateHistoryItem(
            tick_count=self.tick_count,
            life_cycle_state=LifeCycleState(
                statechart=statechart,
                data=np.array(self.life_cycle_values, dtype=np.float64),
            ),
            observation_state=ObservationState(
                statechart=statechart,
                data=np.array(self.observation_values, dtype=np.float64),
            ),
            cancelled_by=(
                None
                if self.cancelling_node_index is None
                else statechart.get_node_by_index(self.cancelling_node_index)
            ),
            cancellation_error=self.cancellation_error,
        )


# %% a statechart


@dataclass(eq=False)
class StatechartMapping(AlternativeMapping[Statechart]):
    """
    A statechart with its nodes, the conditions of their transitions and the history of
    its state; it is read back in its current state, without expanding a node again.
    """

    context: StatechartContext
    """
    The context the statechart was built and ticked in.
    """

    pass_limit: int
    """
    See :attr:`~cramph.statechart.Statechart.pass_limit`.
    """

    passes_per_nesting_level: int
    """
    See :attr:`~cramph.statechart.Statechart.passes_per_nesting_level`.
    """

    structure: str
    """
    The nodes, the conditions of their transitions and the children of every composite
    node, as the JSON text :meth:`~cramph.statechart.Statechart.to_json` writes, naming
    the entities of the context's world by their id.
    """

    state_history: list[NodeStatesAtTick]
    """
    The state of the statechart at every tick it changed at, in the order of the ticks.
    """

    @classmethod
    def from_domain_object(cls, obj: Statechart) -> Self:
        return cls(
            context=obj.context,
            pass_limit=obj.pass_limit,
            passes_per_nesting_level=obj.passes_per_nesting_level,
            structure=json.dumps(obj.to_json()),
            state_history=[
                NodeStatesAtTick.from_item(item) for item in obj.state_history.items
            ],
        )

    def to_domain_object(self) -> Statechart:
        world = self.context.world
        statechart = Statechart.from_json(
            json.loads(self.structure),
            context=self.context,
            world=world,
            **WorldEntityWithIDKwargsTracker.from_world(world).create_kwargs(),
        )
        statechart.pass_limit = self.pass_limit
        statechart.passes_per_nesting_level = self.passes_per_nesting_level
        for states in self.state_history:
            statechart.state_history.append(states.item_of(statechart))
        latest = statechart.state_history.latest_item
        if latest is not None:
            statechart.life_cycle_state.data = (
                latest.life_cycle_state.copy_covering_every_node().data
            )
            statechart.observation_state.data = (
                latest.observation_state.copy_covering_every_node().data
            )
        statechart.compiler.add_transitions()
        return statechart

    @classmethod
    def required_pre_build_classes(cls) -> list[type]:
        return [World]
