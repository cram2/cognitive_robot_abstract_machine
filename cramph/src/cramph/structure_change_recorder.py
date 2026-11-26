"""
Where a node reports the changes to its children, its chosen children and its
conditions, whether or not it belongs to a statechart yet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from cramph.composites import CompositeNodeChoosingItsChild
    from cramph.data_types import TransitionKind
    from cramph.node import CompositeNode, StatechartNode


# %% recorder interface


class StructureChangeRecorder(ABC):
    """
    Records the changes to the structure of a statechart that a node makes to itself.
    """

    @abstractmethod
    def _record_changed_children(self, *, node: CompositeNode) -> None:
        """
        Records that the children of `node` changed.
        """

    @abstractmethod
    def _record_changed_chosen_children(
        self, *, node: CompositeNodeChoosingItsChild
    ) -> None:
        """
        Records that `node` chose a child.
        """

    @abstractmethod
    def _record_changed_condition(
        self, *, node: StatechartNode, transition_kind: TransitionKind
    ) -> None:
        """
        Records that the condition of the transition of `node` of `transition_kind`
        changed.
        """


# %% nodes outside a statechart


@dataclass(kw_only=True)
class StructureChangeRecorderOutsideStatechart(StructureChangeRecorder):
    """
    Records nothing, for a node that belongs to no statechart: the statechart it joins
    records its children, chosen children and conditions as they are then.
    """

    def _record_changed_children(self, *, node: CompositeNode) -> None:
        return None

    def _record_changed_chosen_children(
        self, *, node: CompositeNodeChoosingItsChild
    ) -> None:
        return None

    def _record_changed_condition(
        self, *, node: StatechartNode, transition_kind: TransitionKind
    ) -> None:
        return None
