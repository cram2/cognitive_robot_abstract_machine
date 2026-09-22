"""
What a live source offers for its state to be queryable.

The bridge depends on this abstraction only, so a demo supplies its domain vocabulary
without cramera knowing anything about that demo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass

from typing_extensions import List, TYPE_CHECKING

from cramera.knowledge.presets import Preset
from cramera.knowledge.queryable_knowledge import QueryableKnowledge

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


class NoQuerySourceRegistered(Exception):
    """
    Raised when the bridge is asked a question and no demo offered to answer it.
    """

    def __init__(self) -> None:
        """
        Explain that this session has no registered source of queryable state.
        """
        super().__init__("no query source is registered on this bridge")


@dataclass
class LiveQuerySource(ABC):
    """
    Queryable state from an attached world or a running demo.

    A source declares what its state *is*; how a question is compiled, evaluated and
    rendered is not its concern.
    """

    def read_scope(self) -> AbstractContextManager[object]:
        return nullcontext()

    @abstractmethod
    def title(self) -> str:
        """
        Short name of what is being queried, shown as the panel's answer source.
        """

    @abstractmethod
    def knowledge(self) -> List[QueryableKnowledge]:
        """
        The bodies of knowledge this demo offers to be questioned about, in the order
        their questions are shown.

        Read whenever a query runs, so an answer reflects the demo's current state.
        """

    @abstractmethod
    def presets(self) -> List[Preset]:
        """
        Ready-made queries the panel offers as buttons.
        """

    def unlisted_presets(self) -> List[Preset]:
        """
        Ready-made queries recognized when a question asks for one, but not shown as
        buttons.

        A question naming one type out of many -- one kind of detected event, one kind
        of performed action -- is worth recognizing for every type a demo records, which
        is more questions than a panel has room to show (see
        :class:`~cramera.knowledge.presets.PresetsPerType`).
        """
        return []
