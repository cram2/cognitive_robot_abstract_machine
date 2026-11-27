"""
Statechart nodes that do their work in a thread of their own, so slow work does not hold
up the ticks.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any, Callable

from cramph.node import SucceedsOnObservingTrue, FailsOnObservingFalse
from cramph.data_types import ObservationStateValue
from cramph.node import StatechartNode

logger = logging.getLogger(__name__)


# %% threaded nodes


@dataclass(eq=False, repr=False, kw_only=True)
class ThreadedNode(StatechartNode, ABC):
    """
    Runs :meth:`run` in a daemon thread of its own every time it starts, and keeps what
    it returned or raised for :meth:`on_tick`. The thread runs on when this node ends,
    but the outcome of a run that a later start superseded is discarded.

    .. warning:: What :meth:`run` calls is usually not serializable, so such a node only
        works in a locally ticked statechart.
    """

    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    """
    The thread :meth:`run` runs in, from the start of this node on.
    """

    _result: Any = field(default=None, init=False, repr=False)
    """
    What :meth:`run` returned the last time it returned.
    """

    _error: BaseException | None = field(default=None, init=False, repr=False)
    """
    What :meth:`run` raised, if it raised.
    """

    _start_count: int = field(default=0, init=False, repr=False)
    """
    How often this node started; only the run of the latest start keeps its outcome.
    """

    @abstractmethod
    def run(self) -> Any:
        """
        The work this node does, in its own thread.

        :return: The outcome :meth:`on_tick` observes.
        """

    @property
    def has_finished(self) -> bool:
        """
        :return: Whether :meth:`run` returned or raised since this node last started.
        """
        return self._thread is not None and not self._thread.is_alive()

    def on_start(self) -> None:
        """
        Start :meth:`run` in a new thread, superseding any run of an earlier start.
        """
        self._result = None
        self._error = None
        self._start_count += 1
        self._thread = threading.Thread(
            target=self._run_and_keep_outcome,
            kwargs={"start_number": self._start_count},
            name=self.unique_name,
            daemon=True,
        )
        self._thread.start()

    def _run_and_keep_outcome(self, *, start_number: int) -> None:
        """
        Call :meth:`run` and keep what it returned or raised, unless this node started
        again since.

        :param start_number: Which start of this node the run belongs to.
        """
        result = None
        error = None
        try:
            result = self.run()
        except BaseException as raised:  # noqa: BLE001 - handed to the tick
            error = raised
        if start_number != self._start_count:
            return
        self._result = result
        self._error = error


# %% function calls


@dataclass(eq=False, repr=False, kw_only=True)
class FunctionCall(SucceedsOnObservingTrue, FailsOnObservingFalse, ThreadedNode):
    """
    Calls a function once when it starts, in a thread of its own, while the statechart
    ticks on.

    It observes unknown until the function returned or raised, then succeeds, or fails
    if the function raised one of :attr:`failure_types`, so a surrounding node can react
    to that failure. Any other exception is raised out of the tick.
    """

    function: Callable[[], Any]
    """
    The function to call.
    """

    failure_types: tuple[type[BaseException], ...] = ()
    """
    The exceptions that make this node fail rather than being raised out of the tick.
    """

    def run(self) -> Any:
        return self.function()

    def on_tick(self) -> ObservationStateValue:
        """
        Observe whether the function succeeded, once it returned or raised.

        :raises BaseException: What the function raised, unless it is one of
            :attr:`failure_types`.
        """
        if not self.has_finished:
            return ObservationStateValue.UNKNOWN
        if self._error is None:
            return ObservationStateValue.TRUE
        if isinstance(self._error, self.failure_types):
            logger.info("%s failed: %s", self.unique_name, self._error)
            return ObservationStateValue.FALSE
        raise self._error
