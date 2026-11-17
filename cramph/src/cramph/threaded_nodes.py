from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Any, Callable

from cramph.node import SucceedsOnObservingTrue, FailsOnObservingFalse
from cramph.data_types import ObservationStateValues
from cramph.node import StatechartNode

logger = logging.getLogger(__name__)


# %% threaded nodes


@dataclass(eq=False, repr=False, kw_only=True)
class ThreadedNode(StatechartNode, ABC):
    """
    Runs :meth:`run` once in a daemon thread of its own every time it starts, so slow
    work does not hold up the tick that started it.

    What :meth:`run` returned or raised is kept for :meth:`on_tick` to observe, since a
    thread cannot raise into the tick.

    .. warning:: What :meth:`run` calls is usually not serializable, so such a node only
        works in a locally ticked statechart.

    .. warning:: Nothing stops the thread when this node ends, and a thread still
        running from an earlier start can overwrite the outcome of a later one, so
        restart such a node only once :attr:`has_finished` holds.
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

    def wait_until_finished(self) -> None:
        """
        Block until :meth:`run` returned or raised.
        """
        self._thread.join()

    def on_start(self) -> None:
        self._result = None
        self._error = None
        self._thread = threading.Thread(
            target=self._run_and_keep_outcome, name=self.unique_name, daemon=True
        )
        self._thread.start()

    def _run_and_keep_outcome(self) -> None:
        """
        Call :meth:`run` and keep what it returned or raised.
        """
        try:
            self._result = self.run()
        except BaseException as error:  # noqa: BLE001 - handed to the tick
            self._error = error


# %% function calls


@dataclass(eq=False, repr=False, kw_only=True)
class FunctionCall(SucceedsOnObservingTrue, FailsOnObservingFalse, ThreadedNode):
    """
    Calls a function once when it starts, in a thread of its own.

    It succeeds once the function returned, and fails if the function raised one of
    :attr:`failure_types`, so a surrounding node can react to that failure. Any other
    exception is raised out of the tick.

    The tick after the start waits for the function, so no control cycle passes while it
    runs, and functions started in the same tick run at the same time.
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

    def on_tick(self) -> ObservationStateValues | None:
        """
        Wait for the function and observe whether it succeeded.

        :raises BaseException: What the function raised, unless it is one of
            :attr:`failure_types`.
        """
        self.wait_until_finished()
        if self._error is None:
            return ObservationStateValues.TRUE
        if isinstance(self._error, self.failure_types):
            logger.info("%s failed: %s", self.unique_name, self._error)
            return ObservationStateValues.FALSE
        raise self._error
