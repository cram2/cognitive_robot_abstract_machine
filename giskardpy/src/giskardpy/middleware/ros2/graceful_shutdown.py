from __future__ import annotations

import signal
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from types import FrameType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from signal import _HANDLER as SignalHandler


@dataclass
class GracefulShutdownSignals(AbstractContextManager):
    """
    Turns SIGINT and SIGTERM into a :class:`KeyboardInterrupt` for the duration of a
    ``with`` block, and restores whatever handlers were installed before on exit.

    A ROS2 launch file sends SIGINT first and escalates to SIGTERM if the process does
    not exit in time. Handling both the same way lets the robot stop the same way
    regardless of which one arrives.
    """

    _original_sigint_handler: SignalHandler = field(init=False, default=None)
    """
    The SIGINT handler that was installed before entering the ``with`` block.
    """

    _original_sigterm_handler: SignalHandler = field(init=False, default=None)
    """
    The SIGTERM handler that was installed before entering the ``with`` block.
    """

    def __enter__(self) -> GracefulShutdownSignals:
        self._original_sigint_handler = signal.signal(
            signal.SIGINT, self._raise_keyboard_interrupt
        )
        self._original_sigterm_handler = signal.signal(
            signal.SIGTERM, self._raise_keyboard_interrupt
        )
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        signal.signal(signal.SIGTERM, self._original_sigterm_handler)

    @staticmethod
    def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt
