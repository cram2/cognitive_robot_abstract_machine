from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import field, dataclass

from typing_extensions import Optional

logger = logging.getLogger(__name__)

@dataclass(eq=False)
class PropagatingThread(threading.Thread, ABC):
    """
    Thread that propagates exceptions.
    """

    exc: Optional[Exception] = field(default=None, init=False)
    """
    Exception that was raised in the thread.
    """

    def __post_init__(self):
        super().__init__()  # calls threading.Thread.__init__()
        self.kill_event = threading.Event()

    def run(self):
        self.exc = None
        self._run()

    @abstractmethod
    def _run(self):
        pass

    def stop(self):
        """
        Stop the event detector.
        """
        self.kill_event.set()
        self._join()

    @abstractmethod
    def _join(self, timeout=None):
        pass
