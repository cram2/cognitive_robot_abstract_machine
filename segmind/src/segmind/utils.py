from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from typing_extensions import Optional

logger = logging.getLogger(__name__)

class PropagatingThread(threading.Thread, ABC):
    exc: Optional[Exception] = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
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
