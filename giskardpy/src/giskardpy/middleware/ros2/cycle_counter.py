from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CycleCounter:
    """
    Counts how often the motion server made progress.

    Both the idle loop and the control loop tick it, so an observer can wait for the
    server to make progress without knowing whether it is currently waiting for a goal
    or executing one.
    """

    completed_cycles: int = 0
    """
    Number of cycles the motion server completed, monotonically increasing.
    """

    def tick(self) -> None:
        """
        Record that one cycle was completed.
        """
        self.completed_cycles += 1
