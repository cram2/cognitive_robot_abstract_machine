"""
Jobs that tests divide among workers.
"""

from __future__ import annotations

# %% imports
import time
from dataclasses import dataclass
from multiprocessing.managers import EventProxy
from pathlib import Path

from krrood_compute_resources.workers import ParallelJob

# %% jobs


@dataclass(kw_only=True)
class NumbersWrittenToAFile(ParallelJob[int, int]):
    """
    Writes every number it works through as a line at the end of a file, which several
    processes can share, and gives the number back.
    """

    path: Path
    """
    The file.
    """

    seconds_per_number: float = 0.0
    """
    How long writing one number takes, in seconds.
    """

    def work_through_items(self, *, items: list[int], stop: EventProxy) -> list[int]:
        written = []
        for number in items:
            if stop.is_set():
                break
            time.sleep(self.seconds_per_number)
            with self.path.open("a") as file:
                file.write(f"{number}\n")
            written.append(number)
        return written

    def written_numbers(self) -> list[int]:
        """
        :return: The numbers written so far, in the order they were written.
        """
        if not self.path.exists():
            return []
        return [int(line) for line in self.path.read_text().splitlines()]
