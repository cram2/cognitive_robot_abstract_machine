"""
What a class diagram reports about its own building.

Building a diagram of a large package resolves the type of every field of every class
and takes long enough to be worth watching. A process that starts another one to do it
sees only its output, so the progress is written there, one line per class, for that
process to read back.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import Optional

LINE_PREFIX = "krrood-class-diagram-progress "
"""
What marks a line of output as a progress report rather than something a log wrote.
"""


class ProgressEnvironmentVariable(StrEnum):
    """
    Environment variables asking a class diagram to report its progress.
    """

    REPORT_PROGRESS = "KRROOD_REPORT_CLASS_DIAGRAM_PROGRESS"
    """
    Set by a process that reads the reports back; unset, nothing is written.
    """


class ProgressField(StrEnum):
    """
    Names a report carries its parts under.
    """

    CLASS_NAME = "class_name"
    """
    The class the report is about.
    """

    TOTAL_CLASSES = "total_classes"
    """
    How many classes the diagram being built holds.
    """


@dataclass
class ClassDiagramProgress:
    """
    One class of a diagram, finished.
    """

    class_name: str
    """
    Name of the class whose relations were just created.
    """

    total_classes: int
    """
    How many classes the diagram holds, so a reader knows what to count towards.
    """

    @classmethod
    def from_line(cls, line: str) -> Optional[ClassDiagramProgress]:
        """
        Read a report back from a line of a process's output.

        :param line: One line of output, of any kind.
        :return: The report the line carries, or nothing when it carries none.
        """
        if not line.startswith(LINE_PREFIX):
            return None
        payload = json.loads(line[len(LINE_PREFIX) :])
        return cls(
            payload[ProgressField.CLASS_NAME], payload[ProgressField.TOTAL_CLASSES]
        )

    def to_line(self) -> str:
        """
        Render this report as one line of output.

        :return: The line, without its terminating line break.
        """
        payload = {
            ProgressField.CLASS_NAME.value: self.class_name,
            ProgressField.TOTAL_CLASSES.value: self.total_classes,
        }
        return LINE_PREFIX + json.dumps(payload)


def is_progress_wanted() -> bool:
    """
    Whether whoever started this process reads progress reports.

    :return: Whether to report.
    """
    return bool(os.environ.get(ProgressEnvironmentVariable.REPORT_PROGRESS))


def report_progress(class_name: str, total_classes: int) -> None:
    """
    Write one finished class to the output the starting process reads.

    :param class_name: Name of the class that is done.
    :param total_classes: How many classes the diagram holds.
    """
    print(ClassDiagramProgress(class_name, total_classes).to_line(), flush=True)
