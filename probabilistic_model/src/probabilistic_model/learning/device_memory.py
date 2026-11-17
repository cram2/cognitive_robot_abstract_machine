"""
How much memory the device that trains a circuit has left, and how much memory one
training step needs, so that mini-batches can be sized to fit.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from krrood.patterns.path_enumeration import PathEnumeration
from krrood.ormatic.utils import classproperty

from probabilistic_model.learning.exceptions import (
    MissingMemoryFieldError,
    UnmeasurableStepMemoryError,
)

# %% linux memory files


class LinuxMemoryFile(PathEnumeration):
    """
    The files Linux reports memory in.
    """

    MEMORY_INFORMATION = "/proc/meminfo"
    PROCESS_STATUS = "/proc/self/status"
    CLEAR_REFERENCES = "/proc/self/clear_refs"
    CONTROL_GROUP_LIMIT = "/sys/fs/cgroup/memory.max"
    CONTROL_GROUP_USAGE = "/sys/fs/cgroup/memory.current"


class LinuxMemoryField(StrEnum):
    """
    The fields of :attr:`LinuxMemoryFile.MEMORY_INFORMATION` and
    :attr:`LinuxMemoryFile.PROCESS_STATUS` that are read, each in kibibytes.
    """

    AVAILABLE = "MemAvailable"
    RESIDENT = "VmRSS"
    PEAK_RESIDENT = "VmHWM"


class LinuxMemoryValue(StrEnum):
    """
    Fixed values of the Linux memory files.
    """

    UNLIMITED = "max"
    """
    The content of :attr:`LinuxMemoryFile.CONTROL_GROUP_LIMIT` without a limit.
    """

    RESET_PEAK_RESIDENT = "5"
    """
    Written to :attr:`LinuxMemoryFile.CLEAR_REFERENCES`, resets the peak resident memory.
    """


# %% memory of a device


class DeviceMemory(ABC):
    """
    The memory of a device that arrays are stored on.
    """

    @abstractmethod
    def free_bytes(self) -> int:
        """
        :return: The number of bytes that can still be allocated on the device.
        """


class PeakTrackingMemory(DeviceMemory, ABC):
    """
    Device memory that records the most memory in use since its peak was last reset.
    """

    @abstractmethod
    def in_use_bytes(self) -> int:
        """
        :return: The number of bytes in use now.
        """

    @abstractmethod
    def peak_bytes(self) -> int:
        """
        :return: The most bytes in use at once since :meth:`reset_peak`.
        """

    @abstractmethod
    def reset_peak(self) -> None:
        """
        Start recording the peak anew from the memory in use now.
        """


@dataclass(kw_only=True)
class HostMemory(PeakTrackingMemory):
    """
    The main memory of a Linux host, as seen by this process and its control group.
    """

    @classproperty
    def bytes_per_kibibyte(cls) -> int:
        """
        :return: The unit the Linux memory fields are given in, in bytes.
        """
        return 1024

    def free_bytes(self) -> int:
        available = self.field_bytes(
            memory_file=LinuxMemoryFile.MEMORY_INFORMATION,
            memory_field=LinuxMemoryField.AVAILABLE,
        )
        if not LinuxMemoryFile.CONTROL_GROUP_LIMIT.exists():
            return available
        limit = LinuxMemoryFile.CONTROL_GROUP_LIMIT.read_text().strip()
        if limit == LinuxMemoryValue.UNLIMITED:
            return available
        usage = int(LinuxMemoryFile.CONTROL_GROUP_USAGE.read_text())
        return min(available, int(limit) - usage)

    def in_use_bytes(self) -> int:
        return self.field_bytes(
            memory_file=LinuxMemoryFile.PROCESS_STATUS,
            memory_field=LinuxMemoryField.RESIDENT,
        )

    def peak_bytes(self) -> int:
        return self.field_bytes(
            memory_file=LinuxMemoryFile.PROCESS_STATUS,
            memory_field=LinuxMemoryField.PEAK_RESIDENT,
        )

    def reset_peak(self) -> None:
        LinuxMemoryFile.CLEAR_REFERENCES.write_text(
            LinuxMemoryValue.RESET_PEAK_RESIDENT
        )

    def field_bytes(
        self, *, memory_file: LinuxMemoryFile, memory_field: LinuxMemoryField
    ) -> int:
        """
        :param memory_file: The file that holds the field, one ``name: value kB`` line
            per field.
        :param memory_field: The field to read.
        :return: The value of the field, in bytes.
        """
        for line in memory_file.read_text().splitlines():
            name, value = line.split(":", 1)
            if name == memory_field:
                return int(value.split()[0]) * self.bytes_per_kibibyte
        raise MissingMemoryFieldError(
            memory_file=str(memory_file), memory_field=memory_field
        )


# %% memory of a training step


@dataclass(kw_only=True)
class ProbeMeasurement:
    """
    The memory one training step needed on a batch of a given number of rows.
    """

    rows: int
    """
    The number of rows of the batch.
    """

    step_bytes: int
    """
    The bytes the step allocated beyond what was in use before it.
    """


@dataclass(kw_only=True)
class StepMemory:
    """
    The memory one training step allocates beyond what is already in use, as a fixed
    part and a part that grows with the rows of the batch.
    """

    fixed_bytes: float
    """
    The bytes a step allocates whatever the size of its batch.
    """

    bytes_per_row: float
    """
    The bytes a step allocates for every row of its batch.
    """

    @classmethod
    def from_two_probes(
        cls, *, smaller: ProbeMeasurement, larger: ProbeMeasurement
    ) -> StepMemory:
        """
        :param smaller: The step on the batch with fewer rows.
        :param larger: The step on the batch with more rows.
        :return: The line through both measurements.
        :raises UnmeasurableStepMemoryError: If the larger batch did not need more
            memory.
        """
        bytes_per_row = (larger.step_bytes - smaller.step_bytes) / (
            larger.rows - smaller.rows
        )
        if bytes_per_row <= 0:
            raise UnmeasurableStepMemoryError(
                smaller_rows=smaller.rows, bytes_per_row=bytes_per_row
            )
        return cls(
            fixed_bytes=smaller.step_bytes - bytes_per_row * smaller.rows,
            bytes_per_row=bytes_per_row,
        )

    def rows_fitting_in(self, *, free_bytes: int) -> int:
        """
        :param free_bytes: The memory a step may allocate.
        :return: The most rows a batch can have for its step to fit.
        """
        return math.floor((free_bytes - self.fixed_bytes) / self.bytes_per_row)
