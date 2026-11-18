"""
The memory of the devices arrays are stored on: how much is free, how much is in use, and
the most that was in use at once.
"""

from __future__ import annotations

# %% imports
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from krrood.patterns.classproperty import classproperty
from krrood.patterns.path_enumeration import PathEnumeration

from krrood_compute_hardware.exceptions import MissingMemoryFieldError

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


# %% memory budget


@dataclass(kw_only=True)
class BudgetedMemory(DeviceMemory):
    """
    The memory of a device of which a process may use at most a fixed amount, since other
    processes share the device.
    """

    memory: DeviceMemory
    """
    The memory of the device.
    """

    budget_bytes: int
    """
    The most bytes the process may allocate.
    """

    def free_bytes(self) -> int:
        return min(self.memory.free_bytes(), self.budget_bytes)
