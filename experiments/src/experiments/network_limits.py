"""
Check that this machine can receive the large messages the robot sends it.

The world snapshot served by ``/semantic_digital_twin/fetch_world`` is about 30 MB in a
single service response, and camera frames are hundreds of kilobytes each. Both travel
as UDP datagrams far larger than the link's maximum transmission unit, so the kernel has
to hold many partly reassembled datagrams at once.

Over a wire that is fine, because nothing is lost and every datagram completes. Over a
wireless link it is not: the stock 4 MB reassembly cache, whose entries live for 30
seconds, fills with datagrams that lost a fragment, and those evict the ones still
arriving. Nothing completes, the reliable delivery retransmits, and the retransmissions
saturate the link. Measured on the world service before the limits were raised, 374,828
of 375,918 fragments failed to reassemble and the call never returned; after, the same
call answered in 11 seconds.

The failure is silent and slow, so a run that would hit it is stopped here instead, with
the command that fixes it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import List, Optional

from krrood.exceptions import DataclassException

# %% the settings that decide whether a large message arrives


class KernelSetting(StrEnum):
    """
    The kernel settings a large message's survival depends on, by their ``sysctl``
    names.
    """

    FRAGMENT_CACHE_CEILING = "net.ipv4.ipfrag_high_thresh"
    FRAGMENT_CACHE_FLOOR = "net.ipv4.ipfrag_low_thresh"
    FRAGMENT_LIFETIME = "net.ipv4.ipfrag_time"
    RECEIVE_BUFFER_CEILING = "net.core.rmem_max"
    RECEIVE_BUFFER_DEFAULT = "net.core.rmem_default"
    DEVICE_BACKLOG = "net.core.netdev_max_backlog"


@dataclass(frozen=True)
class SettingRequirement(ABC):
    """
    What one kernel setting has to be for a large message to arrive.
    """

    setting: KernelSetting
    """
    The setting this speaks for.
    """

    value: int
    """
    The value it is measured against.
    """

    @abstractmethod
    def is_met(self, actual: int) -> bool:
        """
        Whether a machine's own value satisfies this.

        :param actual: The value the machine reports.
        """


@dataclass(frozen=True)
class AtLeast(SettingRequirement):
    """
    A setting that must not be smaller than the value named.
    """

    def is_met(self, actual: int) -> bool:
        return actual >= self.value


@dataclass(frozen=True)
class AtMost(SettingRequirement):
    """
    A setting that must not be larger than the value named.
    """

    def is_met(self, actual: int) -> bool:
        return actual <= self.value


LARGE_MESSAGE_REQUIREMENTS: List[SettingRequirement] = [
    AtLeast(KernelSetting.FRAGMENT_CACHE_CEILING, 268435456),
    AtLeast(KernelSetting.FRAGMENT_CACHE_FLOOR, 201326592),
    AtMost(KernelSetting.FRAGMENT_LIFETIME, 3),
    AtLeast(KernelSetting.RECEIVE_BUFFER_CEILING, 67108864),
    AtLeast(KernelSetting.RECEIVE_BUFFER_DEFAULT, 67108864),
    AtLeast(KernelSetting.DEVICE_BACKLOG, 30000),
]
"""
The configuration measured to carry the world snapshot over this setup's wireless link.

These were raised and tested together, so this is the set known to work rather than the
smallest set that would.
"""


def raise_limits_command() -> str:
    """
    :return: The command that puts this machine's limits where they need to be, until it
        is next rebooted.
    """
    settings = " ".join(
        f"{requirement.setting}={requirement.value}"
        for requirement in LARGE_MESSAGE_REQUIREMENTS
    )
    return f"sudo sysctl -w {settings}"


# %% reading a machine


@dataclass(frozen=True)
class MachineSettings:
    """
    Where a machine reports its own settings, so a test can stand somewhere else in.
    """

    sysctl_root: Path = Path("/proc/sys")
    """
    Directory the kernel exposes its settings under.
    """

    network_root: Path = Path("/sys/class/net")
    """
    Directory the kernel exposes its network interfaces under.
    """

    route_table: Path = Path("/proc/net/route")
    """
    File the kernel writes its routing table to.
    """

    def read(self, setting: KernelSetting) -> int:
        """
        Read one setting's current value.

        :param setting: The setting to read.
        """
        return int((self.sysctl_root / str(setting).replace(".", "/")).read_text())

    def default_route_interface(self) -> Optional[str]:
        """
        :return: The interface traffic with no more specific route leaves over, or None
            if this machine has no default route.
        """
        for route in self.route_table.read_text().splitlines()[1:]:
            interface, destination = route.split()[:2]
            if int(destination, 16) == 0:
                return interface
        return None

    def is_wireless(self, interface: str) -> bool:
        """
        Whether an interface is a wireless one.

        :param interface: The interface's name.
        """
        return (self.network_root / interface / "wireless").exists()


THIS_MACHINE = MachineSettings()
"""
The machine this is running on.
"""


# %% the check


@dataclass
class LargeMessagesCannotArrive(DataclassException):
    """
    Raised when a wireless machine's limits are too low to receive a large message,
    which would otherwise show up as a service call that never returns.
    """

    interface: str
    """
    The wireless interface the traffic would arrive over.
    """

    unmet: List[SettingRequirement]
    """
    The requirements this machine does not meet.
    """

    def error_message(self) -> str:
        settings = ", ".join(str(requirement.setting) for requirement in self.unmet)
        return (
            f"Traffic leaves over {self.interface}, which is wireless, and {settings} "
            "would drop a message the size of the world snapshot."
        )

    def suggest_correction(self) -> str:
        return f"Raise them with: {raise_limits_command()}"


def unmet_requirements(
    machine: MachineSettings = THIS_MACHINE,
) -> List[SettingRequirement]:
    """
    The requirements a machine does not currently meet.

    :param machine: The machine to read.
    """
    return [
        requirement
        for requirement in LARGE_MESSAGE_REQUIREMENTS
        if not requirement.is_met(machine.read(requirement.setting))
    ]


def check_large_messages_can_arrive(machine: MachineSettings = THIS_MACHINE) -> None:
    """
    Stop a run that would lose the messages it depends on.

    A machine whose traffic leaves over a wire is left alone: its datagrams arrive whole,
    so the limits never bind.

    :param machine: The machine to read.
    :raises LargeMessagesCannotArrive: If the traffic is wireless and the limits are too
        low.
    """
    interface = machine.default_route_interface()
    if interface is None or not machine.is_wireless(interface):
        return
    unmet = unmet_requirements(machine)
    if not unmet:
        return
    raise LargeMessagesCannotArrive(interface, unmet)
