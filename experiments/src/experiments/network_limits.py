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

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import TYPE_CHECKING

from krrood.entity_query_language.factories import and_, an, entity, or_, variable
from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from krrood.entity_query_language.query.query import Entity

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
class SettingValue:
    """
    A kernel setting at one value.
    """

    setting: KernelSetting
    """
    The setting.
    """

    value: int
    """
    The value it is at.
    """


# %% reading a machine


@dataclass(frozen=True)
class MachineSettings:
    """
    Where a machine reports its own settings, so a test can stand somewhere else in.
    """

    kernel_settings_root: Path = Path("/proc/sys")
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
        return int(
            (self.kernel_settings_root / str(setting).replace(".", "/")).read_text()
        )

    def setting_values(self) -> list[SettingValue]:
        """
        :return: The current value of every setting a large message depends on.
        """
        return [SettingValue(setting, self.read(setting)) for setting in KernelSetting]

    def default_route_interface(self) -> str | None:
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

    unmet: list[SettingValue]
    """
    The required values this machine does not meet.
    """

    raise_limits_command: str
    """
    The command that puts the machine's limits where they need to be.
    """

    def error_message(self) -> str:
        settings = ", ".join(str(requirement.setting) for requirement in self.unmet)
        return (
            f"Traffic leaves over {self.interface}, which is wireless, and {settings} "
            "would drop a message the size of the world snapshot."
        )

    def suggest_correction(self) -> str:
        return f"Raise them with: {self.raise_limits_command}"


@dataclass
class LargeMessageReceiver:
    """
    A machine that has to receive the large messages the robot sends it, with the kernel
    settings that takes.
    """

    machine: MachineSettings = field(default_factory=MachineSettings)
    """
    The machine the messages arrive at.
    """

    requirements: tuple[SettingValue, ...] = (
        SettingValue(KernelSetting.FRAGMENT_CACHE_CEILING, 268435456),
        SettingValue(KernelSetting.FRAGMENT_CACHE_FLOOR, 201326592),
        SettingValue(KernelSetting.FRAGMENT_LIFETIME, 3),
        SettingValue(KernelSetting.RECEIVE_BUFFER_CEILING, 67108864),
        SettingValue(KernelSetting.RECEIVE_BUFFER_DEFAULT, 67108864),
        SettingValue(KernelSetting.DEVICE_BACKLOG, 30000),
    )
    """
    The configuration measured to carry the world snapshot over this setup's wireless
    link. The fragment lifetime has to be at most its value, every other setting at
    least its value.

    These were raised and tested together, so this is the set known to work rather than
    the smallest set that would.
    """

    def raise_limits_command(self) -> str:
        """
        :return: The command that puts the machine's limits where they need to be, until
            it is next rebooted.
        """
        settings = " ".join(
            f"{requirement.setting}={requirement.value}"
            for requirement in self.requirements
        )
        return f"sudo sysctl -w {settings}"

    def unmet_requirements(self) -> Entity[SettingValue]:
        """
        :return: The query for the requirements the machine does not currently meet.
        """
        actual = variable(SettingValue, domain=self.machine.setting_values())
        required = variable(SettingValue, domain=self.requirements)
        return an(
            entity(required).where(
                actual.setting == required.setting,
                or_(
                    and_(
                        required.setting != KernelSetting.FRAGMENT_LIFETIME,
                        actual.value < required.value,
                    ),
                    and_(
                        required.setting == KernelSetting.FRAGMENT_LIFETIME,
                        actual.value > required.value,
                    ),
                ),
            )
        )

    def check_messages_can_arrive(self) -> None:
        """
        Stop a run that would lose the messages it depends on.

        A machine whose traffic leaves over a wire is left alone: its datagrams arrive
        whole, so the limits never bind.

        :raises LargeMessagesCannotArrive: If the traffic is wireless and the limits are
            too low.
        """
        interface = self.machine.default_route_interface()
        if interface is None or not self.machine.is_wireless(interface):
            return
        unmet = self.unmet_requirements().tolist()
        if not unmet:
            return
        raise LargeMessagesCannotArrive(interface, unmet, self.raise_limits_command())
