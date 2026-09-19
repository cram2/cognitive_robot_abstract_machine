"""
Tests for the check that this machine can receive the large messages the robot sends.
"""

from __future__ import annotations

import pytest

from experiments.network_limits import (
    AtLeast,
    AtMost,
    KernelSetting,
    LargeMessagesCannotArrive,
    MachineSettings,
    check_large_messages_can_arrive,
    raise_limits_command,
    unmet_requirements,
)

# %% fixtures

WIRED_ROUTE = (
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\n"
    "eth0\t00000000\t0100A8C0\t0003\t0\t0\t100\t00000000\n"
)
"""
A routing table whose default route leaves over a wired interface.
"""

WIRELESS_ROUTE = (
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\n"
    "wlan0\t0064A8C0\t00000000\t0001\t0\t0\t600\t00FCFFFF\n"
    "wlan0\t00000000\t0100A8C0\t0003\t0\t0\t600\t00000000\n"
)
"""
A routing table whose default route leaves over a wireless interface, with a subnet
route listed first so the default is not simply the first row.
"""


def build_machine(
    tmp_path, route_table: str, wireless: bool, **settings
) -> MachineSettings:
    """
    Build a stand-in for a machine's own ``/proc`` and ``/sys``.

    :param tmp_path: Directory to lay the files out under.
    :param route_table: Contents of the routing table.
    :param wireless: Whether the interface it names is wireless.
    :param settings: The value to report for each :class:`KernelSetting`, by member
        name.
    :return: The stand-in.
    """
    interface = route_table.strip().splitlines()[-1].split("\t")[0]
    routes = tmp_path / "route"
    routes.write_text(route_table)
    network_root = tmp_path / "net"
    (network_root / interface).mkdir(parents=True)
    if wireless:
        (network_root / interface / "wireless").mkdir()
    sysctl_root = tmp_path / "sys"
    for setting in KernelSetting:
        path = sysctl_root / str(setting).replace(".", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{settings[setting.name]}\n")
    return MachineSettings(
        sysctl_root=sysctl_root, network_root=network_root, route_table=routes
    )


TUNED = {
    "FRAGMENT_CACHE_CEILING": 268435456,
    "FRAGMENT_CACHE_FLOOR": 201326592,
    "FRAGMENT_LIFETIME": 3,
    "RECEIVE_BUFFER_CEILING": 67108864,
    "RECEIVE_BUFFER_DEFAULT": 67108864,
    "DEVICE_BACKLOG": 30000,
}
"""
Values that meet every requirement.
"""

UNTUNED = {**TUNED, "FRAGMENT_CACHE_CEILING": 4194304, "FRAGMENT_LIFETIME": 30}
"""
The stock values that were measured to lose 99.99% of a large message.
"""


# %% requirements


def test_a_floor_is_met_by_the_value_it_names_and_by_more():
    requirement = AtLeast(KernelSetting.DEVICE_BACKLOG, 1000)

    assert requirement.is_met(1000)
    assert requirement.is_met(1001)
    assert not requirement.is_met(999)


def test_a_ceiling_is_met_by_the_value_it_names_and_by_less():
    requirement = AtMost(KernelSetting.FRAGMENT_LIFETIME, 3)

    assert requirement.is_met(3)
    assert requirement.is_met(2)
    assert not requirement.is_met(4)


# %% reading the machine


def test_the_default_route_names_the_interface_it_leaves_over(tmp_path):
    machine = build_machine(tmp_path, WIRELESS_ROUTE, wireless=True, **TUNED)

    assert machine.default_route_interface() == "wlan0"


def test_an_interface_with_no_wireless_entry_is_wired(tmp_path):
    machine = build_machine(tmp_path, WIRED_ROUTE, wireless=False, **TUNED)

    assert not machine.is_wireless("eth0")


def test_a_setting_is_read_as_the_number_the_kernel_reports(tmp_path):
    machine = build_machine(tmp_path, WIRED_ROUTE, wireless=False, **TUNED)

    assert machine.read(KernelSetting.FRAGMENT_LIFETIME) == TUNED["FRAGMENT_LIFETIME"]


# %% the check


def test_stock_limits_are_reported_as_unmet(tmp_path):
    machine = build_machine(tmp_path, WIRELESS_ROUTE, wireless=True, **UNTUNED)

    unmet = {requirement.setting for requirement in unmet_requirements(machine)}

    assert unmet == {
        KernelSetting.FRAGMENT_CACHE_CEILING,
        KernelSetting.FRAGMENT_LIFETIME,
    }


def test_tuned_limits_leave_nothing_unmet(tmp_path):
    machine = build_machine(tmp_path, WIRELESS_ROUTE, wireless=True, **TUNED)

    assert unmet_requirements(machine) == []


def test_a_wireless_machine_with_stock_limits_is_refused(tmp_path):
    machine = build_machine(tmp_path, WIRELESS_ROUTE, wireless=True, **UNTUNED)

    with pytest.raises(LargeMessagesCannotArrive):
        check_large_messages_can_arrive(machine)


def test_a_wired_machine_is_left_alone_whatever_its_limits(tmp_path):
    machine = build_machine(tmp_path, WIRED_ROUTE, wireless=False, **UNTUNED)

    check_large_messages_can_arrive(machine)


def test_a_tuned_wireless_machine_is_allowed_through(tmp_path):
    machine = build_machine(tmp_path, WIRELESS_ROUTE, wireless=True, **TUNED)

    check_large_messages_can_arrive(machine)


def test_the_refusal_names_the_command_that_raises_the_limits(tmp_path):
    machine = build_machine(tmp_path, WIRELESS_ROUTE, wireless=True, **UNTUNED)

    with pytest.raises(LargeMessagesCannotArrive) as refusal:
        check_large_messages_can_arrive(machine)

    assert raise_limits_command() in str(refusal.value)


def test_the_command_sets_every_requirement_it_checks():
    command = raise_limits_command()

    for setting in KernelSetting:
        assert str(setting) in command
