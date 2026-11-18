"""
Tests of the division of a machine into shares.
"""

from __future__ import annotations

# %% imports
import pytest

from krrood_compute_resources.devices import Device
from krrood_compute_resources.exceptions import (
    EmptyMachineDivisionError,
    MissingGraphicsProcessorError,
)
from krrood_compute_resources.machine import (
    GraphicsProcessorVariable,
    Machine,
    MachineDivision,
    ThreadVariable,
)

from .dataset.memory import FixedFreeMemory

# %% fixtures


@pytest.fixture
def machine_with_graphics_processor() -> Machine:
    """
    :return: A machine of eight cores with free host and GPU memory of fixed size.
    """
    return Machine(
        processor_cores=set(range(8)),
        host_memory=FixedFreeMemory(free=1000),
        graphics_processor_memory=FixedFreeMemory(free=600),
    )


@pytest.fixture
def division() -> MachineDivision:
    """
    :return: A division into one share of the GPU and three of the processor cores.
    """
    return MachineDivision(graphics_processor_shares=1, processor_core_shares=3)


# %% shares


def test_shares_of_the_graphics_processor_come_first(
    machine_with_graphics_processor: Machine, division: MachineDivision
):
    """
    The shares of the GPU lead, followed by the shares of the processor cores.
    """
    shares = division.shares(machine=machine_with_graphics_processor)

    assert [share.device for share in shares] == [
        Device.GRAPHICS_PROCESSOR,
        Device.CENTRAL_PROCESSOR,
        Device.CENTRAL_PROCESSOR,
        Device.CENTRAL_PROCESSOR,
    ]


def test_every_share_has_cores_of_its_own(
    machine_with_graphics_processor: Machine, division: MachineDivision
):
    """
    The cores are divided equally, and no core belongs to two shares.
    """
    shares = division.shares(machine=machine_with_graphics_processor)

    assert [sorted(share.processor_cores) for share in shares] == [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
    ]


def test_memory_budgets_divide_the_free_memory_of_each_device(
    machine_with_graphics_processor: Machine, division: MachineDivision
):
    """
    A GPU share's budget divides the free GPU memory among the GPU shares, a CPU share's
    budget divides the free host memory among all shares.
    """
    shares = division.shares(machine=machine_with_graphics_processor)
    host_share = int(
        division.host_memory_fraction
        * machine_with_graphics_processor.host_memory.free_bytes()
        / len(shares)
    )
    graphics_processor_share = int(
        division.graphics_processor_memory_fraction
        * machine_with_graphics_processor.graphics_processor_memory.free_bytes()
    )

    assert [share.memory_budget_bytes for share in shares] == [
        graphics_processor_share,
        host_share,
        host_share,
        host_share,
    ]
    assert {share.host_memory_limit_bytes for share in shares} == {
        int(division.host_memory_limit_shares * host_share)
    }


def test_shares_of_a_missing_graphics_processor_fail(division: MachineDivision):
    """
    Shares of the GPU cannot be made of a machine without one.
    """
    machine = Machine(processor_cores={0}, host_memory=FixedFreeMemory(free=1000))

    with pytest.raises(MissingGraphicsProcessorError):
        division.shares(machine=machine)


def test_a_division_without_shares_fails():
    """
    A division needs at least one share.
    """
    with pytest.raises(EmptyMachineDivisionError):
        MachineDivision(graphics_processor_shares=0, processor_core_shares=0)


def test_a_division_shares_only_devices_it_has_shares_of():
    """
    A division without shares of the GPU computes on the processor cores alone.
    """
    division = MachineDivision(graphics_processor_shares=0)

    assert [division.shares_device(device=device) for device in Device] == [
        True,
        False,
    ]


# %% environment of a process


def test_a_processor_core_share_hides_the_graphics_processor(
    machine_with_graphics_processor: Machine, division: MachineDivision
):
    """
    A process on a share of the processor cores sees no GPU and caps its threads at the
    cores of its share.
    """
    share = division.shares(machine=machine_with_graphics_processor)[-1]

    environment = share.process_environment(environment={})

    assert environment == {
        **{variable: str(len(share.processor_cores)) for variable in ThreadVariable},
        GraphicsProcessorVariable.VISIBLE_DEVICES: "",
    }


def test_a_graphics_processor_share_does_not_preallocate(
    machine_with_graphics_processor: Machine, division: MachineDivision
):
    """
    A process on a share of the GPU keeps jax from claiming the memory of the other
    shares.
    """
    share = division.shares(machine=machine_with_graphics_processor)[0]

    environment = share.process_environment(environment={})

    assert environment[GraphicsProcessorVariable.JAX_MEMORY_PREALLOCATION] == "false"
    assert GraphicsProcessorVariable.VISIBLE_DEVICES not in environment
