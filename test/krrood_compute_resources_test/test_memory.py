"""
Tests of the memory of the host and of a budgeted share of a device.
"""

from __future__ import annotations

# %% imports
import os

import numpy as np

from krrood_compute_resources.memory import BudgetedMemory, HostMemory

from .dataset.memory import FixedFreeMemory

# %% memory of the host


def test_the_free_host_memory_is_at_most_the_physical_memory():
    """
    The host reports some memory free, and never more than it has.
    """
    physical = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")

    assert 0 < HostMemory().free_bytes() <= physical


def test_the_peak_host_memory_records_an_allocation():
    """
    An array allocated after the peak was reset raises the peak.
    """
    memory = HostMemory()
    memory.reset_peak()
    before = memory.peak_bytes()

    array = np.ones(2**23)

    assert memory.peak_bytes() > before


# %% budgeted memory


def test_a_budget_below_the_free_memory_caps_it():
    """
    A budget smaller than the free memory is all a process may allocate.
    """
    memory = BudgetedMemory(memory=FixedFreeMemory(free=1000), budget_bytes=300)

    assert memory.free_bytes() == memory.budget_bytes


def test_a_budget_above_the_free_memory_leaves_it():
    """
    A budget larger than the free memory does not raise what may be allocated.
    """
    device = FixedFreeMemory(free=200)

    assert BudgetedMemory(memory=device, budget_bytes=300).free_bytes() == device.free
