"""
Fixtures shared by the tests of :mod:`probabilistic_model`.
"""

from __future__ import annotations

import numpy as np
import pytest
from random_events.variable import Continuous, Integer

# %% random state


@pytest.fixture(autouse=True)
def seeded_random_state() -> None:
    """
    Seeds the global random state of :mod:`numpy`, so a test that samples draws the same
    values on every run, whichever tests ran before it.
    """
    np.random.seed(69)


# %% variables


@pytest.fixture
def x() -> Continuous:
    """
    A continuous variable.
    """
    return Continuous("x")


@pytest.fixture
def y() -> Continuous:
    """
    A second continuous variable.
    """
    return Continuous("y")


@pytest.fixture
def z() -> Continuous:
    """
    A third continuous variable.
    """
    return Continuous("z")


@pytest.fixture
def count() -> Integer:
    """
    An integer variable.
    """
    return Integer("n")
