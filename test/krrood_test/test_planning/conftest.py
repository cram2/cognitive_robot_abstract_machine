"""
Fixtures shared by the planning tests.
"""

from __future__ import annotations

import pytest

from ..dataset.stacking_blocks import StackingWorld

# %% worlds


@pytest.fixture
def stacking_world() -> StackingWorld:
    """
    :return: A fresh :class:`~test.krrood_test.dataset.stacking_blocks.StackingWorld` for
        each test.
    """
    return StackingWorld.build()
