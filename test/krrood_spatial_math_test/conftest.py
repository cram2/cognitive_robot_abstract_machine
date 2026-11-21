"""
Fixtures the krrood_spatial_math tests share: Cartesian frames, the space they chart,
and the code of the README's example of adding a space.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from .dataset.frames import NamedFrame, Room

# %% Cartesian frames


@pytest.fixture
def table_frame() -> NamedFrame:
    return NamedFrame("table")


@pytest.fixture
def camera_frame() -> NamedFrame:
    return NamedFrame("camera")


@pytest.fixture
def gripper_frame() -> NamedFrame:
    return NamedFrame("gripper")


# %% the space frames chart


@pytest.fixture
def room() -> Room:
    return Room()


# %% the README's example of adding a space


@pytest.fixture(scope="session")
def adding_a_space_example_code() -> str:
    """
    :return: The code blocks of the README's *Adding a space* section, joined in order.
    """
    readme = (
        Path(__file__).parents[2]
        / "domain_models"
        / "krrood_spatial_math"
        / "README.md"
    ).read_text()
    section = readme.split("## Adding a space", maxsplit=1)[1]
    return "\n".join(re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL))
