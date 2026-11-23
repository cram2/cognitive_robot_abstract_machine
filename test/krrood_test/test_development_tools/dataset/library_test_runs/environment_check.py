"""
A test that passes only where the variables the layout next to it declares for the
library ``base`` are set.

Named so the suite does not collect it.
"""

from __future__ import annotations

import os
from pathlib import Path

from krrood.development_tools.continuous_integration_jobs import (
    ContinuousIntegrationJobLayout,
)


def test_environment_of_base_is_set() -> None:
    """
    Every variable the tests of ``base`` need is set to its value.
    """
    layout = ContinuousIntegrationJobLayout.from_toml(
        text=(Path(__file__).parent / "continuous_integration_jobs.toml").read_text()
    )
    environment = layout.test_environments["base"]
    assert {name: os.environ.get(name) for name in environment} == environment
