"""
Configuration of the goal lifecycle of the Giskard motion server.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GiskardServerConfig:
    """
    Configures the goal lifecycle of the Giskard motion server.
    """

    idle_frequency: float = 20.0
    """
    Frequency in hertz at which Giskard waits for goals.
    """
