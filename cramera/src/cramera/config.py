"""
Configuration shared by a live Cramera visualization session.
"""

from dataclasses import dataclass


# %% live visualization
@dataclass(frozen=True)
class CrameraConfig:
    """
    Configure body publication for one live session.
    """

    robot_base_key: str = "__base__"
    """
    Publication key reserved for the robot root instead of a loose object.
    """

    rebind_interval_seconds: float = 3.0
    """
    Time between periodic discoveries of the world's bodies and connections.
    """
