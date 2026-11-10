"""
Fixtures for the tests of a live session: cell scripts, and a session driving a
simulated pendulum.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
from typing_extensions import Any, Iterator

from experiments.live_session.session import LiveSession
from semantic_digital_twin.adapters.multi_sim import MujocoSim
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import RevoluteConnection

from ...pytest_environment import runs_in_continuous_integration
from ...semantic_digital_twin_test.test_adapters.test_mujoco_servos import (
    _servoed_pendulum_world,
)

# %% cell scripts


@pytest.fixture
def cell_scripts() -> Path:
    """
    The directory holding the cell scripts the tests perform.
    """
    return Path(__file__).parent / "cell_scripts"


@pytest.fixture
def swing_the_hinge_script(cell_scripts: Path) -> Path:
    """
    A script of two cells, each swinging the pendulum's hinge to another set point.
    """
    return cell_scripts / "swing_the_hinge.py"


@pytest.fixture
def divide_by_zero_script(cell_scripts: Path) -> Path:
    """
    A script of one cell that raises after its first report.
    """
    return cell_scripts / "divide_by_zero.py"


# %% a session driving a simulated pendulum


@dataclass(kw_only=True)
class PendulumSession:
    """
    A live session driving a simulated pendulum, and what its cells refer to.
    """

    session: LiveSession
    """
    The session under test.
    """

    world: World
    """
    The pendulum's world.
    """

    hinge: RevoluteConnection
    """
    The pendulum's servoed hinge.
    """

    namespace: dict[str, Any]
    """
    The names the cells are performed with.
    """

    @property
    def hinge_position(self) -> float:
        """
        Where the physics has the hinge now.
        """
        return self.session.simulation.simulator.get_joint_value("hinge").result


@pytest.fixture
def pendulum_session() -> Iterator[PendulumSession]:
    """
    A session over a stepped simulation of a servoed pendulum, its arm as the tool.
    """
    if not runs_in_continuous_integration():
        pytest.skip("MuJoCo tests only run in CI")
    pendulum = _servoed_pendulum_world()
    world = pendulum.world
    simulation = MujocoSim(world=world, headless=True)
    simulation.start_stepped_simulation()
    session = LiveSession(
        simulation=simulation,
        tool_frame=pendulum.hinge.child,
        robot_base=pendulum.hinge.parent,
    )
    namespace = {
        "session": session,
        "simulation": simulation,
        "world": world,
        "hinge": pendulum.hinge,
        "timedelta": timedelta,
    }
    yield PendulumSession(
        session=session, world=world, hinge=pendulum.hinge, namespace=namespace
    )
    simulation.stop_simulation()
