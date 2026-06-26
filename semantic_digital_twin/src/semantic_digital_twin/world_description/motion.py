"""
Motion type for the Body Motion Problem framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from semantic_digital_twin.world_description.world_entity import Connection

if TYPE_CHECKING:
    from semantic_digital_twin.physics.physics_model import PhysicsModel


@dataclass
class MotionTrajectory:
    """
    Maps each tracked connection to its recorded position sequence.

    All connections produced by a single physics simulation are stored here
    in lock-step, so they can be replayed together without any distinction
    between roles. Which connection is the robot's actuator is known by the
    caller (via :attr:`~semantic_digital_twin.world_description.motion.Motion.actuator`),
    not by this class.
    """

    data: dict[Connection, list[float]] = field(default_factory=dict)
    """Position sequence for each connection, indexed by simulation step."""

    def is_empty(self) -> bool:
        """
        :return: ``True`` when no positions have been recorded for any connection.
        """
        return not any(self.data.values())

    def position_updates_at(self, step: int) -> dict[Connection, float]:
        """
        :return: The position of every tracked connection at the given step index.
        """
        return {conn: traj[step] for conn, traj in self.data.items()}

    def positions_for(self, connection: Connection) -> list[float]:
        """
        :return: All recorded positions for the given connection, or an empty list
                 if the connection was not tracked.
        """
        return self.data.get(connection, [])


@dataclass(eq=True)
class Motion:
    """
    Represents a candidate motion as a sequence of connection positions.

    When a physics model is provided and the trajectory is ``None``, the model
    is used to generate the trajectory on demand during the causal sufficiency check.
    """

    connection: Connection
    """The connection (joint) that is manipulated by this motion."""

    motion_model: Optional[PhysicsModel] = field(default=None)
    """Optional physics model used to generate the trajectory when it is absent."""

    motion_trajectory: Optional[MotionTrajectory] = field(default=None)
    """
    The recorded position sequences for all tracked connections.

    ``None`` means the trajectory has not yet been computed. Set automatically
    by :meth:`~semantic_digital_twin.reasoning.bmp_predicates.Causes` when a
    :attr:`motion_model` is present, or supplied directly for pre-computed motions.
    """

    time_step: Optional[float] = field(default=None)
    """
    Time step between trajectory samples in seconds.

    When set, :meth:`~semantic_digital_twin.reasoning.bmp_predicates.Causes` steps
    all HasUpdateState connections after each position update, allowing coupled
    physics (e.g. fill-level ODEs) to be inferred from the primary trajectory
    without explicit secondary trajectories.
    """
