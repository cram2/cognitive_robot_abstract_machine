"""
The live transform graph: every connection of the executing world, with the moment its
transform last changed and who wrote it.

This is what a TF tree view is for a world that has no TF: a frame is fresh exactly when
the connection carrying it moved recently, and stale when nothing has written it for a
while. Unlike TF the graph is typed — a fixed connection can never go stale, and a pose
the viewer dragged is told apart from one the demo drove.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import StrEnum

from typing_extensions import (
    Any,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
)

from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
    FixedConnection,
)

from cramera.body_geometry import POSE_PRECISION

if TYPE_CHECKING:
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Connection


# %% vocabulary
class ConnectionKind(StrEnum):
    """
    How a connection holds its child frame, which decides whether it can move at all.
    """

    FIXED = "fixed"
    ACTUATED = "actuated"
    FREE = "free"
    OTHER = "other"

    @classmethod
    def of_connection(cls, connection: Connection) -> ConnectionKind:
        """
        The kind one world connection belongs to.

        :param connection: The connection to classify.
        """
        if isinstance(connection, FixedConnection):
            return cls.FIXED
        if isinstance(connection, ActiveConnection1DOF):
            return cls.ACTUATED
        if isinstance(connection, Connection6DoF):
            return cls.FREE
        return cls.OTHER


class TransformWriter(StrEnum):
    """
    Who last wrote a connection's transform.
    """

    DEMO = "demo"
    VIEWER = "viewer"
    NOBODY = "nobody"


class TransformFreshness(StrEnum):
    """
    How recently a connection's transform changed.

    The values double as the node-status keys the graph panel draws its rings from, so
    they are spelled the way the panel's other status vocabularies are.
    """

    MOVING = "MOVING"
    SETTLED = "SETTLED"
    STALE = "STALE"
    STATIC = "STATIC"


# %% one connection's activity
@dataclass(frozen=True)
class ConnectionActivity:
    """
    One connection of the world, and when it last moved.
    """

    MOVING_SECONDS: ClassVar[float] = 0.5
    """
    How long after a change a connection still reads as moving.
    """

    SETTLED_SECONDS: ClassVar[float] = 5.0
    """
    How long after a change a connection still reads as recently settled.
    """

    name: str
    """
    The connection's own prefixed name.
    """

    parent: str
    """
    Name of the frame the connection hangs from.
    """

    child: str
    """
    Name of the frame the connection carries.
    """

    kind: ConnectionKind
    """
    How the connection holds its child frame.
    """

    values: Tuple[float, ...]
    """
    The connection's degree-of-freedom positions as last observed; empty when it has
    none and therefore cannot move.
    """

    changed_at: Optional[float] = None
    """
    Timestamp of the last observed change, or None while nothing has written it.
    """

    writer: TransformWriter = TransformWriter.NOBODY
    """
    Who wrote the change :attr:`changed_at` records.
    """

    def structure(self) -> Tuple[str, str, str, str]:
        """
        The part of this entry that a signature covers: everything but its activity.
        """
        return self.name, self.parent, self.child, str(self.kind)

    def freshness(self, now: float) -> TransformFreshness:
        """
        How this connection reads at ``now``.

        :param now: The timestamp to age the last change against.
        """
        if not self.values:
            return TransformFreshness.STATIC
        if self.changed_at is None:
            return TransformFreshness.STALE
        age = now - self.changed_at
        if age <= self.MOVING_SECONDS:
            return TransformFreshness.MOVING
        if age <= self.SETTLED_SECONDS:
            return TransformFreshness.SETTLED
        return TransformFreshness.STALE

    def to_payload(self, now: float) -> Dict[str, Any]:
        """
        The entry in the JSON shape the viewer reads.

        :param now: The timestamp ages are computed against.
        """
        return {
            "name": self.name,
            "parent": self.parent,
            "child": self.child,
            "kind": str(self.kind),
            "writer": str(self.writer),
            "freshness": str(self.freshness(now)),
            "ageSeconds": (
                None
                if self.changed_at is None
                else round(now - self.changed_at, POSE_PRECISION)
            ),
        }


@dataclass(frozen=True)
class TransformSnapshot:
    """
    The world's whole connection graph at one simulation tick.
    """

    signature: str = ""
    """
    Digest of the graph's structure; the viewer rebuilds only when it changes.
    """

    activities: Tuple[ConnectionActivity, ...] = ()
    """
    Every connection of the world, in a stable order.
    """

    def to_payload(self, now: float) -> Dict[str, Any]:
        """
        The snapshot in the JSON shape the viewer reads.

        :param now: The timestamp ages are computed against.
        """
        return {
            "signature": self.signature,
            "connections": [activity.to_payload(now) for activity in self.activities],
        }


# %% the tracker
@dataclass
class TransformGraph:
    """
    Tracks, across simulation ticks, when each connection of the world last changed.

    Lives on the simulation thread: :meth:`observe` reads the world, and the frozen
    snapshot it returns is what the HTTP layer serves.
    """

    activities: Dict[str, ConnectionActivity] = field(default_factory=dict)
    """
    The newest entry per connection name.
    """

    _viewer_written: Set[str] = field(default_factory=set)
    """
    Connections a viewer drag wrote since the last observation.
    """

    def note_viewer_write(self, connection_name: str) -> None:
        """
        Record that a viewer drag, not the demo, wrote this connection.

        :param connection_name: Name of the connection the drag was applied to.
        """
        self._viewer_written.add(connection_name)

    def observe(
        self, connections: Iterable[Connection], world: World, now: float
    ) -> TransformSnapshot:
        """
        Read every connection's degrees of freedom and stamp the ones that changed.

        :param connections: The world's connections, as the bridge discovered them.
        :param world: The world holding the degree-of-freedom state.
        :param now: The timestamp a change observed here is stamped with.
        """
        written, self._viewer_written = self._viewer_written, set()
        observed: Dict[str, ConnectionActivity] = {}
        for connection in connections:
            name = str(connection.name)
            values = self._values(connection, world)
            previous = self.activities.get(name)
            if previous is None:
                observed[name] = ConnectionActivity(
                    name=name,
                    parent=str(connection.parent.name),
                    child=str(connection.child.name),
                    kind=ConnectionKind.of_connection(connection),
                    values=values,
                )
                continue
            if values == previous.values and name not in written:
                observed[name] = previous
                continue
            observed[name] = replace(
                previous,
                values=values,
                changed_at=now,
                writer=(
                    TransformWriter.VIEWER if name in written else TransformWriter.DEMO
                ),
            )
        self.activities = observed
        return self._snapshot()

    @staticmethod
    def _values(connection: Connection, world: World) -> Tuple[float, ...]:
        """
        A connection's degree-of-freedom positions, rounded to the precision the rest of
        the bridge publishes poses at, so numerical noise does not read as motion.

        :param connection: The connection to read.
        :param world: The world holding the degree-of-freedom state.
        """
        return tuple(
            round(world.state[degree_of_freedom.id].position, POSE_PRECISION)
            for degree_of_freedom in connection.dofs
        )

    def _snapshot(self) -> TransformSnapshot:
        """
        The current entries as the frozen snapshot the HTTP layer serves.
        """
        ordered: List[ConnectionActivity] = sorted(
            self.activities.values(), key=lambda activity: activity.name
        )
        structure = json.dumps([activity.structure() for activity in ordered])
        return TransformSnapshot(
            signature=hashlib.sha1(structure.encode()).hexdigest(),
            activities=tuple(ordered),
        )
