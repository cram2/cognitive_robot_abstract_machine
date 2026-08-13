"""
The debug-marker overlay: what the CRAM system's ROS markers become in the viewer.

CRAM components publish ``visualization_msgs`` markers while a demo runs — collision
closest-points, spatial types, probabilistic costmaps. This module keeps the current
marker set the way RViz would (add/modify, delete, delete-all per topic) and turns
each marker into the plain payload the viewer renders. It is deliberately free of ROS
imports: messages are read duck-typed, so the logic tests without a ROS environment
(see :mod:`cramera.live.ros_markers` for the actual subscriber).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List, Optional, Tuple

from cramera.body_geometry import POSE_PRECISION

ADD_ACTION = 0
"""
``Marker.ADD`` / ``Marker.MODIFY`` — upsert the marker.
"""

DELETE_ACTION = 2
"""
``Marker.DELETE`` — remove one marker.
"""

DELETE_ALL_ACTION = 3
"""
``Marker.DELETEALL`` — clear every marker of the topic.
"""

MARKER_KINDS = {
    0: "arrow",
    1: "cube",
    2: "sphere",
    3: "cylinder",
    4: "line_strip",
    5: "line_list",
    6: "cube_list",
    7: "sphere_list",
    8: "points",
    9: "text",
}
"""
The ``visualization_msgs`` marker types the viewer renders, by their type number.

Mesh-resource and triangle-list markers are not supported and are skipped.
"""


@dataclass(frozen=True)
class MarkerEntry:
    """
    One marker in the form the viewer renders it.
    """

    ns: str
    """
    The marker's namespace, scoping its id like RViz does.
    """

    id: int
    """
    The marker's id within its namespace.
    """

    kind: str
    """
    Which primitive the viewer builds, one of :data:`MARKER_KINDS`'s values.
    """

    frame: str
    """
    The frame the marker's pose is expressed in.
    """

    position: List[float]
    """
    The marker's position in its frame, as ``[x, y, z]``.
    """

    quaternion: List[float]
    """
    The marker's orientation in its frame, as ``[qx, qy, qz, qw]``.
    """

    scale: List[float]
    """
    The marker's scale, with the meaning ``visualization_msgs`` gives it per type.
    """

    color: str
    """
    The marker's colour as a ``#rrggbb`` hex string.
    """

    opacity: float
    """
    The marker's opacity between 0 and 1.
    """

    points: List[List[float]] = field(default_factory=list)
    """
    The marker's point list (line, point and list types), relative to its pose.
    """

    text: str = ""
    """
    The text of a ``text`` marker.
    """

    @classmethod
    def from_message(cls, marker: Any) -> Optional[MarkerEntry]:
        """
        A ``visualization_msgs`` marker as the viewer renders it, read duck-typed.

        :param marker: The marker message.
        :return: The entry, or None for a marker type the viewer does not render.
        """
        kind = MARKER_KINDS.get(int(marker.type))
        if kind is None:
            return None
        position = marker.pose.position
        orientation = marker.pose.orientation
        return cls(
            ns=str(marker.ns),
            id=int(marker.id),
            kind=kind,
            frame=str(marker.header.frame_id),
            position=_rounded([position.x, position.y, position.z]),
            quaternion=_rounded(
                [orientation.x, orientation.y, orientation.z, orientation.w]
            ),
            scale=_rounded([marker.scale.x, marker.scale.y, marker.scale.z]),
            color="#%02x%02x%02x"
            % (
                round(_clamped(marker.color.r) * 255),
                round(_clamped(marker.color.g) * 255),
                round(_clamped(marker.color.b) * 255),
            ),
            opacity=round(_clamped(marker.color.a), 3),
            points=[_rounded([point.x, point.y, point.z]) for point in marker.points],
            text=str(marker.text),
        )


def _rounded(values: List[float]) -> List[float]:
    """
    :param values: The coordinates to round for publication.
    """
    return [round(float(value), POSE_PRECISION) for value in values]


def _clamped(value: float) -> float:
    """
    :param value: A colour channel, clamped into [0, 1].
    """
    return min(max(float(value), 0.0), 1.0)


@dataclass
class MarkerStore:
    """
    The current markers of one topic, maintained the way RViz maintains them.
    """

    entries: Dict[Tuple[str, int], MarkerEntry] = field(default_factory=dict)
    """
    The live markers, keyed by ``(namespace, id)``.
    """

    revision: int = 0
    """
    Bumped whenever the marker set changes, so publishing can be change-driven.
    """

    def observe(self, markers: List[Any]) -> bool:
        """
        Apply one ``MarkerArray``'s worth of messages.

        :param markers: The array's markers, read duck-typed.
        :return: Whether the marker set changed.
        """
        changed = False
        for marker in markers:
            action = int(marker.action)
            if action == DELETE_ALL_ACTION:
                changed = changed or bool(self.entries)
                self.entries.clear()
                continue
            key = (str(marker.ns), int(marker.id))
            if action == DELETE_ACTION:
                changed = self.entries.pop(key, None) is not None or changed
                continue
            if action != ADD_ACTION:
                continue
            entry = MarkerEntry.from_message(marker)
            if entry is None:
                continue
            if self.entries.get(key) != entry:
                self.entries[key] = entry
                changed = True
        if changed:
            self.revision += 1
        return changed
