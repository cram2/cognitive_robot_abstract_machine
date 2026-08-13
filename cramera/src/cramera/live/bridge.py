"""
The live-viz bridge state: what a running demo publishes to the viewer.

This module is free of HTTP — it holds the :class:`Bridge` singleton whose snapshot
methods run on the *simulation* thread (driven by the world's own callbacks and the
plan callbacks of :mod:`cramera.live.visualization`) and whose ``get_*`` accessors
hand finished, plain-dict snapshots to the HTTP layer.

Node status is where the plan and the statechart differ: coraplex only performs the plan
root (``Plan.perform`` → ``root.perform``); ``ActionNode.notify`` expands its children
but never performs them, so every inner ``PlanNode`` keeps status ``CREATED`` for the
whole run. The real per-step progress arrives through the plan callbacks, which report
each motion node's start and end; those statuses are propagated up the plan tree and
flagged ``derived``.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.parse
import math
from dataclasses import asdict, dataclass, field
from enum import Enum, StrEnum
from http.server import ThreadingHTTPServer
from pathlib import Path

from typing_extensions import (
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Tuple,
    TYPE_CHECKING,
)

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Quaternion,
    RotationMatrix,
)
from cramera.logging_setup import get_logger
from cramera.body_geometry import POSE_PRECISION, rounded_pose
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
    Connection6DoF,
)

from cramera.knowledge.enums import PlanNodeGroup
from cramera.live.markers import MarkerEntry, MarkerStore
from cramera.live.shape_catalog import ShapeEntry, served_mesh_file, shape_entry
from cramera.mesh_format import MeshFormat
from cramera.palette import ObjectPalette
from cramera.robot_parts import RobotPartAnnotation

if TYPE_CHECKING:
    from coraplex.plans.plan import Plan
    from coraplex.plans.plan_node import MotionNode, PlanNode
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart
    from semantic_digital_twin.world import World
    from semantic_digital_twin.world_description.world_entity import Body

logger = get_logger(__name__)


class TaskStatusName(StrEnum):
    """
    The status vocabulary the viewer styles plan and statechart nodes with.

    Mirrors coraplex's ``TaskStatus`` names. A :class:`StrEnum`, because the values
    travel to the frontend as JSON and are compared against the names coraplex itself
    reports.
    """

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    PAUSE = "PAUSE"

    @classmethod
    def _precedence(cls) -> Tuple[TaskStatusName, ...]:
        """
        The statuses from lowest to highest precedence.
        """
        return (
            cls.CREATED,
            cls.SUCCEEDED,
            cls.PAUSE,
            cls.RUNNING,
            cls.INTERRUPTED,
            cls.FAILED,
        )

    @property
    def rank(self) -> int:
        """
        Precedence when a plan node's status is aggregated from its children: the higher
        rank wins.
        """
        return self._precedence().index(self)

    @classmethod
    def rank_of(cls, status: str) -> int:
        """
        The rank of a status name, or the lowest rank for one this enum does not know.

        :param status: A status name as reported by coraplex or the statechart.
        """
        if status not in cls._value2member_map_:
            return 0
        return cls(status).rank


ROBOT_BASE_KEY = "__base__"
"""
Key under which the robot's root body is published, instead of as a loose object.
"""


@runtime_checkable
class DescribesAnAction(Protocol):
    """
    A plan node carrying the designator that describes what it does.

    Structural, because only some coraplex node types have a designator at all.
    """

    designator: Any


@runtime_checkable
class NamesAWorldEntity(Protocol):
    """
    Anything carrying a world-entity name, such as a body a designator refers to.
    """

    name: Any


class MalformedMoveRequest(Exception):
    """
    Raised when the viewer's move payload cannot be read as a pose.
    """


@dataclass(frozen=True)
class MoveRequest:
    """
    A drag the viewer performed on one object, to be written into the world.
    """

    object_key: str
    """
    Mesh key of the dragged object, as published in the geometry catalog.
    """

    position: List[float]
    """
    Target position ``[x, y, z]`` in world coordinates.
    """

    quaternion: Optional[List[float]] = None
    """
    Target orientation ``[qx, qy, qz, qw]``, or None to keep the current one.
    """

    is_final: bool = False
    """
    Whether this is the drag's last update, as opposed to an intermediate one.
    """

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> MoveRequest:
        """
        Build a request from a decoded ``POST /move`` body.

        :param payload: The decoded JSON body of a ``POST /move`` request.
        :raises MalformedMoveRequest: If the object key or the position is unusable.
            Validating here keeps bad input from raising inside the simulation tick,
            where the only recovery is to drop the whole snapshot.
        """
        object_key = payload.get("object")
        if not isinstance(object_key, str) or not object_key:
            raise MalformedMoveRequest("'object' must be a non-empty string")
        position = cls._coordinates(payload.get("position"), "position", 3)
        quaternion = (
            cls._coordinates(payload.get("quaternion"), "quaternion", 4)
            if payload.get("quaternion")
            else None
        )
        return cls(
            object_key=object_key,
            position=position,
            quaternion=quaternion,
            is_final=bool(payload.get("final")),
        )

    @staticmethod
    def _coordinates(value: Any, name: str, length: int) -> List[float]:
        """
        Read a fixed-length list of finite coordinates out of a payload field.

        :param value: The raw payload field to validate and convert.
        :param name: The field's name, used in the error message if it is invalid.
        :param length: The number of coordinates the field must contain.
        """
        if not isinstance(value, (list, tuple)) or len(value) != length:
            raise MalformedMoveRequest(
                "'%s' must be a list of %d numbers" % (name, length)
            )
        coordinates = []
        for entry in value:
            if isinstance(entry, bool) or not isinstance(entry, (int, float)):
                raise MalformedMoveRequest("'%s' must contain only numbers" % name)
            if not math.isfinite(entry):
                raise MalformedMoveRequest("'%s' must be finite" % name)
            coordinates.append(float(entry))
        return coordinates


@dataclass
class MotionNodeProgress:
    """
    What the bridge knows about one plan node's execution.

    Holds the node itself so the identity key derived from it stays unique for as long
    as the entry lives.
    """

    node: PlanNode
    """
    The plan node this progress belongs to.
    """

    status: Optional[TaskStatusName] = None
    """
    The node's last observed execution status, else None.
    """


# %% viewer payload shapes
class ObjectKind(StrEnum):
    """
    How a loose object's geometry is served to the viewer.
    """

    MESH = "mesh"
    BOX = "box"
    SHAPES = "shapes"


@dataclass(frozen=True)
class ObjectCatalogEntry:
    """
    One loose object's geometry-catalog entry, as the viewer spawns it.
    """

    key: str
    """
    Mesh basename this object is published under.
    """

    id: str
    """
    Stem of :attr:`key`, used as the object's display id.
    """

    kind: ObjectKind
    """
    Whether the viewer renders a served mesh or a placeholder box.
    """

    color: str
    """
    Colour assigned to this object from the shared palette.
    """

    mesh: Optional[str] = None
    """
    URL the mesh is served from, set only when :attr:`kind` is ``MESH``.
    """

    format: Optional[str] = None
    """
    Mesh file extension, set only when :attr:`kind` is ``MESH``.
    """

    size: Optional[List[float]] = None
    """
    Box extent in metres, set only when :attr:`kind` is ``BOX``.
    """

    shapes: Optional[List[ShapeEntry]] = None
    """
    The body's shapes, set only when :attr:`kind` is ``SHAPES``.
    """


@dataclass
class PlanNodeEntry:
    """
    One plan node's serialized state, mutated in place as its status resolves.
    """

    id: str
    """
    Identity-based id of this node (``p`` + ``id(node)``).
    """

    parent: Optional[str]
    """
    Id of this node's parent entry, or None for the root.
    """

    kind: str
    """
    The plan node's own class name.
    """

    group: PlanNodeGroup
    """
    Colour group the viewer draws this node in, from :attr:`kind`.
    """

    label: str
    """
    Designator class name if this node describes an action, else :attr:`kind`.
    """

    status: str
    """
    This node's status: its own if it reports one, else a derived one.

    Not typed as :class:`TaskStatusName`, because it mirrors whatever coraplex's own
    ``TaskStatus`` reports, which this bridge does not validate.
    """

    derived: bool
    """
    Whether :attr:`status` was derived (from the statechart or children) rather than
    the node's own reported status.
    """

    arm: Optional[str] = None
    """
    Arm the node's designator names, if any.
    """

    target: Optional[str] = None
    """
    Published object the node's designator refers to, if any.
    """


@dataclass(frozen=True)
class PlanSnapshot:
    """
    The plan tree in the shape the viewer walks.
    """

    signature: str = ""
    """
    Node-id signature of the tree's shape, stable across status-only changes.
    """

    nodes: List[PlanNodeEntry] = field(default_factory=list)
    """
    Every node in the tree, flattened with parent references.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The snapshot plus the legend its groups are drawn with, so the viewer does not
        keep its own copy of the plan-node colour table.
        """
        payload = asdict(self)
        payload["legend"] = [
            {"group": group.value, "label": group.label}
            for group in PlanNodeGroup.legend()
        ]
        return payload


@dataclass(frozen=True)
class ChartNodeStructure:
    """
    The structural part of one statechart node: what does not change per tick.
    """

    id: str
    name: str
    class_name: str
    parent: Optional[str]


@dataclass(frozen=True)
class ChartEdgeEntry:
    """
    One transition edge between two statechart nodes.
    """

    source: str
    target: str
    kind: str

    def to_payload(self) -> Dict[str, str]:
        """
        This edge as the wire shape the frontend reads.

        Uses ``from``/``to`` rather than :attr:`source`/:attr:`target`, since ``from`` is a
        Python keyword and cannot be a dataclass field name.
        """
        return {"from": self.source, "to": self.target, "kind": self.kind}


@dataclass(frozen=True)
class _ChartStructure:
    """
    A statechart's cached structure, rebuilt only when the executor compiles a new one.
    """

    nodes: List[ChartNodeStructure] = field(default_factory=list)
    edges: List[ChartEdgeEntry] = field(default_factory=list)
    node_state_indices: List[int] = field(default_factory=list)
    """
    Each node's index into the chart's life-cycle/observation state vectors.
    """

    signature: str = ""
    """
    Node-id signature of the structure, stable while it does not change.
    """


class ObservationName(StrEnum):
    """
    A statechart node's trinary observation value, by name.
    """

    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ChartNodeEntry:
    """
    One statechart node's structure plus its current life cycle and observation.
    """

    id: str
    name: str
    class_name: str
    parent: Optional[str]
    life_cycle: str
    """
    The node's ``LifeCycleValues`` name (e.g. ``RUNNING``).
    """

    observation: ObservationName
    """
    The node's trinary observation name.
    """


@dataclass(frozen=True)
class ChartSnapshot:
    """
    The motion statechart in the shape the viewer renders.
    """

    signature: str = ""
    title: str = ""
    """
    Name of the action whose motion group this statechart belongs to.
    """

    nodes: List[ChartNodeEntry] = field(default_factory=list)
    edges: List[ChartEdgeEntry] = field(default_factory=list)


@dataclass(frozen=True)
class WorldStateSnapshot:
    """
    The world's joints, base pose and object poses at one simulation tick.
    """

    sequence_number: int = 0
    """
    Monotonic snapshot counter so the viewer can skip unchanged states.
    """

    frames: Dict[str, float] = field(default_factory=dict)
    """
    Movable connection position by prefixed name.
    """

    base: Optional[List[float]] = None
    """
    Robot base pose as ``[x, y, z, qx, qy, qz, qw]``, or None without a robot.
    """

    objects: Dict[str, List[float]] = field(default_factory=dict)
    """
    Loose-object pose by mesh key, in the same 7-element form as :attr:`base`.
    """

    markers_version: int = 0
    """
    Version of the debug-marker overlay; the viewer refetches ``/markers`` on change.
    """

    model_bases: Dict[str, List[float]] = field(default_factory=dict)
    """
    Every bundled model's root pose by world-instance prefix, in the same 7-element
    form as :attr:`base`, so a second robot or a moved environment model animates.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The snapshot in the camel-cased JSON shape the viewer reads.
        """
        payload = asdict(self)
        payload["sequenceNumber"] = payload.pop("sequence_number")
        payload["modelBases"] = payload.pop("model_bases")
        payload["markersVersion"] = payload.pop("markers_version")
        return payload


@dataclass(frozen=True)
class BridgeStatus:
    """
    What the viewer polls to decide whether a live demo is reachable.
    """

    running: bool
    robot: Optional[str]
    objects: List[str]
    movable: bool
    plan: bool
    chart: bool
    sequence_number: int
    model_version: int = 0
    """
    How many model sources the demo has parsed so far; the viewer reloads the live
    scene when this grows, so a model loaded mid-run appears.
    """

    bundle_signature: str = ""
    """
    Digest of what a live bundle built right now would contain (see
    :meth:`ModelBundleContext.signature`); the viewer reloads the live scene when it
    no longer matches the one its loaded bundle carries.
    """

    robot_parts: List[RobotPartAnnotation] = field(default_factory=list)
    """
    The arms and end effectors of the live robot, as sem_dt annotates them.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The status in the JSON shape the viewer polls, with the robot parts in the same
        ``partAnnotations`` shape a recorded scene bundle carries.
        """
        payload = asdict(self)
        payload["sequenceNumber"] = payload.pop("sequence_number")
        payload["modelVersion"] = payload.pop("model_version")
        payload["bundleSignature"] = payload.pop("bundle_signature")
        payload.pop("robot_parts")
        payload["partAnnotations"] = [
            annotation.to_payload() for annotation in self.robot_parts
        ]
        return payload


@dataclass
class Bridge:
    """
    Shared state between the running demo and the viewer.

    All world reads and writes happen on the simulation thread (the tick hook); the HTTP
    handlers only ever read the finished snapshot dicts under :attr:`_lock`.
    """

    REBIND_INTERVAL_SECONDS: ClassVar[float] = 3.0
    """
    How long a world binding stays fresh before bodies are re-discovered.
    """

    DEFAULT_OBJECT_SIZE: ClassVar[Tuple[float, float, float]] = (0.06, 0.06, 0.12)
    """
    Fallback size for an object whose shapes carry no scale, in metres.
    """

    world: Optional[World] = None
    """
    The executing world, captured by the tick hook on its first call.
    """

    robot: Optional[AbstractRobot] = None
    """
    The robot annotation of :attr:`world`, re-discovered on every bind.
    """

    sequence_number: int = 0
    """
    Monotonic snapshot counter so the viewer can skip unchanged states.
    """

    state: WorldStateSnapshot = field(default_factory=WorldStateSnapshot)
    """
    The newest world snapshot in the trajectory-frame format.
    """

    object_metadata: List[ObjectCatalogEntry] = field(default_factory=list)
    """
    Geometry catalog for the viewer: one entry per loose object.
    """

    plan_state: PlanSnapshot = field(default_factory=PlanSnapshot)
    """
    The newest plan-tree snapshot (see :meth:`snapshot_plan`).
    """

    chart_state: ChartSnapshot = field(default_factory=ChartSnapshot)
    """
    The newest motion-statechart snapshot (see :meth:`observe_chart`).
    """

    _connections: List[ActiveConnection1DOF] = field(default_factory=list)
    """
    Actuated world connections whose positions are published as frames.
    """

    _bodies: Dict[str, Body] = field(default_factory=dict)
    """
    Published bodies by mesh key; :data:`ROBOT_BASE_KEY` is the robot root.
    """

    _last_bind_time: float = 0.0
    """
    Timestamp of the last world discovery (see :attr:`REBIND_INTERVAL_SECONDS`).
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards every snapshot dict that the HTTP layer reads.
    """

    _moves: List[MoveRequest] = field(default_factory=list)
    """
    Object moves queued by the viewer, applied on the simulation thread.
    """

    _moves_lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards :attr:`_moves` (written by HTTP threads).
    """

    _mesh_serve: Dict[str, str] = field(default_factory=dict)
    """
    Object key → absolute mesh path served via the ``/mesh`` endpoint.
    """

    _plan: Optional[Plan] = None
    """
    The coraplex plan captured by the ``Plan.perform`` hook.
    """

    _chart: Optional[MotionStatechart] = None
    """
    The motion statechart the executor is currently ticking.
    """

    _chart_structure: Optional[_ChartStructure] = None
    """
    Serialized structure of :attr:`_chart`, rebuilt when it changes.
    """

    _chart_title: str = ""
    """
    Name of the action whose motion group is executing.
    """

    _motion_nodes: Dict[int, MotionNodeProgress] = field(default_factory=dict)
    """
    Execution progress per plan node, keyed by the node's :func:`id`.

    Identity, not equality: coraplex's ``DesignatorNode`` compares by field value, so
    two structurally identical steps of one plan would otherwise share a status. The
    :class:`MotionNodeProgress` entry pins the node itself, which keeps CPython from
    handing its ``id`` to a later object.

    Reset whenever a new plan starts performing, which bounds it to one plan's nodes.
    """

    _tick_count: int = 0
    """
    Tick counter used to throttle the plan snapshot.
    """

    plan_snapshot_tick_interval: int = 5
    """
    How many simulation ticks pass between plan-tree snapshots.

    Walking the plan tree is the expensive part of a tick, and the tree changes far
    more slowly than the world pose does.
    """

    _last_node_states: Optional[Tuple[List[int], List[float]]] = None
    """
    Life-cycle and observation vectors of the last published chart snapshot.
    """

    _model_revision: int = 0
    """
    Counts world attachments and model changes, reported as the status's model version.
    """

    _marker_stores: Dict[str, MarkerStore] = field(default_factory=dict)
    """
    The ROS debug markers per subscribed topic (see :mod:`cramera.live.ros_markers`).
    """

    marker_listener: Optional[Any] = None
    """
    The ROS subscription feeding the marker overlay, while one runs — the viewer's
    marker settings manage its topics through the bridge.
    """

    marker_state: Dict[str, Any] = field(
        default_factory=lambda: {"version": 0, "markers": []}
    )
    """
    The newest marker-overlay snapshot the HTTP layer serves.
    """

    _published_marker_revision: int = -1
    """
    The aggregate store revision :attr:`marker_state` was built from.
    """

    _marker_state_version: int = 0
    """
    Monotonic version of :attr:`marker_state`; the sum of store revisions can revisit
    an old value after a topic is dropped, this never does.
    """

    _bundle_signature: str = ""
    """
    Cached digest of the bundled scene content, recomputed on attach and model change.
    """

    live_server: Optional[ThreadingHTTPServer] = None
    """
    The bridge's HTTP server once it is listening, so a second start reuses it.
    """

    # %% what the visualization drives
    def attach(self, world: World) -> None:
        """
        Bind to the world a demo is executing and publish its geometry catalog.

        :param world: The world the demo is executing in.
        """
        self.world = world
        self._model_revision += 1
        self.bind()
        self._refresh_bundle_signature()
        logger.info(
            "attached to world (robot=%s, %d joints)",
            type(self.robot).__name__ if self.robot else "?",
            len(self._connections),
        )

    def observe_motion_tick(self, chart: MotionStatechart) -> None:
        """
        Publish everything one motion executor tick makes available.

        Applies queued viewer moves first, because the executor tick runs on the only
        thread allowed to write to the world; the world snapshot itself follows from
        the state change the tick causes.

        :param chart: The motion statechart the executor is ticking.
        """
        self.apply_moves()
        self.observe_chart(chart)
        self._tick_count += 1
        if self._tick_count % self.plan_snapshot_tick_interval == 0:
            self.snapshot_plan()

    def observe_motion_started(self, node: MotionNode) -> None:
        """
        Record that a plan node's motion started running.

        :param node: The node whose motion started.
        """
        self._motion_nodes[id(node)] = MotionNodeProgress(
            node=node, status=TaskStatusName.RUNNING
        )
        action_node = node.parent_action_node
        if action_node is not None and action_node.designator is not None:
            self._chart_title = type(action_node.designator).__name__

    def observe_motion_ended(self, node: MotionNode) -> None:
        """
        Pin the final status of a finished motion node and republish the plan.

        :param node: The node whose motion ended.
        """
        self._motion_nodes[id(node)] = MotionNodeProgress(
            node=node, status=TaskStatusName(node.status.name)
        )
        self.snapshot_plan()

    def begin_plan(self, plan: Plan) -> None:
        """
        Record the plan that started performing and publish its tree.

        Drops the previous plan's per-node progress, so a long-running process does not
        accumulate entries for nodes that no longer exist.

        :param plan: The plan that started performing.
        """
        self._plan = plan
        self._motion_nodes.clear()
        self.snapshot_plan()

    def observe_model_change(self) -> None:
        """
        Refresh the catalogs and the bundle signature after a world model change.
        """
        self._model_revision += 1
        self.bind()
        self._refresh_bundle_signature()

    def observe_ros_markers(self, topic: str, markers: List[Any]) -> None:
        """
        Apply one received ``MarkerArray`` (called on the ROS subscriber thread).

        Only the store is touched here; the publishable payload is rebuilt on the
        simulation thread, which may read the world for frame resolution.

        :param topic: The topic the array arrived on.
        :param markers: The array's markers.
        """
        store = self._marker_stores.setdefault(topic, MarkerStore())
        store.observe(markers)

    def _marker_revision(self) -> int:
        """
        The aggregate revision over every topic's marker store.
        """
        return sum(store.revision for store in self._marker_stores.values())

    def _refresh_marker_state(self) -> None:
        """
        Rebuild the marker overlay payload if any store changed since the last build.

        Runs on the simulation thread: excluding the world-model markers and
        resolving marker frames both read the world. Markers whose namespace names a
        world entity are the robot/environment geometry the scene already renders,
        and stay out of the overlay.
        """
        revision = self._marker_revision()
        if revision == self._published_marker_revision:
            return
        world_entity_names = set()
        if self.world is not None:
            world_entity_names = {str(body.name) for body in self.world.bodies} | {
                str(region.name) for region in self.world.regions
            }
        markers = []
        for topic in sorted(self._marker_stores):
            for entry in self._marker_stores[topic].entries.values():
                if entry.ns in world_entity_names:
                    continue
                markers.append(self._marker_payload(topic, entry))
        self._published_marker_revision = revision
        self._marker_state_version += 1
        with self._lock:
            self.marker_state = {
                "version": self._marker_state_version,
                "markers": markers,
            }

    def _marker_payload(self, topic: str, entry: MarkerEntry) -> Dict[str, Any]:
        """
        One marker as the viewer renders it, with its pose resolved into the world.

        :param topic: The topic the marker arrived on.
        :param entry: The marker to publish.
        """
        return {
            "topic": topic,
            "ns": entry.ns,
            "id": entry.id,
            "kind": entry.kind,
            "pose": self._marker_world_pose(entry),
            "scale": entry.scale,
            "color": entry.color,
            "opacity": entry.opacity,
            "points": entry.points,
            "text": entry.text,
        }

    def _marker_world_pose(self, entry: MarkerEntry) -> List[float]:
        """
        A marker's pose in world coordinates, as ``[x, y, z, qx, qy, qz, qw]``.

        A frame naming a world body anchors the marker to that body's current pose;
        the world root (under any of its usual names) and unknown frames read as the
        world itself.

        :param entry: The marker whose pose is resolved.
        """
        local = entry.position + entry.quaternion
        world = self.world
        if world is None:
            return local
        frame_body = self._marker_frame_body(entry.frame)
        if frame_body is None:
            return local
        frame_T_marker = HomogeneousTransformationMatrix.from_xyz_quaternion(*local)
        world_T_marker = frame_body.global_pose.to_homogeneous_matrix() @ frame_T_marker
        return [
            round(value, POSE_PRECISION)
            for value in world_T_marker.to_position_quaternion_list()
        ]

    def _marker_frame_body(self, frame: str) -> Optional[Body]:
        """
        The world body a marker frame names, or None for the world root and frames
        the world does not know.

        :param frame: The marker's ``frame_id``.
        """
        root_name = str(self.world.root.name)
        if frame in ("", "map", "world", root_name, root_name.split("/")[-1]):
            return None
        for body in self.world.bodies:
            name = str(body.name)
            if frame == name or frame == name.split("/")[-1]:
                return body
        return None

    def get_markers(self) -> Dict[str, Any]:
        """
        The debug-marker overlay the viewer renders.
        """
        with self._lock:
            return self.marker_state

    def marker_topics_payload(self) -> Dict[str, Any]:
        """
        The marker settings the viewer offers: what is watched and what the ROS graph
        advertises.
        """
        if self.marker_listener is None:
            return {"ok": True, "ros": False, "subscribed": [], "available": []}
        subscribed = self.marker_listener.subscribed_topics()
        return {
            "ok": True,
            "ros": True,
            "subscribed": subscribed,
            "available": sorted(
                set(self.marker_listener.available_marker_topics()) | set(subscribed)
            ),
        }

    def set_marker_topic(self, topic: str, subscribed: bool) -> Dict[str, Any]:
        """
        Start or stop watching one marker topic, as the viewer's settings ask.

        Stopping also drops the topic's markers, the way removing an RViz display
        clears what it showed.

        :param topic: The topic to watch or drop.
        :param subscribed: Whether the topic should be watched.
        """
        if self.marker_listener is None:
            return {"ok": False, "error": "no ROS in the demo process"}
        if not topic.startswith("/"):
            return {"ok": False, "error": "a topic starts with '/'"}
        if subscribed:
            self.marker_listener.subscribe(topic)
        else:
            self.marker_listener.unsubscribe(topic)
            store = self._marker_stores.pop(topic, None)
            if store is not None and store.entries:
                # force the next snapshot to rebuild without this topic
                self._published_marker_revision = -1
        return self.marker_topics_payload()

    def publish_bodies(self, bodies: Dict[str, Body]) -> None:
        """
        Replace the published bodies and rebuild the viewer's geometry catalog.

        :param bodies: The current published bodies, keyed by mesh key.
        """
        self._bodies = bodies
        self._build_object_metadata(bodies)

    # %% what the HTTP layer reads
    def object_catalog(self) -> List[Dict[str, Any]]:
        """
        The geometry catalog the viewer spawns live objects from.
        """
        with self._lock:
            return [asdict(entry) for entry in self.object_metadata]

    def object_keys(self) -> List[str]:
        """
        Mesh keys of the published loose objects, excluding the robot root.
        """
        with self._lock:
            return [key for key in self._bodies if key != ROBOT_BASE_KEY]

    def mesh_path(self, key: str) -> Optional[str]:
        """
        Absolute path of an object's mesh file, or None if it is not served.

        :param key: Mesh key of the object, as published in the geometry catalog.
        """
        with self._lock:
            return self._mesh_serve.get(key)

    def bundle_signature(self) -> str:
        """
        A digest of the bundled scene's content: the identity, parentage and connection
        type of every body the live bundle serializes, plus the robot's identity.

        Deliberately excludes the overlay's mesh-named objects — a demo re-parenting a
        grasped object changes the world model but not the bundled scene, and must not
        make the viewer reload it. State changes never touch it either.
        """
        return self._bundle_signature

    def _refresh_bundle_signature(self) -> None:
        """
        Recompute the cached bundle signature from the current world model.
        """
        if self.world is None:
            self._bundle_signature = ""
            return
        robot_name = type(self.robot).__name__.lower() if self.robot else None
        entries: List[str] = []
        try:
            for body in self.world.bodies:
                name = str(body.name)
                if MeshFormat.of_path(name.split("/")[-1]) is not None:
                    continue
                connection = body.parent_connection
                entries.append(
                    "%s<-%s:%s"
                    % (
                        name,
                        str(connection.parent.name) if connection else "",
                        type(connection).__name__ if connection else "root",
                    )
                )
        except Exception as error:
            # boundary guard: the world is mid-modification and iterating it is not
            # safe; keep the previous signature rather than flapping the viewer.
            logger.debug("signature refresh skipped: %s", error)
            return
        digest = hashlib.sha1("|".join(sorted(entries)).encode()).hexdigest()[:16]
        self._bundle_signature = "world-%s-robot-%s" % (digest, robot_name)

    def status(self) -> Dict[str, Any]:
        """
        What the viewer polls to decide whether a live demo is reachable.
        """
        bundle_signature = self.bundle_signature()
        with self._lock:
            return BridgeStatus(
                running=self.world is not None,
                robot=type(self.robot).__name__ if self.robot else None,
                objects=[key for key in self._bodies if key != ROBOT_BASE_KEY],
                movable=True,
                plan=bool(self.plan_state.nodes),
                chart=bool(self.chart_state.nodes),
                sequence_number=self.sequence_number,
                model_version=self._model_revision,
                bundle_signature=bundle_signature,
                robot_parts=(
                    RobotPartAnnotation.of_robot(self.robot)
                    if self.robot is not None
                    else []
                ),
            ).to_payload()

    # %% viewer -> world
    def queue_move(self, request: MoveRequest) -> None:
        """
        Queue an object move from the viewer (called on an HTTP thread).

        :param request: The move to apply on the next simulation tick.
        """
        with self._moves_lock:
            self._moves.append(request)

    def apply_moves(self) -> None:
        """
        Apply queued object moves to the world.

        Called from the tick hook — the simulation thread is the only place that may
        write to the world.
        """
        with self._moves_lock:
            moves, self._moves = self._moves, []
        if not moves or self.world is None:
            return
        for move in moves:
            body = self._bodies.get(move.object_key)
            if body is None:
                logger.debug("no published body for %s — move skipped", move.object_key)
                continue
            self._apply_move(move, body)

    def _apply_move(self, move: MoveRequest, body: Body) -> None:
        """
        Write one viewer move into the world.

        Only free-floating (:class:`Connection6DoF`) objects are draggable.
        Objects rigidly fixed to furniture — e.g. a spoon on a drawer that
        must ride along when the drawer opens — keep their ``FixedConnection``
        and are left untouched (a fixed connection has no settable origin).

        .. note:: The object is not re-parented. That is a structural change of the
           kinematic tree (``modify_world`` + forward kinematics recompile), and
           running it inside the tick hook while a giskard goal is live hangs the
           executor. The plain pose write already makes ``body.global_pose`` correct,
           which is what the plan's navigate/pick reachability reads.

        :param move: The queued move to apply.
        :param body: The body the move targets.
        """
        connection = body.parent_connection
        if not isinstance(connection, Connection6DoF):
            logger.info(
                "%s is fixed (%s) — not draggable, skipping",
                move.object_key,
                type(connection).__name__,
            )
            return
        position = move.position
        rotation_matrix = (
            RotationMatrix.from_quaternion(
                Quaternion(
                    x=move.quaternion[0],
                    y=move.quaternion[1],
                    z=move.quaternion[2],
                    w=move.quaternion[3],
                )
            )
            if move.quaternion is not None
            else body.global_pose.to_rotation_matrix()
        )
        world_T_object = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            Point3(x=position[0], y=position[1], z=position[2]),
            rotation_matrix,
            reference_frame=self.world.root,
        )
        # ``origin`` is parent-relative; express the target in the parent
        # frame (a no-op while the parent is the world root)
        parent_T_world = connection.parent.global_pose.to_homogeneous_matrix().inverse()
        parent_T_object = parent_T_world @ world_T_object
        # the matrix product drops the frames; the origin setter transforms whatever
        # frame it is handed into the parent frame, so label the result explicitly
        parent_T_object.reference_frame = connection.parent
        parent_T_object.child_frame = body
        connection.origin = parent_T_object
        logger.info(
            "moved %s -> world (%.3f, %.3f, %.3f) [final=%s]",
            move.object_key,
            position[0],
            position[1],
            position[2],
            move.is_final,
        )

    # %% world discovery
    def bind(self) -> None:
        """
        Discover the robot, joints and publishable bodies of the current world.

        Re-run periodically because demos modify their world (objects get spawned and
        removed mid-run).
        """
        world = self.world
        if world is None:
            return
        self._last_bind_time = time.time()
        robots = world.get_semantic_annotations_by_type(AbstractRobot)
        self.robot = robots[0] if robots else None
        self._connections = self._actuated_connections(world)
        bodies: Dict[str, Body] = {}
        if self.robot is not None:
            bodies[ROBOT_BASE_KEY] = self.robot.root
        try:
            bodies_by_name = {str(body.name): body for body in world.bodies}
            bodies.update(self._discover_overlay_bodies(bodies_by_name))
        except Exception as error:
            # boundary guard: the world is mid-modification (a body is being spawned
            # or removed) and iterating it is not safe. Keep the previous catalog
            # rather than publishing an empty one, which would make the viewer hide
            # every object it already shows.
            logger.debug("body scan skipped this bind: %s", error)
            for key, body in self._bodies.items():
                bodies.setdefault(key, body)
        self.publish_bodies(bodies)

    def _discover_overlay_bodies(
        self, bodies_by_name: Dict[str, Body]
    ) -> Dict[str, Body]:
        """
        Every world body the overlay renders, keyed the way it is published.

        Bodies named like mesh files are the demo's objects — they spawn, get carried
        and disappear mid-run, so their poses stream through the overlay. Every other
        body is part of the bundled scene the viewer loads once.

        :param bodies_by_name: Every world body by its full name.
        """
        robot_root = self.robot.root if self.robot is not None else None
        bodies: Dict[str, Body] = {}
        for full_name, body in bodies_by_name.items():
            if body is robot_root:
                continue
            basename = full_name.split("/")[-1]
            if MeshFormat.of_path(basename) is not None:
                bodies[basename] = body
        return bodies

    @staticmethod
    def _body_shapes(body: Body) -> List[Any]:
        """
        The shapes a body is rendered from: its visual ones, else its collision ones.

        :param body: The body whose shapes are read.
        """
        for shape_collection in (body.visual, body.collision):
            if shape_collection.shapes:
                return list(shape_collection.shapes)
        return []

    @staticmethod
    def _actuated_connections(world: World) -> List[ActiveConnection1DOF]:
        """
        All 1-DOF connections — the joints published as trajectory frames.

        :param world: The world to scan for actuated connections.
        """
        return [
            connection
            for connection in world.connections
            if isinstance(connection, ActiveConnection1DOF)
        ]

    def _build_object_metadata(self, bodies: Dict[str, Body]) -> None:
        """
        Rebuild the geometry catalog the viewer spawns live objects from.

        Each object gets a mesh URL (served by the bridge), its real shapes, or a
        fallback box size, so objects the viewer does not know yet can appear mid-run.

        :param bodies: The current published bodies, keyed by mesh key.
        """
        catalog: List[ObjectCatalogEntry] = []
        serve: Dict[str, str] = {}
        palette = ObjectPalette()
        for index, (key, body) in enumerate(
            item for item in bodies.items() if item[0] != ROBOT_BASE_KEY
        ):
            color = palette.color_for(index)
            object_id = Path(key).stem
            shapes = self._body_shapes(body)
            if shapes:
                catalog.append(self._shape_catalog_entry(key, shapes, color, serve))
                continue
            catalog.append(
                ObjectCatalogEntry(
                    key=key,
                    id=object_id,
                    kind=ObjectKind.BOX,
                    color=color,
                    size=list(self.DEFAULT_OBJECT_SIZE),
                )
            )
        self._mesh_serve = serve
        with self._lock:
            self.object_metadata = catalog

    def _shape_catalog_entry(
        self,
        key: str,
        shapes: List[Any],
        fallback_color: str,
        serve: Dict[str, str],
    ) -> ObjectCatalogEntry:
        """
        The catalog entry of a body published shape by shape.

        Mesh shapes are registered in the serve map under a composite key, so each of
        a body's meshes is downloadable on its own.

        :param key: The body's published key.
        :param shapes: The body's shapes, as :meth:`_body_shapes` selects them.
        :param fallback_color: Palette colour used for shapes without one of their own.
        :param serve: The serve map being built, extended with this body's mesh files.
        """
        entries: List[ShapeEntry] = []
        for shape_index, shape in enumerate(shapes):
            mesh_url = None
            mesh_file = served_mesh_file(shape)
            if mesh_file is not None:
                serve_key = "%s#%d" % (key, shape_index)
                serve[serve_key] = mesh_file
                mesh_url = "/mesh?key=" + urllib.parse.quote(serve_key, safe="")
            entries.append(
                shape_entry(
                    shape,
                    mesh_url,
                    fallback_size=list(self.DEFAULT_OBJECT_SIZE),
                    fallback_color=fallback_color,
                )
            )
        return ObjectCatalogEntry(
            key=key,
            id=Path(key).stem,
            kind=ObjectKind.SHAPES,
            color=entries[0].color,
            shapes=entries,
        )

    # %% world snapshot
    def snapshot(self) -> None:
        """
        Publish the world's joints, base pose and object poses.

        Runs on the simulation thread; rebinds the world periodically so mid-run spawns
        show up.
        """
        if self.world is None:
            return
        if time.time() - self._last_bind_time > self.REBIND_INTERVAL_SECONDS:
            self.bind()
        frames = {
            str(connection.name): round(float(connection.position), POSE_PRECISION)
            for connection in self._connections
        }
        base_pose: Optional[List[float]] = None
        object_poses: Dict[str, List[float]] = {}
        for name, body in self._bodies.items():
            if name == ROBOT_BASE_KEY:
                base_pose = rounded_pose(body)
            else:
                object_poses[name] = rounded_pose(body)
        self._refresh_marker_state()
        with self._lock:
            self.sequence_number += 1
            self.state = WorldStateSnapshot(
                sequence_number=self.sequence_number,
                frames=frames,
                base=base_pose,
                objects=object_poses,
                markers_version=self.marker_state["version"],
            )

    def get_state(self) -> Dict[str, Any]:
        """
        The newest world snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            return self.state.to_payload()

    # %% plan tree
    def _live_motion_status(self, node: PlanNode) -> Optional[str]:
        """
        Status of one plan node as its plan callbacks reported it, or None.

        :param node: The plan node whose live status is looked up.
        """
        progress = self._motion_nodes.get(id(node))
        if progress is None:
            return None
        return progress.status

    def snapshot_plan(self) -> None:
        """
        Serialize the plan tree with per-node execution status.

        A node's own status wins while it says something; otherwise the live statechart
        status is used, else the aggregate of the children (running/failed bubble up; a
        parent whose children are only partly done reads as running, not succeeded).
        """
        plan = self._plan
        if plan is None:
            return
        try:
            root = plan.root
        except Exception:
            # the plan is mid-mutation and not a tree right now — next tick
            return
        nodes: List[PlanNodeEntry] = []
        order: List[str] = []
        self._serialize_plan_node(root, None, nodes, order)
        with self._lock:
            self.plan_state = PlanSnapshot(signature="|".join(order), nodes=nodes)

    def _serialize_plan_node(
        self,
        node: PlanNode,
        parent_id: Optional[str],
        nodes: List[PlanNodeEntry],
        order: List[str],
    ) -> str:
        """
        Serialize one plan node and its subtree; returns the node's status.

        :param node: The plan node to serialize.
        :param parent_id: Id of the node's parent entry, or None for the root.
        :param nodes: Output list every serialized entry is appended to.
        :param order: Output list every serialized node id is appended to, in
            traversal order, to build the tree's signature.
        """
        node_id = "plan_node_%d" % id(node)
        designator = node.designator if isinstance(node, DescribesAnAction) else None
        own_status = node.status.name
        entry = PlanNodeEntry(
            id=node_id,
            parent=parent_id,
            kind=type(node).__name__,
            group=PlanNodeGroup.of_plan_node_kind(type(node).__name__),
            label=(
                type(designator).__name__
                if designator is not None
                else type(node).__name__
            ),
            status=own_status,
            derived=False,
        )
        self._add_designator_metadata(entry, designator)
        nodes.append(entry)
        order.append(node_id)

        child_best, children, done = "CREATED", 0, 0
        for child in node.children:
            child_status = self._serialize_plan_node(child, node_id, nodes, order)
            child_best = self._max_status(child_best, child_status)
            children += 1
            if child_status == "SUCCEEDED":
                done += 1
        if own_status == "CREATED":
            if child_best == "SUCCEEDED" and done < children:
                child_best = "RUNNING"
            derived = self._live_motion_status(node) or (
                child_best if child_best != "CREATED" else None
            )
            if derived:
                entry.status = derived
                entry.derived = True
        return entry.status

    def _add_designator_metadata(
        self, entry: PlanNodeEntry, designator: Optional[Any]
    ) -> None:
        """
        Add arm and target-object info from a node's designator, if any.

        :param entry: The serialized entry to fill in, mutated in place.
        :param designator: The node's designator, or None.
        """
        if designator is None:
            return
        fields = vars(designator)
        arm = fields.get("arm") or fields.get("arms")
        if arm is not None:
            entry.arm = str(arm)
        target = self._designator_target(designator)
        if target:
            entry.target = target

    @staticmethod
    def _max_status(first: str, second: str) -> str:
        """
        The higher-ranked of two statuses.

        :param first: The first status to compare.
        :param second: The second status to compare.
        """
        if TaskStatusName.rank_of(first) >= TaskStatusName.rank_of(second):
            return first
        return second

    def _designator_target(self, designator: Any) -> Optional[str]:
        """
        Published key of the object a designator refers to, if any.

        Matched by basename, because designators name world entities with their full
        prefixed name while some objects are published under a basename key.

        :param designator: The designator to search for a world-entity reference.
        """
        keys_by_basename = {key.split("/")[-1]: key for key in self._bodies}
        for value in vars(designator).values():
            if not isinstance(value, NamesAWorldEntity):
                continue
            basename = str(value.name).split("/")[-1]
            if basename in keys_by_basename:
                return keys_by_basename[basename]
        return None

    def get_plan(self) -> Dict[str, Any]:
        """
        The newest plan snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            return self.plan_state.to_payload()

    # %% motion statechart
    def observe_chart(self, chart: Optional[MotionStatechart]) -> None:
        """
        Publish the executing statechart's structure and node states.

        Called from the tick hook. The structure is re-serialized only when the executor
        compiled a new chart; the life-cycle and observation vectors are cheap and
        republished whenever either of them changes.

        :param chart: The motion statechart the executor is currently ticking, if any.
        """
        if chart is None:
            return
        if chart is not self._chart or self._chart_structure is None:
            self._chart = chart
            self._chart_structure = self._serialize_chart_structure(chart)
            self._last_node_states = None
        structure = self._chart_structure
        if structure is None:
            return
        from giskardpy.motion_statechart.data_types import LifeCycleValues

        life_cycle = [
            int(chart.life_cycle_state.data[index])
            for index in structure.node_state_indices
        ]
        observations = [
            float(chart.observation_state.data[index])
            for index in structure.node_state_indices
        ]
        if (life_cycle, observations) == self._last_node_states:
            return
        self._last_node_states = (life_cycle, observations)
        nodes = [
            ChartNodeEntry(
                id=node.id,
                name=node.name,
                class_name=node.class_name,
                parent=node.parent,
                life_cycle=LifeCycleValues(life_cycle[position]).name,
                observation=self._observation_name(observations[position]),
            )
            for position, node in enumerate(structure.nodes)
        ]
        with self._lock:
            self.chart_state = ChartSnapshot(
                signature=structure.signature,
                title=self._chart_title,
                nodes=nodes,
                edges=structure.edges,
            )

    @staticmethod
    def _observation_name(observation: float) -> ObservationName:
        """
        Trinary observation value → name (0 false, 0.5 unknown, 1 true).

        :param observation: The raw trinary observation value.
        """
        if observation >= 0.75:
            return ObservationName.TRUE
        if observation <= 0.25:
            return ObservationName.FALSE
        return ObservationName.UNKNOWN

    @staticmethod
    def _serialize_chart_structure(chart: MotionStatechart) -> _ChartStructure:
        """
        Nodes and transition edges of a statechart.

        :param chart: The statechart to serialize.
        """
        nodes: List[ChartNodeStructure] = []
        node_state_indices: List[int] = []
        for node in chart.nodes:
            parent_index = node.parent_node_index
            nodes.append(
                ChartNodeStructure(
                    id="chart_node_%d" % node.index,
                    name=node.name,
                    class_name=type(node).__name__,
                    parent=(
                        ("chart_node_%d" % parent_index)
                        if parent_index is not None
                        else None
                    ),
                )
            )
            node_state_indices.append(node.index)
        edges = []
        for source, target, transition in chart.rx_graph.edge_index_map().values():
            edges.append(
                ChartEdgeEntry(
                    source="chart_node_%d" % chart.rx_graph.get_node_data(source).index,
                    target="chart_node_%d" % chart.rx_graph.get_node_data(target).index,
                    kind=transition.kind.name,
                )
            )
        signature = "|".join(node.id + ":" + node.name for node in nodes)
        return _ChartStructure(
            nodes=nodes,
            edges=edges,
            node_state_indices=node_state_indices,
            signature=signature,
        )

    def get_chart(self) -> Dict[str, Any]:
        """
        The newest statechart snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            chart = self.chart_state
        payload = asdict(chart)
        payload["edges"] = [edge.to_payload() for edge in chart.edges]
        return payload


BRIDGE = Bridge()
"""
The process-wide bridge instance shared by hooks and HTTP handlers.
"""
