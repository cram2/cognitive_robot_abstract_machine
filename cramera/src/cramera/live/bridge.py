"""Live world, plan and motion snapshots for one visualization session."""

from __future__ import annotations

import hashlib
import threading
import time
import urllib.parse
from collections.abc import Callable
from contextlib import contextmanager, ExitStack
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from http.server import ThreadingHTTPServer
from pathlib import Path

from typing_extensions import (
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    List,
    Optional,
    TYPE_CHECKING,
)
from coraplex.plans.plan_node import DesignatorNode
from giskardpy.motion_statechart.data_types import LifeCycleValues
from krrood.entity_query_language.evaluable import Evaluable

from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world import World
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from cramera.logging_setup import get_logger
from cramera.config import CrameraConfig
from cramera.body_geometry import NumericPose, POSE_PRECISION, rounded_pose
from semantic_digital_twin.world_description.connections import (
    ActiveConnection1DOF,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import WorldEntity
from cramera.knowledge.enums import PlanNodeGroup
from cramera.live.chart_observer import ChartObserver
from cramera.live.chart_structure import (
    ChartSnapshot,
)
from cramera.knowledge.presets import Preset
from cramera.knowledge.query_runner import EqlQueryRunner, RenderResult
from cramera.knowledge.query_vocabulary import QueryVocabulary
from cramera.knowledge.queryable_knowledge import (
    QueryableKnowledge,
    QueryScope,
    UnknownQueryScope,
)
from cramera.knowledge.question_matching import QuestionMatcher, QuestionMatchResult
from cramera.knowledge.views.kinematics import UrdfViewPayload
from cramera.live.query import NoQuerySourceRegistered
from cramera.live.markers import MarkerEntry, MarkerStore
from cramera.live.shape_catalog import (
    color_to_hex,
    is_default_white,
    served_mesh_file,
    shape_entry,
)
from cramera.live.transforms import TransformGraph, TransformSnapshot
from cramera.world_objects import WorldObjects
from cramera.palette import ObjectPalette
from cramera.robot_parts import RobotPartAnnotation
from cramera.recording_fields import SceneField

if TYPE_CHECKING:
    from collections.abc import Iterator
    from coraplex.datastructures.enums import Arms
    from coraplex.plans.designator import Designator
    from coraplex.plans.plan import Plan
    from coraplex.plans.plan_node import MotionNode, PlanNode
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart
    from semantic_digital_twin.world_description.world_entity import Body, Connection

    from cramera.live.recording import Recording
    from cramera.live.ros_markers import RosMarkerListener

logger = get_logger(__name__)


class DesignatorParameter(StrEnum):
    """Native designator parameters naming the selected arms."""

    ARM = "arm"
    """The native arm selection, including both arms."""


# %% viewer payload shapes
@dataclass(frozen=True)
class ObjectCatalogEntry:
    """
    One loose object's native geometry and publication identity.
    """

    key: str
    """
    Published key shared by this object's geometry and pose snapshots.
    """

    shapes: ShapeCollection
    """
    Visual geometry, collision geometry, or a placeholder for a shapeless body.
    """

    fallback_color: str
    """
    Palette colour applied to shapes with no chosen colour.
    """

    @property
    def id(self) -> str:
        """Return the display identifier derived from the published key."""
        return Path(self.key).stem

    @property
    def color(self) -> str:
        """Return the first shape's colour, or its assigned palette colour."""
        if not self.shapes or is_default_white(self.shapes[0].color):
            return self.fallback_color
        return color_to_hex(self.shapes[0].color)

    def mesh_key(self, shape_index: int) -> str:
        """Identify one shape's served mesh within this object.

        :param shape_index: The shape's position in the native collection.
        :return: The key used to serve the shape's mesh file.
        """
        return "%s#%d" % (self.key, shape_index)

    def to_payload(
        self, mesh_files: dict[str, str], fallback_size: list[float]
    ) -> dict[str, Any]:
        """Describe native shapes with the browser's primitive and asset fields.

        :param mesh_files: Registered mesh files indexed by their published keys.
        :param fallback_size: Box dimensions for an unavailable mesh.
        :return: The object's geometry payload.
        """
        entries = []
        for shape_index, shape in enumerate(self.shapes):
            mesh_key = self.mesh_key(shape_index)
            mesh_url = (
                "/mesh?key=" + urllib.parse.quote(mesh_key, safe="")
                if mesh_key in mesh_files
                else None
            )
            entries.append(
                asdict(shape_entry(shape, mesh_url, fallback_size, self.fallback_color))
            )
        return {
            SceneField.KEY: self.key,
            SceneField.ID: self.id,
            SceneField.COLOR: self.color,
            SceneField.SHAPES: entries,
        }


@dataclass
class PlanNodeEntry:
    """
    One plan node's native lifecycle state and display metadata.
    """

    id: str
    """
    Identity-based id of this node (``plan_node_`` + ``id(node)``).
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

    status: LifeCycleValues
    """
    The native lifecycle state reported by this plan node.
    """

    derived: bool
    """
    Whether the published status was derived; native plan states are reported directly.
    """

    arm: Optional[str] = None
    """
    Arm the node's designator names, if any.
    """

    target: Optional[str] = None
    """
    Published object the node's designator refers to, if any.
    """

    def to_payload(self) -> Dict[str, Any]:
        """Serialize the node with its native lifecycle name.

        :return: The node fields with the lifecycle represented as text.
        """
        payload = asdict(self)
        payload[PlanTreeField.STATUS] = self.status.name
        return payload


class PlanTreeField(StrEnum):
    """Fields describing the published plan hierarchy and node lifecycle."""

    STATUS = "status"
    """The native lifecycle name of a plan node."""

    SIGNATURE = "signature"
    """The plan's node identities in traversal order."""

    NODES = "nodes"
    """The flattened plan entries with their parent references."""

    CHILDREN = "children"
    """Nested plan steps in execution order."""


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
        payload = {
            PlanTreeField.SIGNATURE: self.signature,
            PlanTreeField.NODES: [node.to_payload() for node in self.nodes],
        }
        payload["legend"] = [
            {"group": group.value, "label": group.label}
            for group in PlanNodeGroup.legend()
        ]
        return payload

    def recorded_trees(self) -> List[Dict[str, Any]]:
        """Build the recorded hierarchy from each node's parent reference.

        :return: Plan roots containing their children and action metadata.
        """
        entries = {
            node.id: {**node.to_payload(), PlanTreeField.CHILDREN: []}
            for node in self.nodes
        }
        roots = []
        for node in self.nodes:
            entry = entries[node.id]
            if node.parent in entries:
                entries[node.parent][PlanTreeField.CHILDREN].append(entry)
            else:
                roots.append(entry)
        return roots


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

    ORIENTATION_START: ClassVar[int] = 3
    """
    Index the quaternion begins at in a ``[x, y, z, qx, qy, qz, qw]`` pose.
    """

    markers_version: int = 0
    """
    Version of the debug-marker overlay; the viewer refetches ``/markers`` on change.
    """

    def orientation_of(self, object_key: str) -> Optional[List[float]]:
        """
        The orientation one object stands at, or None if this snapshot has no pose for it.

        :param object_key: Mesh key of the object whose orientation is read.
        """
        pose = self.objects.get(object_key)
        if pose is None:
            return None
        return list(pose[self.ORIENTATION_START :])

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
    """Whether a world is attached."""

    robot: Optional[str]
    """Name of the bound robot model."""

    objects: List[str]
    """Published loose-object identifiers."""

    movable: bool
    """Whether the world contains objects with changing poses."""

    plan: bool
    """Whether a plan snapshot is available."""

    chart: bool
    """Whether a motion-statechart snapshot is available."""

    query: bool
    """
    Whether an attached world or an explicit source can answer live queries.
    """

    sequence_number: int
    """Sequence number of the latest world snapshot."""

    model_version: int = 0
    """
    How many model sources the demo has parsed so far; the viewer reloads the live
    scene when this grows, so a model loaded mid-run appears.
    """

    bundle_signature: str = ""
    """
    Digest of the current geometry; a changed value requests a scene reload.
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

    World callbacks publish snapshots under :attr:`_lock`; HTTP handlers read those
    snapshots without changing the world.
    """

    configuration: CrameraConfig = field(default_factory=CrameraConfig, kw_only=True)
    """Publication keys, discovery timing and fallback geometry for this session."""

    world: Optional[World] = None
    """
    The world explicitly attached to this visualization session.
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

    transform_state: TransformSnapshot = field(default_factory=TransformSnapshot)
    """
    The newest transform-graph snapshot (see :mod:`cramera.live.transforms`).
    """

    query_knowledge: (
        QueryableKnowledge
        | list[QueryableKnowledge]
        | Callable[[], QueryableKnowledge | list[QueryableKnowledge]]
        | None
    ) = None
    """
    Explicit query scopes overriding the attached world's default knowledge.
    """

    _query_title: str = field(default="", init=False)
    """
    Display title of explicitly registered query knowledge.
    """

    _query_presets: list[Preset] | Callable[[], list[Preset]] = field(
        default_factory=list, init=False
    )
    """
    Visible presets for explicitly registered query knowledge.
    """

    _unlisted_query_presets: list[Preset] | Callable[[], list[Preset]] = field(
        default_factory=list, init=False
    )
    """
    Additional registered presets recognized through natural-language questions.
    """

    _query_attachment: int | None = field(default=None, init=False, repr=False)
    """
    Model revision identifying the attachment that owns automatic world queries.
    """

    _query_revision: int = field(default=0, init=False, repr=False)
    """
    Registration revision used to keep knowledge and presets from one configuration.
    """

    _query_lock: threading.RLock = field(default_factory=threading.RLock)
    """
    Serializes queries: EQL evaluation is not written to run twice at once, and the
    bridge answers several viewers from its own thread pool.

    ..note:: This does not keep a query apart from the demo thread, which evaluates EQL
        of its own; what they share is the ``SymbolGraph`` singleton, which serializes
        itself.
    """

    _transforms: TransformGraph = field(default_factory=TransformGraph)
    """
    Tracks when each world connection last changed, across ticks.
    """

    _kinematic_connections: List[Connection] = field(default_factory=list)
    """
    Every world connection, of any kind, as the last bind discovered them.
    """

    _bodies: Dict[str, Body] = field(default_factory=dict)
    """
    Published bodies by mesh key, including the configured robot root key.
    """

    _last_bind_time: float = 0.0
    """
    Timestamp of the last world discovery.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock)
    """
    Guards every snapshot dict that the HTTP layer reads.
    """

    bundle_lock: threading.RLock = field(default_factory=threading.RLock)
    """Serializes this session's geometry and recording exports."""

    _mesh_serve: Dict[str, str] = field(default_factory=dict)
    """
    Object key → absolute mesh path served via the ``/mesh`` endpoint.
    """

    _plan: Optional[Plan] = None
    """
    The plan observed through its native execution callbacks.
    """

    _chart_observer: ChartObserver = field(default_factory=ChartObserver)
    """
    Reads what the executing statechart looks like, remembering what it last saw.
    """

    _chart_title: str = ""
    """
    Name of the action whose motion group is executing.
    """

    _model_revision: int = 0
    """
    Counts world attachments and model changes, reported as the status's model version.
    """

    _marker_stores: Dict[str, MarkerStore] = field(default_factory=dict)
    """
    The ROS debug markers per subscribed topic (see :mod:`cramera.live.ros_markers`).
    """

    _marker_lock: threading.Lock = field(default_factory=threading.Lock)
    """Guards marker ingestion, topic removal and snapshot capture."""

    _marker_revision: int = 0
    """Monotonic revision of the marker contents across all topics."""

    marker_listener: Optional[RosMarkerListener] = None
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
    The marker revision :attr:`marker_state` was built from.
    """

    _marker_state_version: int = 0
    """
    Monotonic version of the published :attr:`marker_state` snapshots.
    """

    _bundle_signature: str = ""
    """
    Cached digest of the bundled scene content, recomputed on attach and model change.
    """

    live_server: Optional[ThreadingHTTPServer] = None
    """
    The bridge's HTTP server once it is listening, so a second start reuses it.
    """

    recording: Optional[Recording] = None
    """
    The current live run's capture buffer, started alongside :meth:`attach` (see
    :mod:`cramera.live.visualization`); None before anything has ever attached.
    """

    # %% what the visualization drives
    def attach(self, world: World) -> int:
        """
        Publish a world's geometry and provide its default live query source.

        :param world: The world to visualize and query.
        :return: The revision identifying this attachment's query ownership.
        """
        self.world = world
        self._model_revision += 1
        self.bind()
        self._refresh_bundle_signature()
        self._query_attachment = self._model_revision
        logger.info(
            "attached to world (robot=%s, %d joints)",
            type(self.robot).__name__ if self.robot else "?",
            len(self._connections),
        )
        return self._query_attachment

    def release_world_queries(self, attachment: int) -> None:
        """
        Release the automatic source if it still belongs to the given attachment.

        Explicit sources and sources from newer attachments remain registered.

        :param attachment: The revision returned when the world was attached.
        """
        if self._query_attachment == attachment:
            self._query_attachment = None

    def observe_motion_started(self, node: MotionNode) -> None:
        """
        Name the executing chart from its motion's parent action.

        :param node: The node whose motion started.
        """
        action_node = node.parent_action_node
        if action_node is not None:
            self._chart_title = type(action_node.action).__name__

    def begin_plan(self, plan: Plan) -> None:
        """
        Record the plan that started performing and publish its tree.

        :param plan: The plan that started performing.
        """
        self._plan = plan
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
        with self._marker_lock:
            store = self._marker_stores.setdefault(topic, MarkerStore())
            if store.observe(markers):
                self._marker_revision += 1

    def _refresh_marker_state(self) -> None:
        """
        Rebuild the marker overlay payload if any store changed since the last build.

        Runs on the simulation thread: excluding the world-model markers and
        resolving marker frames both read the world. Markers whose namespace names a
        world entity are the robot/environment geometry the scene already renders,
        and stay out of the overlay.
        """
        with self._marker_lock:
            revision = self._marker_revision
            if revision == self._published_marker_revision:
                return
            marker_entries = {
                topic: tuple(store.entries.values())
                for topic, store in self._marker_stores.items()
            }
        world_entity_names = set()
        if self.world is not None:
            world_entity_names = {str(body.name) for body in self.world.bodies} | {
                str(region.name) for region in self.world.regions
            }
        markers = []
        for topic in sorted(marker_entries):
            for entry in marker_entries[topic]:
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
        return NumericPose.of_matrix(world_T_marker.to_np()).rounded()

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
            with self._marker_lock:
                store = self._marker_stores.pop(topic, None)
                if store is not None and store.entries:
                    self._marker_revision += 1
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
            return [
                entry.to_payload(
                    self._mesh_serve, list(self.configuration.default_object_size)
                )
                for entry in self.object_metadata
            ]

    def object_keys(self) -> List[str]:
        """
        Mesh keys of the published loose objects, excluding the robot root.
        """
        with self._lock:
            return [
                key for key in self._bodies if key != self.configuration.robot_base_key
            ]

    def _resolve_highlights(self, names: list[str]) -> list[str]:
        """Map native body names to their existing object or robot-link identifiers.

        :param names: Canonical entity names and existing viewer identifiers.
        :return: Sorted unique highlight identifiers, preserving unknown names.
        """
        kinematics_view = UrdfViewPayload()
        identifiers = (
            {
                name: kinematics_view.link_id(name)
                for name in WorldObjects(self.world, self.robot).robot_body_names()
            }
            if self.world is not None
            else {}
        )
        with self._lock:
            identifiers.update(
                {
                    str(body.name): key
                    for key, body in self._bodies.items()
                    if key != self.configuration.robot_base_key
                }
            )
        return sorted({identifiers.get(name, name) for name in names})

    def mesh_path(self, key: str) -> Optional[str]:
        """
        Absolute path of an object's mesh file, or None if it is not served.

        :param key: Mesh key of the object, as published in the geometry catalog.
        """
        with self._lock:
            return self._mesh_serve.get(key)

    def object_body(self, key: str) -> Optional[Body]:
        """
        The published body behind an object-catalog key, or None if it is not published.

        :param key: Mesh key of the object, as published in the geometry catalog.
        """
        with self._lock:
            return self._bodies.get(key)

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
            overlay_bodies = set(self.overlay_bodies())
            for body in self.world.bodies:
                name = str(body.name)
                if body in overlay_bodies:
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
                objects=[
                    key
                    for key in self._bodies
                    if key != self.configuration.robot_base_key
                ],
                movable=True,
                plan=bool(self.plan_state.nodes),
                chart=bool(self.chart_state.nodes),
                query=self.query_knowledge is not None
                or self._query_attachment is not None,
                sequence_number=self.sequence_number,
                model_version=self._model_revision,
                bundle_signature=bundle_signature,
                robot_parts=(
                    RobotPartAnnotation.of_robot(self.robot)
                    if self.robot is not None
                    else []
                ),
            ).to_payload()

    # %% viewer -> questions about the running demo
    def register_query_source(
        self,
        knowledge: (
            QueryableKnowledge
            | list[QueryableKnowledge]
            | Callable[[], QueryableKnowledge | list[QueryableKnowledge]]
        ),
        title: str,
        presets: list[Preset] | Callable[[], list[Preset]],
        unlisted_presets: list[Preset] | Callable[[], list[Preset]] | None = None,
    ) -> None:
        """
        Offer the running demo's state to the viewer's queries.

        :param knowledge: Native query knowledge or a factory supplying current scopes.
        :param title: The name shown for this query source.
        :param presets: Visible queries or a factory supplying their current definitions.
        :param unlisted_presets: Additional queries or a factory supplying them for matching.
        """
        with self._query_lock:
            self.query_knowledge = knowledge
            self._query_title = title
            self._query_presets = presets
            self._unlisted_query_presets = unlisted_presets or []
            self._query_revision += 1
        logger.info("live queries answered by '%s'", title)

    @contextmanager
    def _query_scope(self) -> Iterator[list[QueryableKnowledge]]:
        """
        Serialize queries and lock their exposed native worlds through rendering.

        World locks are acquired in identity order before the query lock. A changed
        registration or attachment restarts selection before evaluation begins.

        :yield: The knowledge selected for this complete query operation.
        :raises NoQuerySourceRegistered: When no live query source is available.
        """
        while True:
            with self._query_lock:
                source = self.query_knowledge
                revision = self._query_revision
                attachment = self._query_attachment
                world = self.world if attachment is not None else None
            if source is not None:
                selected = source() if callable(source) else source
                knowledge = (
                    [selected] if isinstance(selected, QueryableKnowledge) else selected
                )
            elif world is not None:
                knowledge = [
                    QueryableKnowledge(
                        scope=QueryScope.CURRENT_STATE,
                        domains=[],
                        extra_names={World.__name__.lower(): world},
                    )
                ]
            else:
                raise NoQuerySourceRegistered()
            worlds = [
                value
                for item in knowledge
                for value in item.extra_names.values()
                if isinstance(value, World)
            ]
            if world is not None:
                worlds.append(world)
            locks = {
                id(item.state.world_lock): item.state.world_lock for item in worlds
            }
            with ExitStack() as stack:
                for identity in sorted(locks):
                    stack.enter_context(locks[identity])
                with self._query_lock:
                    if (
                        revision != self._query_revision
                        or source is not self.query_knowledge
                        or attachment != self._query_attachment
                        or (world is not None and world is not self.world)
                    ):
                        continue
                    yield knowledge
                    return

    def query_title(self) -> str:
        """
        Name the live source supplying query answers.

        :return: The selected source's display title.
        :raises NoQuerySourceRegistered: When no live query source is available.
        """
        with self._query_scope() as knowledge:
            if self.query_knowledge is not None:
                return self._query_title
            world = knowledge[0].extra_names[World.__name__.lower()]
            return world.name or type(world).__name__

    def query_presets(self) -> List[Preset]:
        """
        List the selected source's visible presets with their English wording.

        :return: Presets in the source's display order.
        :raises NoQuerySourceRegistered: When no live query source is available.
        :raises UnknownQueryScope: When a preset requests unavailable knowledge.
        """
        with self._query_scope() as knowledge:
            return self._worded_presets(knowledge)

    def _worded_presets(self, knowledge: list[QueryableKnowledge]) -> list[Preset]:
        """
        Add English wording to each visible preset in its declared query scope.

        Hold :meth:`_query_scope` while preparing the presets.

        :param knowledge: The knowledge selected for this operation.
        :return: Worded presets in the source's display order.
        :raises UnknownQueryScope: When a preset requests unavailable knowledge.
        """
        if self.query_knowledge is not None:
            presets = (
                self._query_presets()
                if callable(self._query_presets)
                else self._query_presets
            )
        else:
            name = World.__name__.lower()
            presets = Preset.of_world(knowledge[0].extra_names[name], name)
        return [
            preset.worded(self._scope_runner(knowledge, preset.scope))
            for preset in presets
        ]

    def match_question(self, text: str) -> QuestionMatchResult:
        """
        Match a natural-language question to the selected source's presets.

        Visible presets match their labels and English wording; unlisted presets
        match their labels.

        :param text: The question as asked, in natural language.
        :return: The matching preset, if any, and the closest wording's similarity.
        :raises NoQuerySourceRegistered: When no live query source is available.
        :raises UnknownQueryScope: When a preset requests unavailable knowledge.
        """
        with self._query_scope() as knowledge:
            unlisted = (
                self._unlisted_query_presets()
                if callable(self._unlisted_query_presets)
                else self._unlisted_query_presets
            )
            presets = self._worded_presets(knowledge) + unlisted
            return QuestionMatcher(presets).match(text)

    def query_scopes(self) -> List[QueryScope]:
        """
        List the bodies of knowledge available from the selected source.

        :return: Query scopes in the order declared by the source.
        :raises NoQuerySourceRegistered: When no live query source is available.
        """
        with self._query_scope() as knowledge:
            return [item.scope for item in knowledge]

    def query_variables(
        self, scope: QueryScope = QueryScope.CURRENT_STATE
    ) -> List[str]:
        """
        List ready-made query variables or the scope's directly exposed values.

        :param scope: The body of knowledge the names belong to.
        :return: Domain names, or exposed value names when no domains are declared.
        :raises NoQuerySourceRegistered: When no live query source is available.
        :raises UnknownQueryScope: When the source does not offer this scope.
        """
        with self._query_scope() as knowledge:
            selected = self._queryable_knowledge(knowledge, scope)
            return (
                [domain.name for domain in selected.domains]
                if selected.domains
                else list(selected.extra_names)
            )

    def query_vocabulary(
        self, scope: QueryScope = QueryScope.CURRENT_STATE
    ) -> QueryVocabulary:
        """
        Describe the names available to queries within one scope.

        :param scope: The body of knowledge the names belong to.
        :return: Vocabulary for the scope's domains, extra names and workspace classes.
        :raises NoQuerySourceRegistered: When no live query source is available.
        :raises UnknownQueryScope: When the source does not offer this scope.
        """
        with self._query_scope() as knowledge:
            return self._scope_runner(knowledge, scope).vocabulary()

    def _queryable_knowledge(
        self, knowledge: list[QueryableKnowledge], scope: QueryScope
    ) -> QueryableKnowledge:
        """
        Select the knowledge offered by a source for one query scope.

        :param knowledge: The knowledge selected for this operation.
        :param scope: The body of knowledge being asked.
        :return: Knowledge containing the scope's domains and evaluation context.
        :raises UnknownQueryScope: When the source does not offer this scope.
        """
        for item in knowledge:
            if item.scope is scope:
                return item
        raise UnknownQueryScope(name=scope.value)

    def highlightable_ids(self) -> FrozenSet[str]:
        """
        Ids the viewer can light up: every published object, by key and by display id.

        An answer value naming one of these glows in the scene, whatever the query
        asked for (see :attr:`~cramera.knowledge.query_runner.RowRenderer.highlightable_ids`).
        """
        keys = self.object_keys()
        return frozenset(keys) | frozenset(Path(key).stem for key in keys)

    def run_query(
        self, code: str | Evaluable, scope: QueryScope = QueryScope.CURRENT_STATE
    ) -> RenderResult:
        """
        Evaluate and render an EQL query within the selected source's read scope.

        :param code: The EQL query source or an already constructed native expression.
        :param scope: The body of knowledge to query.
        :return: The rendered query answer.
        :raises NoQuerySourceRegistered: When no live query source is available.
        :raises UnknownQueryScope: When the source does not offer this scope.
        """
        with self._query_scope() as knowledge:
            result = self._scope_runner(knowledge, scope).run(code)
            result.highlight = self._resolve_highlights(result.highlight)
            return result

    def _scope_runner(
        self, knowledge: list[QueryableKnowledge], scope: QueryScope
    ) -> EqlQueryRunner:
        """
        Create a query runner over the selected source's knowledge of one scope.

        Hold :meth:`_query_scope` until the answer has been rendered.

        :param knowledge: The knowledge selected for this operation.
        :param scope: The body of knowledge being asked.
        :return: A runner configured with the scope's knowledge and scene highlights.
        :raises UnknownQueryScope: When the source does not offer this scope.
        """
        selected = self._queryable_knowledge(knowledge, scope)
        return EqlQueryRunner(
            domains=selected.domains,
            extra_names=selected.extra_names,
            evaluation=selected.evaluation,
            highlightable_ids=self.highlightable_ids(),
        )

    # %% viewer -> world

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
        self._kinematic_connections = list(world.connections)
        self._connections = self._actuated_connections(self._kinematic_connections)
        bodies: Dict[str, Body] = {}
        if self.robot is not None:
            bodies[self.configuration.robot_base_key] = self.robot.root
        try:
            bodies.update(self._discover_overlay_bodies())
        except Exception as error:
            # boundary guard: the world is mid-modification (a body is being spawned
            # or removed) and iterating it is not safe. Keep the previous catalog
            # rather than publishing an empty one, which would make the viewer hide
            # every object it already shows.
            logger.debug("body scan skipped this bind: %s", error)
            for key, body in self._bodies.items():
                bodies.setdefault(key, body)
        self.publish_bodies(bodies)

    def overlay_bodies(self) -> List[Body]:
        """Return the independent objects currently published by this session."""
        with self._lock:
            return [
                body
                for key, body in self._bodies.items()
                if key != self.configuration.robot_base_key
            ]

    def _discover_overlay_bodies(self) -> Dict[str, Body]:
        """Discover movable objects and retain their identity through attachments."""
        return {
            str(body.name).split("/")[-1]: body
            for body in WorldObjects(self.world, self.robot).overlay_bodies(
                self.overlay_bodies()
            )
        }

    def _body_shapes(self, body: Body) -> ShapeCollection:
        """
        Select visual geometry, collision geometry, or a native placeholder box.

        :param body: The body whose shapes are read.
        :return: The nonempty collection to render and record.
        """
        for shape_collection in (body.visual, body.collision):
            if shape_collection.shapes:
                return shape_collection
        return ShapeCollection(
            shapes=[Box(scale=Scale(*self.configuration.default_object_size))]
        )

    @staticmethod
    def _actuated_connections(
        connections: List[Connection],
    ) -> List[ActiveConnection1DOF]:
        """
        All 1-DOF connections — the joints published as trajectory frames.

        :param connections: The world's connections to pick the actuated ones from.
        """
        return [
            connection
            for connection in connections
            if isinstance(connection, ActiveConnection1DOF)
        ]

    def _build_object_metadata(self, bodies: Dict[str, Body]) -> None:
        """
        Rebuild the geometry catalog the viewer spawns live objects from.

        Retain native geometry and register its mesh files so new objects can appear
        mid-run.

        :param bodies: The current published bodies, keyed by mesh key.
        """
        catalog: List[ObjectCatalogEntry] = []
        serve: Dict[str, str] = {}
        palette = ObjectPalette()
        for index, (key, body) in enumerate(
            item
            for item in bodies.items()
            if item[0] != self.configuration.robot_base_key
        ):
            entry = ObjectCatalogEntry(
                key=key,
                shapes=self._body_shapes(body),
                fallback_color=palette.color_for(index),
            )
            catalog.append(entry)
            for shape_index, shape in enumerate(entry.shapes):
                mesh_file = served_mesh_file(shape)
                if mesh_file is not None:
                    serve[entry.mesh_key(shape_index)] = mesh_file
        with self._lock:
            self._mesh_serve = serve
            self.object_metadata = catalog

    # %% world snapshot
    def snapshot(self) -> None:
        """
        Publish the world's joints, base pose and object poses.

        Runs on the simulation thread; rebinds the world periodically so mid-run spawns
        show up.
        """
        if self.world is None:
            return
        if (
            time.time() - self._last_bind_time
            > self.configuration.rebind_interval_seconds
        ):
            self.bind()
        frames = {
            str(connection.name): round(float(connection.position), POSE_PRECISION)
            for connection in self._connections
        }
        base_pose: Optional[List[float]] = None
        object_poses: Dict[str, List[float]] = {}
        for name, body in self._bodies.items():
            if name == self.configuration.robot_base_key:
                base_pose = rounded_pose(body)
            else:
                object_poses[name] = rounded_pose(body)
        self._refresh_marker_state()
        transforms = self._transforms.observe(
            self._kinematic_connections, self.world, time.monotonic()
        )
        with self._lock:
            self.transform_state = transforms
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
            payload = self.state.to_payload()
        return payload

    def get_transforms(self) -> Dict[str, Any]:
        """
        The newest transform graph, aged as of now (safe to call from HTTP threads).
        """
        with self._lock:
            return self.transform_state.to_payload(time.monotonic())

    # %% plan tree
    def snapshot_plan(self) -> None:
        """
        Publish the current native lifecycle state of every plan node.
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
    ) -> None:
        """
        Serialize one plan node and its subtree.

        :param node: The plan node to serialize.
        :param parent_id: Id of the node's parent entry, or None for the root.
        :param nodes: Output list every serialized entry is appended to.
        :param order: Output list every serialized node id is appended to, in
            traversal order, to build the tree's signature.
        """
        node_id = "plan_node_%d" % id(node)
        designator = node.designator if isinstance(node, DesignatorNode) else None
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
            status=node.status,
            derived=False,
        )
        self._add_designator_metadata(entry, designator)
        nodes.append(entry)
        order.append(node_id)

        for child in node.children:
            self._serialize_plan_node(child, node_id, nodes, order)

    def _add_designator_metadata(
        self, entry: PlanNodeEntry, designator: Designator | None
    ) -> None:
        """
        Add arm and target-object info from a node's designator, if any.

        :param entry: The serialized entry to fill in, mutated in place.
        :param designator: The node's designator, or None.
        """
        if designator is None:
            return
        arm: Arms | None = designator.designator_parameter.get(DesignatorParameter.ARM)
        if arm is not None:
            entry.arm = arm.name
        target = self._designator_target(designator)
        if target:
            entry.target = target

    def _designator_target(self, designator: Designator) -> Optional[str]:
        """
        Published key of the object a designator refers to, if any.

        Matched by basename, because designators name world entities with their full
        prefixed name while some objects are published under a basename key.

        :param designator: The designator to search for a world-entity reference.
        """
        keys_by_basename = {key.split("/")[-1]: key for key in self._bodies}
        for value in designator.designator_parameter.values():
            if not isinstance(value, WorldEntity):
                continue
            basename = value.name.name
            if basename in keys_by_basename:
                return keys_by_basename[basename]
        return None

    def running_step(self) -> Optional[str]:
        """
        Label of the action the plan is performing right now, or None between actions.

        The deepest running action wins: a ``Transport`` that is performing its
        ``Pickup`` is reported as the pickup, which is the step a replay of this moment
        should be labelled with.
        """
        with self._lock:
            running = [
                entry
                for entry in self.plan_state.nodes
                if entry.status == LifeCycleValues.RUNNING
                and entry.group is PlanNodeGroup.ACTION
            ]
        return running[-1].label if running else None

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

        Publish a fresh snapshot only when the chart changes.

        :param chart: The motion statechart the executor is currently ticking, if any.
        """
        self._chart_observer.title = self._chart_title
        snapshot = self._chart_observer.change(chart)
        if snapshot is None:
            return
        with self._lock:
            self.chart_state = snapshot

    def executing_statechart(self) -> Optional[ChartSnapshot]:
        """
        The statechart the executor is currently ticking, or None while no motion runs.
        """
        with self._lock:
            chart = self.chart_state
        return chart if chart.nodes else None

    def get_chart(self) -> Dict[str, Any]:
        """
        The newest statechart snapshot (safe to call from HTTP threads).
        """
        with self._lock:
            chart = self.chart_state
        payload = asdict(chart)
        payload["edges"] = [edge.to_payload() for edge in chart.edges]
        return payload
