"""
Turn a coraplex demo into a self-contained web-viewer scene.

Runs the demo file *unmodified* under instrumentation and emits a scene bundle
into :func:`cram_viz.paths.scenes_dir`::

    <scenes dir>/<name>/
        scene.json         models, robot parts, objects, segments, targets
        trajectory.json    per-tick joints + robot base + object world poses
        <model>.urdf       package:// resolved & rewritten
        meshes/...         all meshes + textures the scene needs

What the hooks capture while the demo runs:
  - every package:// asset resolution and every URDF/STL the world loads
  - per-tick positions of all movable connections (giskardpy Executor.tick)
  - world pose of the robot base and of every loose (STL) object
  - one segment per executed plan ActionNode, with nesting depth
  - the robot's semantic annotation: base body, arms, end-effector link sets

Usage (the interpreter needs the CRAM stack on it)::

    cram-viz-onboard path/to/demo.py --name pr2_kitchen
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import runpy
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.package_resolver import PackageUriResolver
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from typing_extensions import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
    Sequence,
    TYPE_CHECKING,
)

from cram_viz import get_logger, paths
from cram_viz.body_geometry import BodyExtent
from cram_viz.live.bridge import ROBOT_BASE_KEY
from cram_viz.monkey_patch import MethodPatch
from cram_viz.onboard.bundle_urdf import bundle_urdf
from cram_viz.palette import ObjectPalette

if TYPE_CHECKING:
    from coraplex.plans.executables import Executable
    from coraplex.plans.plan_node import ActionNode
    from giskardpy.executor import Executor
    from semantic_digital_twin.world_description.world_entity import Body

logger = get_logger(__name__)

#: when this process started, so progress lines can show elapsed recording time
_STARTED_AT = time.time()

#: decimal places poses and joint positions are rounded to in the bundle
POSE_PRECISION = 5

#: how many recorded frames pass between progress lines
FRAME_LOG_INTERVAL = 2000

#: upper bound on the plan nodes written into a bundle
MAX_SERIALIZED_PLAN_NODES = 400

#: frame rate assumed when the controller does not report its timestep
FALLBACK_FRAMES_PER_SECOND = 50.0

#: the slowest playback the viewer is given, however hard the recording is downsampled
MINIMUM_FRAMES_PER_SECOND = 10

#: how far below the lowest recorded place pose the place marker is drawn, in metres
PLACE_TARGET_DROP = 0.02

#: half-extent of the draggable place area around the recorded place poses, in metres
PLACE_BOUNDS_MARGIN = 0.55

#: extra room in front of the place area, so a target can be dragged towards the robot
PLACE_BOUNDS_FRONT_MARGIN = 0.65

#: how far beyond the objects' spawn poses they may be dragged, in metres
DRAG_BOUNDS_MARGIN_X = 0.35
DRAG_BOUNDS_MARGIN_Y = 0.6

#: how many of a model's links are probed to find its prefix in the composed world
PREFIX_PROBE_LINKS = 12

#: frame count a bundle is downsampled towards when no explicit step is given
TARGET_BUNDLE_FRAMES = 1500

#: how many unresolved assets the summary lists before truncating
MISSING_ASSETS_LOGGED = 10


@runtime_checkable
class NamesAWorldEntity(Protocol):
    """
    Anything carrying a world-entity name, such as a body a designator refers to.
    """

    name: Any


@runtime_checkable
class DescribesAnAction(Protocol):
    """
    A plan node carrying the designator that describes what it does.
    """

    designator: Any


@runtime_checkable
class ReportsAStatus(Protocol):
    """
    A plan node carrying its own execution status.
    """

    status: Any


@runtime_checkable
class HasChildren(Protocol):
    """
    A plan node with a child list to descend into.
    """

    children: Any


@runtime_checkable
class HasAParent(Protocol):
    """
    A plan node that can be walked upwards to its tree's root.
    """

    parent: Any


@runtime_checkable
class HasBodies(Protocol):
    """
    A robot part exposing the bodies it is made of.
    """

    bodies: Any


@runtime_checkable
class HasAnEndEffector(Protocol):
    """
    A manipulator carrying an end effector, which not every arm annotation does.
    """

    end_effector: Any


def log(*parts: object) -> None:
    """
    Emit a progress line prefixed with the elapsed recording time.

    :param parts: Values joined by a space to form the log message.
    """
    logger.info(
        "[%6.1fs] %s",
        time.time() - _STARTED_AT,
        " ".join(str(part) for part in parts),
    )


# %% recorder
@dataclass
class Recorder:
    """
    Records one demo run: assets, per-tick motion and the executed plan.

    .. note:: ``giskardpy`` and ``coraplex`` are only imported inside the
       ``install_*`` hook methods that need them. Unlike ``semantic_digital_twin``,
       which this module already imports at the top, they are not required to parse
       a finished recording into a scene bundle, and this module is imported by the
       ``cram-viz-onboard`` console script, which has to stay importable without
       them. This is one of the documented exceptions to the imports-at-top rule.
    """

    resolutions: Dict[str, str] = field(default_factory=dict)
    """
    ``package://`` URI to the path it resolved to while the demo ran.
    """

    urdf_sources: List[str] = field(default_factory=list)
    """
    URDF/xacro files the world was built from, in load order.
    """

    mesh_sources: List[str] = field(default_factory=list)
    """
    Mesh files of the loose objects, in load order.
    """

    frames: List[Dict[str, float]] = field(default_factory=list)
    """
    Per-tick joint positions, keyed by prefixed connection name.
    """

    base_frames: List[Optional[List[float]]] = field(default_factory=list)
    """
    Per-tick robot base pose as ``[x, y, z, qx, qy, qz, qw]``.
    """

    object_frames: List[Dict[str, List[float]]] = field(default_factory=list)
    """
    Per-tick world pose of every tracked object, keyed by mesh basename.
    """

    actions: List[Dict[str, Any]] = field(default_factory=list)
    """
    One entry per parsed action: its class, arm and target object.
    """

    plan_nodes: List[Any] = field(default_factory=list)
    """
    The plan nodes the demo parsed, used to serialize the executed plan tree.
    """

    world: Optional[Any] = None
    """
    The executing world, captured on the first tick.
    """

    robot: Optional[Any] = None
    """
    The robot annotation of :attr:`world`.
    """

    control_dt: Optional[float] = None
    """
    The controller's timestep, from which the recording's frame rate follows.
    """

    _connections: Optional[List[Any]] = field(default=None)
    """
    Connections whose position is recorded; None until the first tick binds.
    """

    _bodies: Optional[Dict[str, Any]] = field(default=None)
    """
    Recorded bodies by mesh basename, plus :data:`ROBOT_BASE_KEY`.
    """

    # %% asset hooks
    def install_asset_hooks(self) -> None:
        """
        Record every asset resolution so the bundler can copy the files.
        """
        MethodPatch(PackageUriResolver, "resolve").install(self._remember_resolution)
        MethodPatch(URDFParser, "from_file").install(self._remember_urdf_source)
        MethodPatch(STLParser, "__init__").install(self._remember_mesh_source)

    def _remember_resolution(
        self,
        original: Callable[[PackageUriResolver, str], str],
        resolver: PackageUriResolver,
        uri: str,
    ) -> str:
        """
        Resolve as usual, but remember the uri -> path mapping.

        :param original: The real, unpatched ``PackageUriResolver.resolve`` bound
            method.
        :param resolver: The resolver resolving the URI.
        :param uri: The ``package://`` URI being resolved.
        """
        resolved = original(resolver, uri)
        self.resolutions[uri] = resolved
        return resolved

    def _remember_urdf_source(
        self,
        original: Callable[..., URDFParser],
        cls: type,
        file_path: str,
        **kwargs: Any,
    ) -> URDFParser:
        """
        Parse as usual, but remember this URDF/xacro source file.

        :param original: The real, unpatched ``URDFParser.from_file`` classmethod.
        :param cls: The ``URDFParser`` class the method is bound to.
        :param file_path: Path of the URDF/xacro source file being parsed.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        if file_path not in self.urdf_sources:
            self.urdf_sources.append(file_path)
        return original(cls, file_path, **kwargs)

    def _remember_mesh_source(
        self,
        original: Callable[..., None],
        stl_parser: STLParser,
        file_path: str,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize as usual, but remember this loose object's mesh file.

        :param original: The real, unpatched ``STLParser.__init__`` bound method.
        :param stl_parser: The parser being initialized.
        :param file_path: Path of the object's mesh file.
        :param args: Positional arguments forwarded to the wrapped call.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        if file_path not in self.mesh_sources:
            self.mesh_sources.append(file_path)
        return original(stl_parser, file_path, *args, **kwargs)

    # %% trajectory hook
    def install_tick_hook(self) -> None:
        """
        Wrap Executor.tick so every simulation step is snapshotted.
        """
        from giskardpy.executor import Executor

        MethodPatch(Executor, "tick").install(self._record_tick)

    def _record_tick(
        self,
        original: Callable[..., None],
        executor: Executor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Run the real tick, then record its resulting world state.

        :param original: The real, unpatched ``Executor.tick`` bound method.
        :param executor: The executor whose tick is being run.
        :param args: Positional arguments forwarded to the wrapped call.
        :param kwargs: Keyword arguments forwarded to the wrapped call.
        """
        result = original(executor, *args, **kwargs)
        self.record_frame(executor)
        return result

    def bind_to_executor(self, executor: Executor) -> None:
        """
        Locate the world, robot and recordable objects of a running executor.

        Deferred until the first tick, because the world does not exist any earlier.

        :param executor: The executor whose world is bound to.
        """
        from semantic_digital_twin.robots.robot_parts import AbstractRobot

        self.world = executor.context.world
        self.control_dt = executor.context.qp_controller_config.control_dt
        robots = self.world.get_semantic_annotations_by_type(AbstractRobot)
        self.robot = robots[0] if robots else None
        self._bodies = {}
        if self.robot is not None:
            self._bodies[ROBOT_BASE_KEY] = self.robot.root
        for mesh_source in self.mesh_sources:
            name = os.path.basename(mesh_source)
            body = self.world.get_body_by_name(name)
            if body is not None:
                self._bodies[name] = body
        self._connections = [
            connection
            for connection in self.world.connections
            if isinstance(connection, ActiveConnection1DOF)
        ]
        log(
            "bound: robot=%s, %d movable connections, objects=%s"
            % (
                type(self.robot).__name__ if self.robot else None,
                len(self._connections),
                [key for key in self._bodies if key != ROBOT_BASE_KEY],
            )
        )

    @staticmethod
    def _pose_as_position_quaternion(body: Body) -> List[float]:
        """
        A body's world pose as ``[x, y, z, qx, qy, qz, qw]``.

        :param body: The body whose world pose is read.
        """
        return [
            round(value, POSE_PRECISION)
            for value in body.global_pose.to_position_quaternion_list()
        ]

    def record_frame(self, executor: Executor) -> None:
        """
        Append one frame: every movable connection's position, the robot base pose and
        every tracked object's pose.

        :param executor: The executor whose current tick is recorded.
        """
        if self._connections is None:
            self.bind_to_executor(executor)
        self.frames.append(
            {
                str(connection.name): round(float(connection.position), POSE_PRECISION)
                for connection in self._connections
            }
        )
        self.base_frames.append(
            self._pose_as_position_quaternion(self._bodies[ROBOT_BASE_KEY])
            if ROBOT_BASE_KEY in self._bodies
            else None
        )
        self.object_frames.append(
            {
                name: self._pose_as_position_quaternion(body)
                for name, body in self._bodies.items()
                if name != ROBOT_BASE_KEY
            }
        )
        if len(self.frames) % FRAME_LOG_INTERVAL == 0:
            log("... %d frames" % len(self.frames))

    # %% action metadata hook
    # Plans compile all actions into merged motion statecharts, so there is no
    # per-action call boundary at execution time. ``ActionNode.parse`` does fire
    # once per action (in plan order) — the action class, its arm and its target
    # object are recorded there, and the segment *timing* is derived afterwards
    # from the recorded data (object attach/detach + first base motion).
    def _target_of(self, designator: Any) -> Optional[str]:
        """
        The recorded object a designator refers to, matched by mesh basename.

        :param designator: The designator to search for a world-entity reference.
        """
        basenames = {os.path.basename(path) for path in self.mesh_sources}
        for value in vars(designator).values():
            if not isinstance(value, NamesAWorldEntity):
                continue
            basename = str(value.name).split("/")[-1]
            if basename in basenames:
                return basename
        return None

    @staticmethod
    def _arm_of(designator: Any) -> Optional[str]:
        """
        The arm a designator names, whether it calls the field ``arm`` or ``arms``.

        :param designator: The designator to read the arm field from.
        """
        fields = vars(designator)
        arm = fields.get("arm") or fields.get("arms")
        return str(arm) if arm is not None else None

    def install_segment_hook(self) -> None:
        """
        Wrap ActionNode.parse to record each action's class, arm and target as it is
        parsed (in plan order).
        """
        from coraplex.plans.plan_node import ActionNode

        recorder = self
        original_parse = ActionNode.parse

        def parse(node: ActionNode, *args: Any, **kwargs: Any) -> Executable:
            """
            Record this action's designator before letting it parse normally.

            :param node: The action node being parsed.
            :param args: Positional arguments forwarded to the wrapped call.
            :param kwargs: Keyword arguments forwarded to the wrapped call.
            """
            designator = node.designator
            recorder.actions.append(
                {
                    "action": type(designator).__name__,
                    "arm": recorder._arm_of(designator),
                    "target": recorder._target_of(designator),
                }
            )
            recorder.plan_nodes.append(node)
            log(
                "action parsed:",
                recorder.actions[-1]["action"],
                "->",
                recorder.actions[-1]["target"] or "-",
            )
            return original_parse(node, *args, **kwargs)

        ActionNode.parse = parse

    # %% the executed plan tree, serialized from the real PlanNode graph
    def serialize_plans(
        self, max_nodes: int = MAX_SERIALIZED_PLAN_NODES
    ) -> List[Dict[str, Any]]:
        """
        The executed plan trees, deduplicated by root, as nested dicts.

        :param max_nodes: Upper bound on the total node count across all trees; the
            recording stops descending once it is reached.
        """
        roots: List[Any] = []
        seen_roots = set()
        for node in self.plan_nodes:
            root = node
            while isinstance(root, HasAParent) and root.parent is not None:
                root = root.parent
            if id(root) not in seen_roots:
                seen_roots.add(id(root))
                roots.append(root)
        serialized_count = itertools.count()

        def serialize(node: Any) -> Optional[Dict[str, Any]]:
            """
            One plan node and its children as a dict, or None past ``max_nodes``.

            :param node: The plan node to serialize.
            """
            if next(serialized_count) >= max_nodes:
                return None
            designator = (
                node.designator if isinstance(node, DescribesAnAction) else None
            )
            status = node.status if isinstance(node, ReportsAStatus) else None
            entry = {
                "kind": type(node).__name__,
                "label": (
                    type(designator).__name__
                    if designator is not None
                    else type(node).__name__
                ),
                "status": status.name if status is not None else "",
            }
            if designator is not None:
                target = self._target_of(designator)
                if target:
                    entry["target"] = target
                arm = self._arm_of(designator)
                if arm is not None:
                    entry["arm"] = arm
            children = node.children if isinstance(node, HasChildren) else ()
            entry["children"] = [
                child
                for child in (serialize(child) for child in children or ())
                if child
            ]
            return entry

        return [tree for tree in (serialize(root) for root in roots) if tree]


# %% post-processing the recording
#: how far a pose must travel to count as moved at all, in metres
MOVEMENT_TOLERANCE = 0.02

#: how far an object must travel over the whole run to count as transported, in metres
TRANSPORT_TOLERANCE = 0.03

#: how far the robot base must travel to count as having driven off, in metres
BASE_MOTION_TOLERANCE = 0.05


def moved(
    first: Sequence[float],
    second: Sequence[float],
    tolerance: float = MOVEMENT_TOLERANCE,
) -> bool:
    """
    Whether two poses are more than ``tolerance`` apart (planar distance plus height).

    :param first: The pose to compare against ``second``.
    :param second: The pose to compare against ``first``.
    :param tolerance: Tolerance in metres, so sensor jitter does not read as movement.
    """
    return (
        math.hypot(first[0] - second[0], first[1] - second[1])
        + abs(first[2] - second[2])
        > tolerance
    )


def object_windows(recorder: Recorder) -> List[Dict[str, Any]]:
    """
    The attach..detach frame window of every object that travelled overall.

    An object whose first and last pose differ was transported; the window spans from
    the first frame that differs from where it started to just past the last frame that
    differs from where it ended up.

    :param recorder: The finished recording to derive windows from.
    """
    object_frames = recorder.object_frames
    frame_count = len(object_frames)
    windows = []
    for name in object_frames[0]:
        spawn = object_frames[0].get(name)
        final = object_frames[frame_count - 1].get(name)
        if not spawn or not final or not moved(spawn, final, TRANSPORT_TOLERANCE):
            continue
        attach = next(
            (
                index
                for index in range(frame_count)
                if name in object_frames[index]
                and moved(object_frames[index][name], spawn)
            ),
            frame_count - 1,
        )
        detach = (
            next(
                (
                    index
                    for index in range(frame_count - 1, -1, -1)
                    if name in object_frames[index]
                    and moved(object_frames[index][name], final)
                ),
                0,
            )
            + 1
        )
        if attach < detach:
            windows.append(
                {
                    "object": name,
                    "attach": attach,
                    "detach": detach,
                    "place": final[:3],
                }
            )
    windows.sort(key=lambda window: window["attach"])
    return windows


def first_base_motion(recorder: Recorder, before: int) -> int:
    """
    The first frame before ``before`` at which the robot base left its spawn.

    :param recorder: The finished recording to search.
    :param before: Frame index to stop searching before.
    :return: That frame's index, or ``before`` if the base never moved.
    """
    spawn = recorder.base_frames[0]
    for index in range(min(before, len(recorder.base_frames))):
        pose = recorder.base_frames[index]
        if not pose or not spawn:
            continue
        if math.hypot(pose[0] - spawn[0], pose[1] - spawn[1]) > BASE_MOTION_TOLERANCE:
            return index
    return before


def derive_segments(recorder: Recorder) -> List[Dict[str, Any]]:
    """
    Segments = data-driven windows, labelled from the parsed action list.

    :param recorder: The finished recording to derive segments from.
    """
    frame_count = len(recorder.frames)
    transport_windows = object_windows(recorder)
    manipulation_actions = [
        action for action in recorder.actions if action.get("target")
    ]
    leading_actions = []
    for action in recorder.actions:
        if action.get("target"):
            break
        leading_actions.append(action)

    segments = []
    previous_end = 0
    if transport_windows:
        lead_end = first_base_motion(recorder, transport_windows[0]["attach"])
        if lead_end > 10:
            label = (
                leading_actions[0]["action"].replace("Action", "").lower()
                if len(leading_actions) == 1
                else "prepare"
            )
            segments.append(
                {
                    "step": label,
                    "action": ",".join(
                        leading_action["action"] for leading_action in leading_actions
                    )
                    or None,
                    "arm": None,
                    "start": 0,
                    "end": lead_end,
                }
            )
            previous_end = lead_end
    unmatched_actions = list(manipulation_actions)
    for window_index, window in enumerate(transport_windows):
        matching_action = next(
            (
                action
                for action in unmatched_actions
                if action["target"] == window["object"]
            ),
            None,
        )
        if matching_action:
            unmatched_actions.remove(matching_action)
        object_id = os.path.splitext(window["object"])[0]
        verb = (
            matching_action["action"].replace("Action", "").lower()
            if matching_action
            else "move"
        )
        has_next_window = window_index + 1 < len(transport_windows)
        next_attach = (
            transport_windows[window_index + 1]["attach"]
            if has_next_window
            else frame_count - 1
        )
        end = (
            min(
                next_attach,
                window["detach"] + max(10, (next_attach - window["detach"]) // 2),
            )
            if has_next_window
            else frame_count - 1
        )
        segments.append(
            {
                "step": "%s_%s" % (verb, object_id),
                "action": matching_action["action"] if matching_action else None,
                "arm": matching_action["arm"] if matching_action else None,
                "start": previous_end,
                "end": end,
                "picks": object_id,
                "attach": window["attach"],
                "detach": window["detach"],
                "place": window["place"],
            }
        )
        previous_end = end
    if not segments:
        label = (
            recorder.actions[0]["action"].replace("Action", "").lower()
            if len(recorder.actions) == 1
            else "plan"
        )
        segments.append(
            {
                "step": label,
                "action": None,
                "arm": None,
                "start": 0,
                "end": frame_count - 1,
            }
        )
    return segments


def link_set(part: Any) -> List[str]:
    """
    A robot part's link names, stripped of their model-name prefix.

    :param part: The robot part whose link names are read.
    """
    if not isinstance(part, HasBodies):
        return []
    link_names = []
    for body in part.bodies or []:
        name = str(body.name) if isinstance(body, NamesAWorldEntity) else str(body)
        link_names.append(name.split("/", 1)[1] if "/" in name else name)
    return link_names


def build_scene(
    recorder: Recorder, name: str, out_dir: str, step: int
) -> Dict[str, Any]:
    """
    Downsample the recording to every step-th frame (always keeping the last) and
    assemble scene.json + trajectory.json from it.

    :param recorder: The finished recording to build a scene from.
    :param name: Scene name, used as the output folder and in the scene metadata.
    :param out_dir: Directory the scene bundle is written into.
    :param step: Downsampling step; every ``step``-th frame is kept.
    """
    frame_count = len(recorder.frames)
    kept_indices = list(range(0, frame_count, step))
    if kept_indices and kept_indices[-1] != frame_count - 1:
        kept_indices.append(frame_count - 1)
    downsampled_index = {raw_index: kept for kept, raw_index in enumerate(kept_indices)}

    def nearest(raw_index: int) -> int:
        """
        The downsampled index closest to a raw frame index.

        :param raw_index: Frame index in the original, un-downsampled recording.
        """
        return downsampled_index.get(
            raw_index,
            downsampled_index[
                min(downsampled_index, key=lambda kept: abs(kept - raw_index))
            ],
        )

    frames = [recorder.frames[index] for index in kept_indices]
    base = [recorder.base_frames[index] for index in kept_indices]
    object_poses = [recorder.object_frames[index] for index in kept_indices]

    raw_frames_per_second = (
        1.0 / recorder.control_dt if recorder.control_dt else FALLBACK_FRAMES_PER_SECOND
    )
    frames_per_second = max(
        MINIMUM_FRAMES_PER_SECOND, round(raw_frames_per_second / step)
    )

    # %% robot description
    robot = recorder.robot
    root_name = str(robot.root.name)
    prefix = root_name.split("/", 1)[0] if "/" in root_name else ""
    base_body = root_name.split("/", 1)[1] if "/" in root_name else root_name
    parts = {}
    for arm in robot.get_arms():
        arm_links = link_set(arm)
        end_effector = arm.end_effector if isinstance(arm, HasAnEndEffector) else None
        end_effector_links = link_set(end_effector) if end_effector is not None else []
        parts[type(arm).__name__] = sorted(set(arm_links) - set(end_effector_links))
        if end_effector is not None:
            parts[type(end_effector).__name__] = sorted(set(end_effector_links))

    # %% segments: data-derived windows, labelled from the parsed actions
    segments = []
    for raw_segment in derive_segments(recorder):
        segment = dict(raw_segment)
        segment["start"] = nearest(raw_segment["start"])
        segment["end"] = nearest(raw_segment["end"])
        if "attach" in segment:
            segment["attach"] = nearest(raw_segment["attach"])
            segment["detach"] = nearest(raw_segment["detach"])
        segments.append(segment)
    # a scene with two transports of the same object would otherwise name both steps
    # identically, and the viewer keys its playback captions on the step name
    step_counts: Dict[str, int] = {}
    for segment in segments:
        step_counts[segment["step"]] = step_counts.get(segment["step"], 0) + 1
        if step_counts[segment["step"]] > 1:
            segment["step"] = "%s_%d" % (
                segment["step"],
                step_counts[segment["step"]],
            )

    # %% objects
    objects = []
    palette = ObjectPalette()
    for index, source in enumerate(recorder.mesh_sources):
        mesh = os.path.basename(source)
        if mesh not in recorder.object_frames[0]:
            continue
        destination = os.path.join(out_dir, "meshes", "objects", mesh)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy2(source, destination)
        entry = {
            "id": os.path.splitext(mesh)[0],
            "key": mesh,
            "mesh": "meshes/objects/" + mesh,
            "spawn": recorder.object_frames[0][mesh],
            "color": palette.color_for(index),
        }
        # recorded from the world, so the knowledge base does not have to guess it;
        # omitted when the object's shapes report no measurable size
        body = (recorder._bodies or {}).get(mesh)
        extent = BodyExtent.of(body) if body is not None else None
        if extent is not None:
            entry["height"] = round(extent.z, POSE_PRECISION)
        objects.append(entry)

    # %% place target + drag bounds
    places = [segment["place"] for segment in segments if segment.get("place")]
    place_target = None
    if places:
        center_x = sum(place[0] for place in places) / len(places)
        center_y = sum(place[1] for place in places) / len(places)
        lowest_z = min(place[2] for place in places)
        place_target = {
            "pos": [round(center_x, 3), round(center_y, 3)],
            "z": round(lowest_z - PLACE_TARGET_DROP, 3),
            "bounds": {
                "minX": round(center_x - PLACE_BOUNDS_MARGIN, 2),
                "maxX": round(center_x + PLACE_BOUNDS_MARGIN, 2),
                "minY": round(center_y - PLACE_BOUNDS_MARGIN, 2),
                "maxY": round(center_y + PLACE_BOUNDS_FRONT_MARGIN, 2),
            },
        }
    drag_bounds = None
    if objects:
        spawn_x = [entry["spawn"][0] for entry in objects]
        spawn_y = [entry["spawn"][1] for entry in objects]
        drag_bounds = {
            "minX": round(min(spawn_x) - DRAG_BOUNDS_MARGIN_X, 2),
            "maxX": round(max(spawn_x) + DRAG_BOUNDS_MARGIN_X, 2),
            "minY": round(min(spawn_y) - DRAG_BOUNDS_MARGIN_Y, 2),
            "maxY": round(max(spawn_y) + DRAG_BOUNDS_MARGIN_Y, 2),
        }

    # %% bundle the URDF models
    world_body_names = [
        str(body.name) if isinstance(body, NamesAWorldEntity) else ""
        for body in recorder.world.bodies
    ]
    models = []
    missing: List[str] = []
    for source in recorder.urdf_sources:
        base_name = os.path.splitext(os.path.basename(source))[0]
        report = bundle_urdf(source, base_name, out_dir, hints=recorder.resolutions)
        missing += report.missing
        # find this model's prefix in the composed world via one of its links
        model_prefix = ""
        for link in report.links[:PREFIX_PROBE_LINKS]:
            prefixed = next(
                (
                    body_name
                    for body_name in world_body_names
                    if body_name.endswith("/" + link)
                ),
                None,
            )
            if prefixed:
                model_prefix = prefixed.split("/", 1)[0]
                break
        is_robot = base_body in report.links
        models.append(
            {
                "name": base_name,
                "urdf": "%s.urdf" % base_name,
                "prefix": model_prefix,
                "robot": is_robot,
                "links": len(report.links),
                "movableJoints": report.movable_joints,
            }
        )
        log(
            "bundled %-28s prefix=%-12s robot=%s meshes=%d missing=%d"
            % (
                base_name,
                model_prefix or "-",
                is_robot,
                report.meshes_copied,
                len(report.missing),
            )
        )

    scene = {
        "name": name,
        "fps": frames_per_second,
        "trajectory": "trajectory.json",
        "models": models,
        "robot": {
            "name": type(robot).__name__.lower(),
            "prefix": prefix,
            "baseBody": base_body,
            "parts": parts,
        },
        "objects": objects,
        "segments": segments,
        "actions": recorder.actions,
        "planTrees": recorder.serialize_plans(),
        "placeTarget": place_target,
        "dragBounds": drag_bounds,
        "missingAssets": sorted(set(missing)),
    }
    _write_json(Path(out_dir) / "scene.json", scene, indent=1)
    _write_json(
        Path(out_dir) / "trajectory.json",
        {
            "fps": frames_per_second,
            "frames": frames,
            "base": base,
            "objects": object_poses,
        },
    )
    return scene


def _update_scene_index(path: Path, name: str) -> None:
    """
    Register a freshly written scene in the index the viewer reads.

    A missing or unreadable index is rebuilt from scratch; an index that exists but
    lacks the keys is filled in rather than crashing on them.

    :param path: Path of the scene index file.
    :param name: Name of the scene to register.
    """
    index: Dict[str, Any] = {}
    if path.is_file():
        index = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        index = {}
    index.setdefault("scenes", [])
    index.setdefault("default", name)
    if name not in index["scenes"]:
        index["scenes"].append(name)
    _write_json(path, index, indent=1)


def _write_json(path: Path, payload: Any, indent: Optional[int] = None) -> None:
    """
    Write a bundle file, replacing it only once it is complete.

    A bundle is the artifact of a long recording, so a failure part-way through a write
    must not leave a truncated file behind.

    :param path: Destination path of the file.
    :param payload: JSON-serializable content to write.
    :param indent: Indentation passed to :func:`json.dumps`, or None to compact it.
    """
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    temporary.replace(path)


# %% the cram-viz-onboard entry point
def main() -> None:
    """
    ``cram-viz-onboard`` — record one demo run into a scene bundle.
    """
    # force: the demo's own imports configure the root logger before we get here
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("demo", help="path to the coraplex demo .py file")
    parser.add_argument("--name", required=True, help="scene name (output folder)")
    parser.add_argument(
        "--out",
        default=str(paths.scenes_dir()),
        help="scenes directory (default: CRAM_VIZ_SCENES or ~/.cram_viz/scenes)",
    )
    parser.add_argument(
        "--step", type=int, default=0, help="downsample step (0 = auto)"
    )
    args = parser.parse_args()

    try:
        import coraplex  # noqa: F401
    except ModuleNotFoundError:
        sys.exit(
            "The CRAM stack is not importable — run under the action-cram venv:\n"
            "  the workspace venv (uv sync), then: cram-viz-onboard ..."
        )

    recorder = Recorder()
    recorder.install_asset_hooks()
    recorder.install_tick_hook()
    recorder.install_segment_hook()

    demo = os.path.abspath(args.demo)
    log("running demo:", demo)
    sys.path.insert(0, os.path.dirname(demo))
    # make repo-level helper packages (e.g. test.conftest) importable — the
    # demos rely on pytest's rootdir behaviour for that
    candidate = os.path.dirname(demo)
    while candidate != os.path.dirname(candidate):
        if os.path.isdir(os.path.join(candidate, "coraplex")) and os.path.isdir(
            os.path.join(candidate, "test")
        ):
            sys.path.insert(0, candidate)
            log("repo root on sys.path:", candidate)
            break
        candidate = os.path.dirname(candidate)
    runpy.run_path(demo, run_name="__main__")
    log(
        "demo finished: %d raw frames, %d actions"
        % (len(recorder.frames), len(recorder.actions))
    )

    if not recorder.frames:
        sys.exit("No frames captured — did the demo perform a plan?")
    if recorder.robot is None:
        sys.exit("No AbstractRobot semantic annotation found in the world.")

    step = args.step or max(1, len(recorder.frames) // TARGET_BUNDLE_FRAMES)
    out_dir = os.path.join(args.out, args.name)
    os.makedirs(out_dir, exist_ok=True)
    scene = build_scene(recorder, args.name, out_dir, step)
    _update_scene_index(Path(args.out) / "index.json", args.name)

    log("scene '%s' written to %s" % (args.name, out_dir))
    log(
        "  models:  %s"
        % ", ".join(
            "%s%s" % (model["name"], " (robot)" if model["robot"] else "")
            for model in scene["models"]
        )
    )
    log("  objects: %s" % ", ".join(entry["id"] for entry in scene["objects"]))
    log("  segments: %s" % " → ".join(entry["step"] for entry in scene["segments"]))
    if scene["missingAssets"]:
        log("  warning — %d missing assets:" % len(scene["missingAssets"]))
        for asset in scene["missingAssets"][:MISSING_ASSETS_LOGGED]:
            log("   ", asset)
    sys.stdout.flush()
    os._exit(0)  # don't hang on non-daemon ROS/viz threads the demo started


if __name__ == "__main__":
    main()
