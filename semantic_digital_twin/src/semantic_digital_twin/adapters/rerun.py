# Rerun visualization adapter. Requires the rerun-sdk dependency; does not require ROS.
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import rerun
from typing_extensions import Optional

from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.world import World


class RerunMode(StrEnum):
    """
    Where the Rerun recording stream sends its data.
    """

    SPAWN = "spawn"
    """
    Spawn and stream to a local Rerun viewer.
    """

    CONNECT = "connect"
    """
    Stream to an already-running viewer over gRPC (uses ``target`` URL).
    """

    SAVE = "save"
    """
    Record to an ``.rrd`` file with no viewer (uses ``target`` path).
    """
    NONE = "none"
    """
    Do not attach an output; the caller manages the recording's output.
    """


@dataclass(eq=False)
class RerunModelCallback(ModelChangeCallback):
    """
    Logs the world's static geometry, re-logging it whenever the model changes.

    Separate from the state-change callback so geometry is re-sent only
    on model changes, not on every state update.
    """

    recording: rerun.RecordingStream = field(kw_only=True)
    """
    The recording stream geometry is logged to.
    """

    root_entity_path: str = field(default="world", kw_only=True)
    """
    Entity path under which the kinematic tree is logged.
    """

    def on_model_change(self, **kwargs) -> None:
        self._log_model(self._world)

    def _log_model(self, world: World) -> None:
        """
        Log every body's static visual geometry to Rerun.

        Each shape's mesh is logged as a static :class:`rerun.Mesh3D`
        and its ``origin`` as a static :class:`rerun.Transform3D` on the
        same entity, so only the per-body transforms need re-logging on
        a state change.

        :param world: The world whose geometry is logged.
        """
        rerun.log(
            self.root_entity_path,
            rerun.ViewCoordinates.RIGHT_HAND_Z_UP,
            static=True,
            recording=self.recording,
        )
        for body in world.bodies:
            entity_path = f"{self.root_entity_path}/{body.name.name}"
            shapes = body.visual.shapes if body.visual.shapes else body.collision.shapes
            for index, shape in enumerate(shapes):
                visual_path = f"{entity_path}/visual_{index}"
                origin = shape.origin.to_np()
                rerun.log(
                    visual_path,
                    rerun.Transform3D(
                        translation=origin[:3, 3],
                        mat3x3=origin[:3, :3],
                    ),
                    static=True,
                    recording=self.recording,
                )
                mesh = shape.mesh.copy()
                if hasattr(mesh.visual, "to_color"):
                    mesh.visual = mesh.visual.to_color()
                rerun.log(
                    visual_path,
                    rerun.Mesh3D(
                        vertex_positions=mesh.vertices,
                        triangle_indices=mesh.faces,
                        vertex_normals=mesh.vertex_normals,
                        vertex_colors=mesh.visual.vertex_colors,
                    ),
                    static=True,
                    recording=self.recording,
                )


@dataclass(eq=False)
class RerunAdapter(StateChangeCallback):
    """
    Logs a world to Rerun and keeps the recording in sync as the world changes.

    The state-change callback: re-logs the per-body transforms on every state change
    and owns a :class:`RerunModelCallback` for the geometry. ``mode`` selects the
    output (see :class:`RerunMode`). On construction it configures the output and
    logs the initial geometry and state.
    """

    root_entity_path: str = "world"
    """
    Entity path under which the kinematic tree is logged.
    """

    application_id: str = "test"
    """
    Rerun application id for the recording.
    """

    mode: RerunMode = field(default=RerunMode.SPAWN, kw_only=True)
    """
    Where the recording sends its data.
    """

    target: Optional[str] = field(default=None, kw_only=True)
    """
    GRPC URL for ``CONNECT`` or file path for ``SAVE``.
    """

    timeline: str = field(default="state_version", kw_only=True)
    """
    Name of the Rerun timeline driven by the world state version.
    """

    state_history: bool = field(default=False, kw_only=True)
    """
    Keep a scrubbable state history (bounded by ``memory_limit``); if
    ``False``, keep only the current state.
    """

    memory_limit: str = field(default="10%", kw_only=True)
    """
    Spawned-viewer memory budget (e.g. ``"2GB"``); oldest data is dropped past
    it.

    Only used by the ``SPAWN`` mode.
    """

    recording: rerun.RecordingStream = field(init=False)
    """
    The Rerun recording stream all data is logged to.
    """
    model_cb: RerunModelCallback = field(init=False)
    """
    The owned callback that logs and re-logs geometry on model changes.
    """

    def __post_init__(self) -> None:
        self.recording = rerun.RecordingStream(self.application_id)
        match self.mode:
            case RerunMode.SPAWN:
                self.recording.spawn(memory_limit=self.memory_limit)
            case RerunMode.CONNECT:
                if self.target is None:
                    raise ValueError("RerunMode.CONNECT requires a target gRPC URL.")
                self.recording.connect_grpc(self.target)
            case RerunMode.SAVE:
                if self.target is None:
                    raise ValueError("RerunMode.SAVE requires a target file path.")
                self.recording.save(self.target)
            case RerunMode.NONE:
                pass
        self.model_cb = RerunModelCallback(
            _world=self._world,
            recording=self.recording,
            root_entity_path=self.root_entity_path,
        )
        self.model_cb.notify_model_change()
        super().__post_init__()
        self.on_state_change()

    def _log_state(self, world: World, *, static: bool = False) -> None:
        """
        Log every body's current forward-kinematics transform to Rerun.

        Logged at the same ``world/<body>`` path the geometry is
        parented under, so the two compose. With ``static=True`` the
        transforms overwrite in place (no timeline history), giving a
        constant-memory current-state view.

        :param world: The world whose state is logged.
        :param static: Whether to log without timeline history
            (overwrite in place).
        """
        for body in world.bodies:
            world_transform_body = world.compute_forward_kinematics_np(world.root, body)
            rerun.log(
                f"{self.root_entity_path}/{body.name.name}",
                rerun.Transform3D(
                    translation=world_transform_body[:3, 3],
                    mat3x3=world_transform_body[:3, :3],
                ),
                static=static,
                recording=self.recording,
            )

    def on_state_change(self, **kwargs) -> None:
        if self.state_history:
            rerun.set_time(
                self.timeline,
                sequence=self._world.state.version,
                recording=self.recording,
            )
            self._log_state(self._world)
        else:
            self._log_state(self._world, static=True)

    def stop(self) -> None:
        """
        Detach the callbacks from the world and flush pending data to the sink.
        """
        super().stop()
        self.model_cb.stop()
        self.recording.flush()

    @staticmethod
    def read_recording_entities(path: str, dataset_name: str = "semdt") -> set[str]:
        """
        Return the entity paths recorded in an ``.rrd`` file.

        Reads back through Rerun's in-process server / DataFusion
        reader. Only the schema (the logged entity paths) is recovered
        -- cell values (geometry, transforms) are not read back.
        Intended for verifying what was recorded.

        :param path: Path to the ``.rrd`` file to inspect.
        :param dataset_name: Handle the recording is registered under
            while reading.
        :return: The set of entity paths present in the recording.
        """
        with rerun.server.Server(datasets={dataset_name: [path]}) as server:
            reader = server.client().get_dataset(dataset_name).reader(None)
            columns = reader.schema().names
        return {name.split(":", 1)[0] for name in columns}
