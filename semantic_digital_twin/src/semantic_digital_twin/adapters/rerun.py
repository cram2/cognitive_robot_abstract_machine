# Rerun visualization adapter. Requires the rerun-sdk dependency; does not require ROS.
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import rerun
from typing_extensions import Optional

from krrood.symbolic_math.symbolic_math import (
    CompiledFunction,
    Matrix,
    VariableParameters,
)
from semantic_digital_twin.callbacks.callback import (
    ModelChangeCallback,
    StateChangeCallback,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


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
    Logs the world's static geometry and compiles the body forward kinematics
    on every model change.
    """

    recording: rerun.RecordingStream = field(kw_only=True)
    """
    The recording stream geometry is logged to.
    """

    root_entity_path: str = field(default="world", kw_only=True)
    """
    Entity path under which the kinematic tree is logged.
    """

    compiled_body_fks: CompiledFunction = field(init=False, repr=False)
    """
    Stacked forward kinematics of all bodies, evaluated in one call.

    The body at index ``i`` in ``world.bodies`` occupies rows ``i * 4`` to
    ``i * 4 + 4``.
    """

    def on_model_change(self, **kwargs) -> None:
        self._log_model()
        self._compile_body_fks()

    def _compile_body_fks(self) -> None:
        """
        Compile the stacked forward kinematics of all bodies into one function.
        """
        bodies = self._world.bodies
        if not bodies:
            return
        body_fks = [
            (
                HomogeneousTransformationMatrix()
                if body == self._world.root
                else self._world.compose_forward_kinematics_expression(
                    self._world.root, body
                )
            )
            for body in bodies
        ]
        stacked_body_fks = Matrix.vstack(body_fks)
        self.compiled_body_fks = stacked_body_fks.compile(
            parameters=VariableParameters.from_lists(
                self._world.state.position_float_variables
            )
        )
        if not stacked_body_fks.is_constant():
            self.compiled_body_fks.bind_args_to_memory_view(
                0, self._world.state.positions
            )

    def compute(self) -> np.ndarray:
        """
        Evaluate the stacked forward kinematics of all bodies.
        """
        return self.compiled_body_fks.evaluate()

    def _log_model(self) -> None:
        """
        Log every body's static visual geometry to Rerun.
        """
        rerun.log(
            self.root_entity_path,
            rerun.ViewCoordinates.RIGHT_HAND_Z_UP,
            static=True,
            recording=self.recording,
        )
        for body in self._world.bodies:
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
    Logs a world to Rerun and keeps the recording in sync with its state.
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

    def _log_state(self, static: bool = False) -> None:
        """
        Log the current world-relative transform of every body to Rerun.

        :param static: Whether to overwrite in place without timeline history.
        """
        bodies = self._world.bodies
        if not bodies:
            return
        batched_body_fks = self.model_cb.compute()
        for index, body in enumerate(bodies):
            world_transform_body = batched_body_fks[index * 4 : index * 4 + 4]
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
            self._log_state()
        else:
            self._log_state(static=True)

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
