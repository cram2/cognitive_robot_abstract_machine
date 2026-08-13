# Rerun visualization adapter. Requires the rerun-sdk dependency; does not require ROS.
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
import rerun
import rerun.blueprint
import trimesh
from PIL import Image
from typing_extensions import TYPE_CHECKING, Optional

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

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.geometry import Shape
    from semantic_digital_twin.world_description.world_entity import Body


def body_entity_path(root_entity_path: str, body: Body) -> str:
    """
    The entity path a body is logged under.

    Uses the body's full prefixed name, so equally named bodies from different
    merged worlds stay distinct and bodies sharing a prefix group together in
    the viewer's entity tree.
    """
    return f"{root_entity_path}/{body.name}"


def _material_image(
    material: Optional[trimesh.visual.material.Material],
) -> Optional[Image.Image]:
    """
    :return: The albedo image of a trimesh material, or ``None`` when it has none.
    """
    if isinstance(material, trimesh.visual.material.SimpleMaterial):
        return material.image
    if isinstance(material, trimesh.visual.material.PBRMaterial):
        return material.baseColorTexture
    return None


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
            entity_path = body_entity_path(self.root_entity_path, body)
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
                rerun.log(
                    visual_path,
                    self.mesh_archetype(shape),
                    static=True,
                    recording=self.recording,
                )

    @staticmethod
    def mesh_archetype(shape: Shape) -> rerun.Mesh3D:
        """
        Build the Rerun mesh archetype for a shape.

        A textured mesh keeps its UV coordinates and albedo texture, a colored
        mesh its per-vertex colors, and a colorless mesh is tinted with the
        shape's color.
        """
        mesh = shape.mesh.copy()
        if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
            image = _material_image(mesh.visual.material)
            if mesh.visual.uv is not None and len(mesh.visual.uv) and image is not None:
                # Trimesh UVs have a bottom-left origin, Rerun expects top-left.
                texture_coordinates = np.asarray(
                    mesh.visual.uv, dtype=np.float32
                ).copy()
                texture_coordinates[:, 1] = 1.0 - texture_coordinates[:, 1]
                return rerun.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.faces,
                    vertex_normals=mesh.vertex_normals,
                    vertex_texcoords=texture_coordinates,
                    albedo_texture=np.asarray(image.convert("RGBA")),
                )
            mesh.visual = mesh.visual.to_color()
        if mesh.visual.kind is None:
            return rerun.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                albedo_factor=shape.color.to_rgba(),
            )
        return rerun.Mesh3D(
            vertex_positions=mesh.vertices,
            triangle_indices=mesh.faces,
            vertex_normals=mesh.vertex_normals,
            vertex_colors=mesh.visual.vertex_colors,
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

    state_log_stride: int = field(default=1, kw_only=True)
    """
    Log only every N-th state version when keeping history.

    Keyed on the world state version, so which states are kept is
    deterministic. :meth:`log_current_state` bypasses the stride.
    """

    send_default_blueprint: bool = field(default=True, kw_only=True)
    """
    Send :meth:`default_blueprint` to the viewer when one attaches.
    """

    event_log_entity_path: str = field(default="plan", kw_only=True)
    """
    Entity path whose text logs the default layout shows beside the 3D scene.
    """

    last_logged_version: Optional[int] = field(init=False, default=None)
    """
    The world state version most recently logged, or ``None`` before the first log.
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
        if self.send_default_blueprint and self.mode in (
            RerunMode.SPAWN,
            RerunMode.CONNECT,
        ):
            self.recording.send_blueprint(self.default_blueprint())
        self.model_cb = RerunModelCallback(
            _world=self._world,
            recording=self.recording,
            root_entity_path=self.root_entity_path,
        )
        self.model_cb.notify_model_change()
        super().__post_init__()
        self.log_current_state()

    def default_blueprint(self) -> rerun.blueprint.Blueprint:
        """
        The layout shown when a viewer attaches: the 3D scene beside a text log
        of plan events, with the timeline panel collapsed.
        """
        return rerun.blueprint.Blueprint(
            rerun.blueprint.Horizontal(
                rerun.blueprint.Spatial3DView(
                    origin=self.root_entity_path,
                    background=[27, 27, 31],
                ),
                rerun.blueprint.TextLogView(origin=self.event_log_entity_path),
                column_shares=[4, 1],
            ),
            rerun.blueprint.TimePanel(state=rerun.blueprint.PanelState.Collapsed),
        )

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
                body_entity_path(self.root_entity_path, body),
                rerun.Transform3D(
                    translation=world_transform_body[:3, 3],
                    mat3x3=world_transform_body[:3, :3],
                ),
                static=static,
                recording=self.recording,
            )

    def on_state_change(self, **kwargs) -> None:
        if (
            self.state_history
            and self._world.state.version % self.state_log_stride != 0
        ):
            return
        self.log_current_state()

    def log_current_state(self) -> None:
        """
        Log the current state now, regardless of the stride.
        """
        if self.state_history:
            rerun.set_time(
                self.timeline,
                sequence=self._world.state.version,
                recording=self.recording,
            )
            self._log_state()
        else:
            self._log_state(static=True)
        self.last_logged_version = self._world.state.version

    def stop(self) -> None:
        """
        Detach the callbacks from the world and flush pending data to the sink.
        """
        super().stop()
        self.model_cb.stop()
        self.recording.flush()

    @staticmethod
    def read_recording_entities(
        path: str, dataset_name: str = "semdt", timeline: Optional[str] = None
    ) -> set[str]:
        """
        Return the entity paths recorded in an ``.rrd`` file.

        Reads back through Rerun's in-process server / DataFusion
        reader. Only the schema (the logged entity paths) is recovered
        -- cell values (geometry, transforms) are not read back.
        Intended for verifying what was recorded.

        :param path: Path to the ``.rrd`` file to inspect.
        :param dataset_name: Handle the recording is registered under
            while reading.
        :param timeline: Timeline whose entities are included alongside the
            static ones; ``None`` recovers only statically logged entities.
        :return: The set of entity paths present in the recording.
        """
        with rerun.server.Server(datasets={dataset_name: [path]}) as server:
            reader = server.client().get_dataset(dataset_name).reader(timeline)
            columns = reader.schema().names
        return {name.split(":", 1)[0] for name in columns}
