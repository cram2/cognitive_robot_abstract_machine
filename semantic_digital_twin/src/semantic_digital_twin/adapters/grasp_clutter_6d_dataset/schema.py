from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

import numpy as np
import trimesh
from typing_extensions import Self

from semantic_digital_twin.adapters.grasp_clutter_6d_dataset.exceptions import (
    GraspClutter6DImageNotFoundError,
    GraspClutter6DObjectModelNotFoundError,
    GraspClutter6DSceneFilesMissingError,
)
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    RotationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Mesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class GraspClutter6DCameraInfo:
    """
    Camera parameters of one frame, parsed from one entry of a scene's
    `scene_camera.json`, following the BOP dataset format
    (https://github.com/thodan/bop_toolkit).
    """

    MILLIMETERS_TO_METERS: ClassVar[float] = 1e-3
    """
    GraspClutter6D's BOP-format JSON files give every translation in millimeters; the
    rest of this package works in meters (SI), so every translation parsed from the
    dataset is converted with this factor.
    """

    field_of_view: FieldOfView
    """
    This camera's field of view, estimated from the frame's `cam_K` intrinsics matrix -
    every other camera in this package (e.g.
    :class:`~semantic_digital_twin.robots.pr2.PR2KinectV1`) only ever stores a
    `FieldOfView`, never a raw intrinsics matrix, so `cam_K` is used only transiently, to
    derive this, rather than kept as a field of its own. `scene_camera.json` does not
    give the image resolution directly, so this assumes the principal point sits at the
    image center (``width = 2 * cx``, ``height = 2 * cy``) - a common approximation, not
    a value read from the dataset itself.
    """

    depth_scale: float
    """Multiply this frame's raw depth image values by this factor to get millimeters."""

    camera_T_world: Optional[HomogeneousTransformationMatrix] = None
    """
    This frame's world-to-camera transform (`cam_R_w2c`/`cam_t_w2c` bundled together):
    rotates/translates a point given in the dataset's world frame into this camera's
    frame. Not yet bound to a reference/child frame - that only exists once a world is
    being built (see :meth:`GraspClutter6DScene._add_world_frame`). `None` if the scene
    defines no world frame, in which case object poses are only meaningful relative to
    the camera.
    """

    @classmethod
    def from_json(cls, data: Dict) -> Self:
        rotation = data.get("cam_R_w2c")
        translation = data.get("cam_t_w2c")
        camera_T_world = None
        if rotation is not None and translation is not None:
            camera_T_world = HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=Point3(
                    *(
                        np.array(translation, dtype=float)
                        * cls.MILLIMETERS_TO_METERS
                    )
                ),
                rotation_matrix=RotationMatrix(
                    data=np.array(rotation, dtype=float).reshape(3, 3)
                ),
            )
        intrinsics = np.array(data["cam_K"], dtype=float).reshape(3, 3)
        focal_x, focal_y = intrinsics[0, 0], intrinsics[1, 1]
        principal_x, principal_y = intrinsics[0, 2], intrinsics[1, 2]
        return cls(
            field_of_view=FieldOfView(
                horizontal_angle=2 * np.arctan(principal_x / focal_x),
                vertical_angle=2 * np.arctan(principal_y / focal_y),
            ),
            depth_scale=float(data["depth_scale"]),
            camera_T_world=camera_T_world,
        )


@dataclass
class GraspClutter6DObjectPose:
    """
    One object's ground-truth pose in one frame, parsed from one entry of a scene's
    `scene_gt.json`, following the BOP dataset format.
    """

    object_id: int
    """
    1-indexed object id, matching the `obj_%06d` numbering of the dataset's model files
    and `models_info.json` keys.
    """

    camera_T_object: HomogeneousTransformationMatrix
    """
    This object's pose (`cam_R_m2c`/`cam_t_m2c` bundled together) relative to the
    camera. Not yet bound to a reference/child frame - `GraspClutter6DScene` rebinds it
    to the actual camera/object bodies once they exist, e.g.
    ``HomogeneousTransformationMatrix(data=pose.camera_T_object, reference_frame=camera_body, child_frame=object_body)``.
    """

    @classmethod
    def from_json(cls, data: Dict) -> Self:
        return cls(
            object_id=int(data["obj_id"]),
            camera_T_object=HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=Point3(
                    *(
                        np.array(data["cam_t_m2c"], dtype=float)
                        * GraspClutter6DCameraInfo.MILLIMETERS_TO_METERS
                    )
                ),
                rotation_matrix=RotationMatrix(
                    data=np.array(data["cam_R_m2c"], dtype=float).reshape(3, 3)
                ),
            ),
        )


@dataclass
class GraspClutter6DFrame:
    """
    One annotated camera frame of a scene: its camera parameters and the ground-truth pose
    of every object visible in it.
    """

    image_id: str
    """
    The frame's id, exactly as given by `scene_camera.json`/`scene_gt.json` (e.g.
    ``"1"``). GraspClutter6D numbers its four physical cameras (RealSense D415/D435,
    Azure Kinect, Zivid) and their viewpoints into one flat sequence of frames per scene,
    so this id does not by itself say which physical camera captured it.
    """

    camera: GraspClutter6DCameraInfo
    """This frame's camera parameters."""

    object_poses: List[GraspClutter6DObjectPose] = field(default_factory=list)
    """The ground-truth pose of every object visible in this frame."""


@dataclass
class GraspClutter6DScene:
    """
    One scene of the GraspClutter6D dataset (https://sites.google.com/view/graspclutter6d),
    parsed from its BOP-format `scene_camera.json`/`scene_gt.json`.

    .. important::
        GraspClutter6D's `models_info.json` gives each object's geometric bounds only
        (diameter and bounding box), not a semantic name - `create_world`'s
        `object_names` parameter must be supplied separately if named semantic
        annotations are wanted.
    """

    scene_id: str
    """The scene's id, e.g. ``"000005"``."""

    directory: Path
    """The scene's own extracted folder, directly containing `scene_camera.json`/`scene_gt.json`."""

    frames: List[GraspClutter6DFrame] = field(default_factory=list)
    """Every frame of this scene. Each frame carries its own `image_id` - see :meth:`frame`."""

    def frame(self, image_id: str) -> GraspClutter6DFrame:
        """
        :param image_id: The frame id to look up, e.g. ``"1"``.
        :raises GraspClutter6DImageNotFoundError: if no frame in `self.frames` has this id.
        :return: The matching frame.
        """
        for frame in self.frames:
            if frame.image_id == image_id:
                return frame
        raise GraspClutter6DImageNotFoundError(
            scene_id=self.scene_id, image_id=image_id
        )

    @classmethod
    def from_directory(cls, scene_id: str, directory: Path) -> Self:
        """
        Parse a scene from its extracted directory.

        :param scene_id: The scene's id, e.g. ``"000005"``.
        :param directory: The scene's own extracted folder.
        :raises GraspClutter6DSceneFilesMissingError: if `scene_camera.json`/
            `scene_gt.json` are not both found in `directory`.
        :return: The parsed scene.
        """
        camera_file = directory / "scene_camera.json"
        gt_file = directory / "scene_gt.json"
        if not camera_file.is_file() or not gt_file.is_file():
            raise GraspClutter6DSceneFilesMissingError(
                scene_id=scene_id, directory=directory
            )

        camera_json = json.loads(camera_file.read_text())
        gt_json = json.loads(gt_file.read_text())

        frames = [
            GraspClutter6DFrame(
                image_id=image_id,
                camera=GraspClutter6DCameraInfo.from_json(camera_data),
                object_poses=[
                    GraspClutter6DObjectPose.from_json(pose)
                    for pose in gt_json.get(image_id, [])
                ],
            )
            for image_id, camera_data in camera_json.items()
        ]
        return cls(scene_id=scene_id, directory=directory, frames=frames)

    def create_world(
        self,
        image_id: str,
        models_directory: Path,
        *,
        with_world_frame: bool = False,
        mesh_unit_scale: float = GraspClutter6DCameraInfo.MILLIMETERS_TO_METERS,
        object_names: Optional[Dict[int, str]] = None,
    ) -> World:
        """
        Build a :class:`~semantic_digital_twin.world.World` for one frame of this scene.

        Every object's pose is computed from its `cam_R_m2c`/`cam_t_m2c` ground truth
        relative to the camera - this is always present and unambiguous, unlike the
        scene's optional world frame - but its :class:`~semantic_digital_twin.world_description.connections.Connection6DoF`
        is attached directly to `world.root` (the camera body itself, or the `map` body
        when `with_world_frame` adds one), not to the camera body when that is a
        different, non-root body: MuJoCo requires a free joint (what Connection6DoF
        becomes when this world is mirrored into a simulator) to sit directly on the
        world root.

        :param image_id: The frame to build, one of `self.frames`' `image_id`s.
        :param models_directory: The extracted `models`/`models_eval`/`models_m` directory
            containing this frame's objects' `obj_%06d.ply` mesh files.
        :param with_world_frame: If True and the frame's camera has a `camera_T_world`,
            also add a `map` root body and place the camera under it via the inverse of
            that world-to-camera transform. If False (the default), or if the frame has
            no world-frame information, the camera body itself is the world's root.
        :param mesh_unit_scale: Factor applied to every loaded mesh's vertices. The
            default assumes millimeter-unit meshes (the `models`/`models_eval` archives);
            pass ``1.0`` when using the meter-unit `models_m`/`models_obj_m` archives.
        :param object_names: Optional ``object_id -> name`` lookup for the semantic
            annotation added to each object (see the class docstring for why this is not
            in the dataset itself). Objects without an entry get a generic
            ``f"object_{object_id:06d}"`` name.
        :raises GraspClutter6DImageNotFoundError: if `image_id` is not in `self.frames`.
        :raises GraspClutter6DObjectModelNotFoundError: if an object's mesh file is not
            found in `models_directory`.
        :return: The built world.
        """
        frame = self.frame(image_id)

        world = World()
        camera_body = Body()
        camera_body.name = PrefixedName(
            name="camera", prefix=f"{self.scene_id}_{image_id}"
        )
        with world.modify_world():
            world.add_body(camera_body)

        root_body = camera_body
        if with_world_frame and frame.camera.camera_T_world is not None:
            root_body = self._add_world_frame(world, frame.camera, camera_body)

        for index, pose in enumerate(frame.object_poses):
            self._create_object_in_world(
                world=world,
                pose=pose,
                index=index,
                camera_body=camera_body,
                root_body=root_body,
                image_id=image_id,
                models_directory=models_directory,
                mesh_unit_scale=mesh_unit_scale,
                object_names=object_names,
            )

        return world

    def _add_world_frame(
        self, world: World, camera_info: GraspClutter6DCameraInfo, camera_body: Body
    ) -> Body:
        """
        Add a `map` root body and place `camera_body` under it via the inverse of the
        frame's world-to-camera transform.

        :param world: The world `camera_body` was already added to.
        :param camera_info: The frame's camera parameters, with `camera_T_world` present.
        :param camera_body: The already-added camera body.
        :return: The new `map` body.
        """
        map_body = Body()
        map_body.name = PrefixedName(name="map", prefix=camera_body.name.prefix)

        # `camera_T_world` rotates/translates a point from the dataset's world frame
        # into the camera frame; the camera's pose in the map frame is its inverse.
        camera_T_map = HomogeneousTransformationMatrix(
            data=camera_info.camera_T_world,
            reference_frame=camera_body,
            child_frame=map_body,
        )
        map_T_camera = camera_T_map.inverse()

        with world.modify_world():
            connection = FixedConnection.create_with_dofs(
                world=world,
                parent=map_body,
                child=camera_body,
                parent_T_connection_expression=map_T_camera,
            )
            world.add_body(map_body)
            world.add_connection(connection)

        return map_body

    def _create_object_in_world(
        self,
        world: World,
        pose: GraspClutter6DObjectPose,
        index: int,
        camera_body: Body,
        root_body: Body,
        image_id: str,
        models_directory: Path,
        mesh_unit_scale: float,
        object_names: Optional[Dict[int, str]],
    ) -> Body:
        """
        Create one object's body, mesh, connection to `root_body`, and semantic
        annotation in `world`.

        :param camera_body: The frame's camera body - always used to compute the
            object's pose from its ground-truth `cam_R_m2c`/`cam_t_m2c`, since that is
            always available and unambiguous.
        :param root_body: The body the object's :class:`Connection6DoF` is actually
            attached to. Must be `world.root` (`camera_body` itself when there is no
            `map` frame, or the `map` body otherwise) - MuJoCo requires a free joint (what
            Connection6DoF becomes when this world is mirrored for
            :class:`~semantic_digital_twin.adapters.mujoco_video_recording.MujocoVideoRecorder`/
            :class:`~semantic_digital_twin.adapters.multi_sim.MujocoSim`) to sit directly
            on the world root; attaching it one level deeper (e.g. to `camera_body` while
            a separate `map` body is the actual root) raises "free joint can only be used
            on top level".
        :param index: This object's position in the frame's `object_poses`, used to keep
            two instances of the same `object_id` in one frame distinct.
        """
        mesh_path = models_directory / f"obj_{pose.object_id:06d}.ply"
        if not mesh_path.is_file():
            raise GraspClutter6DObjectModelNotFoundError(
                object_id=pose.object_id, models_directory=models_directory
            )

        name = (object_names or {}).get(pose.object_id, f"object_{pose.object_id:06d}")

        body = Body()
        body.name = PrefixedName(
            name=f"{name}_{index}", prefix=f"{self.scene_id}_{image_id}"
        )

        camera_T_object = HomogeneousTransformationMatrix(
            data=pose.camera_T_object, reference_frame=camera_body, child_frame=body
        )
        root_T_object = world.transform(camera_T_object, root_body)

        loaded_mesh = trimesh.load(str(mesh_path), process=False, force="mesh")
        loaded_mesh.apply_scale(mesh_unit_scale)
        mesh = Mesh.from_trimesh(
            mesh=loaded_mesh,
            origin=HomogeneousTransformationMatrix(reference_frame=body),
        )
        body.visual = ShapeCollection([mesh], reference_frame=body)
        body.collision = ShapeCollection([mesh], reference_frame=body)

        with world.modify_world():
            connection = Connection6DoF.create_with_dofs(
                world=world,
                parent=root_body,
                child=body,
                parent_T_connection_expression=root_T_object,
            )
            world.add_body(body)
            world.add_connection(connection)

        with world.modify_world():
            world.add_semantic_annotation(
                NaturalLanguageWithTypeDescription(
                    root=body, description=name, type_description="object"
                )
            )

        return body
