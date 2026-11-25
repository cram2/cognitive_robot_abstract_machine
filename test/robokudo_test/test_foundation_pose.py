"""
Tests of the FoundationPose annotator against a scripted pose estimator: class meshes,
matching across frames, pose update modes and visualization.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from pathlib import Path

import numpy
import pytest
import trimesh
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from py_trees.common import Status
from robokudo.annotators.exceptions import (
    DuplicateObjectClassMesh,
    ImageResolutionMismatch,
)
from robokudo.annotators.foundation_pose import (
    DrawableGeometryKey,
    EstimateAllObjects,
    EstimateAndTrack,
    EstimateUnposedObjects,
    FoundationPoseAnnotator,
    ObjectClassMesh,
    PosedObject,
    PoseUpdateMode,
    PoseVisualization,
    PreviousFramePoses,
    TrackPosedObjects,
    VisualizationFrame,
)
from robokudo.annotators.object_observation import ColorAndDepthFrame, ObjectObservation
from robokudo.annotators.object_pose_estimator import (
    MeshBasedPoseEstimator,
    ObjectPoseEstimator,
)
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import ColorToDepthRatio, Point2D, Rect
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.annotator_helper import transform_pose_from_camera_to_world
from robokudo.utils.transform import get_transform_matrix_from_q
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color, Mesh, Scale
from semantic_digital_twin.world_description.world_entity import Body

# %% Scripted pose estimation


@dataclass(kw_only=True)
class ScriptedPoseEstimator(MeshBasedPoseEstimator):
    """
    Answers every estimate and track with a preset pose and records what it was asked.
    """

    estimated_pose: Pose | None = field(
        default_factory=lambda: object_pose_at_depth(0.5)
    )
    """
    Pose every estimate returns.
    """
    tracked_pose: Pose = field(default_factory=lambda: object_pose_at_depth(0.6))
    """
    Pose every track returns.
    """
    loaded_mesh_files: list[Path] = field(default_factory=list)
    """
    Mesh files of every :meth:`load_meshes` call, in order.
    """
    selected_mesh_files: list[Path] = field(default_factory=list)
    """
    Mesh file of every :meth:`select_mesh` call, in order.
    """
    estimated_object_observations: list[ObjectObservation] = field(default_factory=list)
    """
    Observation of every :meth:`estimate_pose` call, in order.
    """
    previous_poses: list[Pose] = field(default_factory=list)
    """
    Previous pose of every :meth:`track_pose` call, in order.
    """

    def load_meshes(self, *, mesh_files: list[Path]) -> None:
        self.loaded_mesh_files.extend(mesh_files)

    def select_mesh(self, *, mesh_file: Path) -> None:
        self.selected_mesh_files.append(mesh_file)

    def estimate_pose(self, *, object_observation: ObjectObservation) -> Pose | None:
        self.estimated_object_observations.append(object_observation)
        return self.estimated_pose

    def track_pose(
        self, *, object_observation: ObjectObservation, previous_pose: Pose
    ) -> Pose:
        self.previous_poses.append(previous_pose)
        return self.tracked_pose


def object_pose_at_depth(depth: float) -> Pose:
    """
    :return: Pose of an object on the optical axis, ``depth`` meters in front of the camera.
    """
    return Pose.from_xyz_quaternion(pos_z=depth)


# %% Pose estimators with and without meshes


@dataclass(kw_only=True)
class PoseEstimatorWithoutMeshes(ObjectPoseEstimator):
    """
    Poses objects without knowing any mesh.
    """

    def estimate_pose(self, *, object_observation: ObjectObservation) -> Pose | None:
        return object_pose_at_depth(0.5)

    def track_pose(
        self, *, object_observation: ObjectObservation, previous_pose: Pose
    ) -> Pose:
        return previous_pose


def test_pose_estimator_can_exist_without_meshes() -> None:
    estimator = PoseEstimatorWithoutMeshes()

    assert not isinstance(estimator, MeshBasedPoseEstimator)
    assert not hasattr(estimator, MeshBasedPoseEstimator.load_meshes.__name__)
    assert not hasattr(estimator, MeshBasedPoseEstimator.select_mesh.__name__)


def test_mesh_based_pose_estimator_has_to_load_and_select_meshes() -> None:
    assert MeshBasedPoseEstimator.__abstractmethods__ == {
        ObjectPoseEstimator.estimate_pose.__name__,
        ObjectPoseEstimator.track_pose.__name__,
        MeshBasedPoseEstimator.load_meshes.__name__,
        MeshBasedPoseEstimator.select_mesh.__name__,
    }


# %% Scene


@dataclass(kw_only=True)
class FlatSurfaceScene:
    """
    A camera looking at a flat surface one meter away, with classified objects on it.
    """

    class_name: str = "box"
    """
    Class of the objects the annotator has a mesh for.
    """

    image_resolution: CameraResolution = field(
        default_factory=lambda: CameraResolution(width=64, height=48)
    )
    """
    Resolution of the color and the depth image.
    """

    object_region: Rect = field(
        default_factory=lambda: Rect(pos=Point2D(x=28, y=20), width=8, height=8)
    )
    """
    Color image region every object covers, around the image center.
    """

    def fill_cas(self, *, annotator: FoundationPoseAnnotator) -> None:
        """
        Give the CAS of the annotator a new frame of the surface.
        """
        cas = annotator.get_cas()
        world = World()
        root = Body(name=PrefixedName(name="world"))
        with world.modify_world():
            world.add_body(body=root)
        camera_model = PinholeCameraModel.from_field_of_view(
            resolution=self.image_resolution,
            field_of_view=FieldOfView(
                horizontal_angle=numpy.radians(60.0),
                vertical_angle=numpy.radians(45.0),
            ),
        )
        cas.camera_observation = CameraObservation(
            camera=Camera(
                name=PrefixedName(name="camera"), root=root, camera_model=camera_model
            ),
            effective_camera_model=camera_model,
            world_T_camera=None,
            timestamp_nanoseconds=0,
        )
        shape = (self.image_resolution.height, self.image_resolution.width)
        cas.set(
            view_name=CASViews.COLOR_IMAGE,
            value=numpy.zeros((*shape, 3), dtype=numpy.uint8),
        )
        cas.set(
            view_name=CASViews.DEPTH_IMAGE,
            value=numpy.full(
                shape,
                annotator.descriptor.parameters.depth_units_per_meter,
                dtype=numpy.uint16,
            ),
        )
        cas.color_to_depth_ratio = ColorToDepthRatio()

    def add_object_hypothesis(
        self, *, annotator: FoundationPoseAnnotator, class_name: str | None = None
    ) -> ObjectHypothesis:
        """
        Add a classified hypothesis with a full mask over :attr:`object_region` to the
        CAS of the annotator.

        :param class_name: Class of the hypothesis, :attr:`class_name` if ``None``.
        """
        object_hypothesis = ObjectHypothesis()
        object_hypothesis.roi.roi = Rect(
            pos=Point2D(x=self.object_region.pos.x, y=self.object_region.pos.y),
            width=self.object_region.width,
            height=self.object_region.height,
        )
        object_hypothesis.roi.mask = numpy.full(
            (self.object_region.height, self.object_region.width),
            numpy.iinfo(numpy.uint8).max,
            dtype=numpy.uint8,
        )
        object_hypothesis.annotations.append(
            Classification(classname=class_name or self.class_name, confidence=1.0)
        )
        annotator.get_cas().annotations.append(object_hypothesis)
        return object_hypothesis

    def class_mesh(self, *, mesh_file: Path, scale: float = 1.0) -> ObjectClassMesh:
        """
        :param mesh_file: The mesh file.
        :param scale: Uniform factor bringing the file coordinates to meters.
        :return: The mesh of the objects of the scene.
        """
        return ObjectClassMesh(
            class_name=self.class_name,
            shape=Mesh(filename=str(mesh_file), scale=Scale(x=scale, y=scale, z=scale)),
        )

    def next_frame(self, *, annotator: FoundationPoseAnnotator) -> None:
        """
        Start a new CAS in the pipeline of the annotator, holding a new frame.
        """
        annotator.get_parent_pipeline().create_new_cas()
        self.fill_cas(annotator=annotator)


# %% Fixtures


@pytest.fixture
def scene() -> FlatSurfaceScene:
    """
    The scene with its default object class and region.
    """
    return FlatSurfaceScene()


@pytest.fixture
def box_mesh_file(tmp_path: Path) -> Path:
    """
    A box mesh in meters, centered on its origin.
    """
    mesh_file = tmp_path / "box.obj"
    trimesh.creation.box(extents=(0.04, 0.06, 0.1)).export(mesh_file)
    return mesh_file


@pytest.fixture
def pose_estimator() -> ScriptedPoseEstimator:
    """
    A scripted estimator with its default poses.
    """
    return ScriptedPoseEstimator()


@pytest.fixture
def annotator(
    scene: FlatSurfaceScene,
    box_mesh_file: Path,
    pose_estimator: ScriptedPoseEstimator,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> FoundationPoseAnnotator:
    """
    An annotator for the class of the scene, in a pipeline whose CAS holds a frame of
    it.
    """
    descriptor = FoundationPoseAnnotator.Descriptor()
    descriptor.parameters = descriptor.Parameters()
    descriptor.parameters.class_meshes = [scene.class_mesh(mesh_file=box_mesh_file)]
    annotator = FoundationPoseAnnotator(
        descriptor=descriptor, pose_estimator=pose_estimator
    )
    pipeline = Pipeline("FoundationPose")
    pipeline.add_child(annotator)
    pipeline.setup()
    scene.fill_cas(annotator=annotator)
    return annotator


def add_pose_annotation(
    object_hypothesis: ObjectHypothesis, pose: Pose
) -> PoseAnnotation:
    """
    Give an object hypothesis a pose annotation of an unrotated pose.

    :param pose: Pose whose translation the annotation gets.
    """
    pose_annotation = PoseAnnotation()
    pose_annotation.translation = pose.position.to_np()[:3].tolist()
    pose_annotation.rotation = [0.0, 0.0, 0.0, 1.0]
    object_hypothesis.annotations.append(pose_annotation)
    return pose_annotation


def pose_annotations(object_hypothesis: ObjectHypothesis) -> list[PoseAnnotation]:
    """
    :return: The pose annotations of an object hypothesis, in order.
    """
    return [
        annotation
        for annotation in object_hypothesis.annotations
        if isinstance(annotation, PoseAnnotation)
    ]


def annotation_matrix(pose_annotation: PoseAnnotation) -> numpy.ndarray:
    """
    :return: The pose of an annotation as a 4x4 homogeneous transform.
    """
    return get_transform_matrix_from_q(
        pose_annotation.rotation, pose_annotation.translation
    )


# %% Object class meshes


def test_unscaled_single_mesh_is_estimated_from_its_own_file(
    scene: FlatSurfaceScene,
    box_mesh_file: Path,
):
    """
    A class mesh that needs no conversion hands the pose estimator the file it was
    given.
    """
    class_mesh = scene.class_mesh(mesh_file=box_mesh_file)

    assert class_mesh.estimator_mesh_file == box_mesh_file


def test_scaled_mesh_is_rewritten_in_meters(
    scene: FlatSurfaceScene,
    box_mesh_file: Path,
):
    """
    A scaled class mesh hands the pose estimator a new file with the scale applied.
    """
    scale = 0.5

    class_mesh = scene.class_mesh(mesh_file=box_mesh_file, scale=scale)

    rewritten_extents = trimesh.load(class_mesh.estimator_mesh_file).extents
    source_extents = trimesh.load(box_mesh_file).extents
    assert class_mesh.estimator_mesh_file != box_mesh_file
    numpy.testing.assert_allclose(rewritten_extents, source_extents * scale)


def test_mesh_file_declaring_its_unit_is_rewritten_in_meters(
    scene: FlatSurfaceScene,
    tmp_path: Path,
):
    """
    A mesh file that declares millimeters as its unit is converted into meters.
    """
    box = trimesh.creation.box(extents=(100.0, 100.0, 100.0))
    box.units = "millimeters"
    mesh_file = tmp_path / "millimeters.glb"
    box.export(mesh_file)

    class_mesh = scene.class_mesh(mesh_file=mesh_file)

    expected_extents = box.extents * trimesh.units.unit_conversion(
        "millimeters", "meters"
    )
    numpy.testing.assert_allclose(class_mesh.shape.mesh.extents, expected_extents)
    numpy.testing.assert_allclose(
        trimesh.load_scene(class_mesh.estimator_mesh_file).extents, expected_extents
    )


def test_multi_part_mesh_is_rewritten_as_one_mesh(
    scene: FlatSurfaceScene,
    tmp_path: Path,
):
    """
    A mesh file with several parts is converted into one mesh holding every vertex.
    """
    parts = [
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)).apply_translation((0.3, 0, 0)),
    ]
    mesh_file = tmp_path / "parts.glb"
    trimesh.Scene(parts).export(mesh_file)

    class_mesh = scene.class_mesh(mesh_file=mesh_file)

    (rewritten,) = trimesh.load(class_mesh.estimator_mesh_file).geometry.values()
    assert len(rewritten.vertices) == sum(len(part.vertices) for part in parts)


def test_single_mesh_scene_is_estimated_from_its_own_file(
    scene: FlatSurfaceScene, tmp_path: Path
):
    """
    A scene file holding a single mesh is handed to the pose estimator unchanged.
    """
    mesh_file = tmp_path / "single.glb"
    trimesh.Scene([trimesh.creation.box(extents=(0.1, 0.1, 0.1))]).export(mesh_file)

    class_mesh = scene.class_mesh(mesh_file=mesh_file)

    assert class_mesh.estimator_mesh_file == mesh_file


def test_two_meshes_of_one_class_are_rejected(
    scene: FlatSurfaceScene,
    box_mesh_file: Path,
    pose_estimator: ScriptedPoseEstimator,
):
    """
    Two meshes configured for one object class raise :class:`DuplicateObjectClassMesh`.
    """
    descriptor = FoundationPoseAnnotator.Descriptor()
    descriptor.parameters = descriptor.Parameters()
    descriptor.parameters.class_meshes = [
        scene.class_mesh(mesh_file=box_mesh_file),
        scene.class_mesh(mesh_file=box_mesh_file),
    ]

    with pytest.raises(DuplicateObjectClassMesh):
        FoundationPoseAnnotator(descriptor=descriptor, pose_estimator=pose_estimator)


def test_estimator_loads_every_class_mesh(
    annotator: FoundationPoseAnnotator, pose_estimator: ScriptedPoseEstimator
):
    assert (
        pose_estimator.loaded_mesh_files == annotator.class_meshes.estimator_mesh_files
    )


# %% Frames


def test_frame_rejects_images_of_different_resolutions(
    annotator: FoundationPoseAnnotator,
):
    frame = annotator.color_and_depth_frame()

    with pytest.raises(ImageResolutionMismatch):
        ColorAndDepthFrame(
            color_image=frame.color_image[::2, ::2],
            depth_image=frame.depth_image,
            camera_model=frame.camera_model,
        )


# %% Matching across frames


def test_match_takes_the_most_overlapping_object_of_the_class(scene: FlatSurfaceScene):
    """
    Of the previous frame's objects of the class, the one whose mask overlaps most is
    matched.
    """
    overlapping = PosedObject(
        class_name=scene.class_name,
        pose=object_pose_at_depth(0.5),
        mask_region=scene.object_region,
    )
    distant = PosedObject(
        class_name=scene.class_name,
        pose=object_pose_at_depth(0.7),
        mask_region=Rect(pos=Point2D(x=0, y=0), width=8, height=8),
    )
    previous_frame_poses = PreviousFramePoses(unmatched_objects=[distant, overlapping])

    pose = previous_frame_poses.take_match(
        class_name=scene.class_name,
        mask_region=scene.object_region,
        minimum_overlap=0.1,
    )

    assert pose is overlapping.pose
    assert previous_frame_poses.unmatched_objects == [distant]


def test_match_ignores_objects_of_other_classes(scene: FlatSurfaceScene):
    """
    An object of another class is never matched, however much it overlaps.
    """
    other_class = PosedObject(
        class_name="other",
        pose=object_pose_at_depth(0.5),
        mask_region=scene.object_region,
    )
    previous_frame_poses = PreviousFramePoses(unmatched_objects=[other_class])

    assert (
        previous_frame_poses.take_match(
            class_name=scene.class_name,
            mask_region=scene.object_region,
            minimum_overlap=0.1,
        )
        is None
    )


def test_match_requires_the_minimum_overlap(scene: FlatSurfaceScene):
    """
    An object that overlaps less than the minimum overlap is not matched.
    """
    distant = PosedObject(
        class_name=scene.class_name,
        pose=object_pose_at_depth(0.5),
        mask_region=Rect(pos=Point2D(x=0, y=0), width=8, height=8),
    )
    previous_frame_poses = PreviousFramePoses(unmatched_objects=[distant])

    assert (
        previous_frame_poses.take_match(
            class_name=scene.class_name,
            mask_region=scene.object_region,
            minimum_overlap=0.1,
        )
        is None
    )
    assert previous_frame_poses.unmatched_objects == [distant]


# %% Pose update modes


@pytest.mark.parametrize(
    "mode, is_posed",
    [
        (EstimateUnposedObjects(), True),
        (TrackPosedObjects(), False),
    ],
)
def test_mode_leaves_object_without_pose(
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
    mode: PoseUpdateMode,
    is_posed: bool,
    scene: FlatSurfaceScene,
):
    """
    Estimating only unposed objects leaves a posed object alone, and tracking only posed
    objects leaves an unposed one alone.
    """
    annotator.descriptor.parameters.pose_update_mode = mode
    object_hypothesis = scene.add_object_hypothesis(annotator=annotator)
    if is_posed:
        add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))
    annotations_before = list(object_hypothesis.annotations)

    annotator.compute()

    assert object_hypothesis.annotations == annotations_before
    assert pose_estimator.estimated_object_observations == []
    assert pose_estimator.previous_poses == []


@pytest.mark.parametrize(
    "mode",
    [
        EstimateAndTrack(),
        EstimateUnposedObjects(),
        EstimateAllObjects(),
    ],
)
def test_mode_estimates_object_without_pose(
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
    mode: PoseUpdateMode,
    scene: FlatSurfaceScene,
):
    """
    Every mode that estimates objects without a pose annotates the estimated pose.
    """
    annotator.descriptor.parameters.pose_update_mode = mode
    object_hypothesis = scene.add_object_hypothesis(annotator=annotator)

    annotator.compute()

    (pose_annotation,) = pose_annotations(object_hypothesis)
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotation), pose_estimator.estimated_pose.to_np()
    )
    assert pose_estimator.previous_poses == []


@pytest.mark.parametrize("mode", [EstimateAndTrack(), TrackPosedObjects()])
def test_mode_tracks_object_from_its_pose_annotation(
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
    mode: PoseUpdateMode,
    scene: FlatSurfaceScene,
):
    """
    Every mode that tracks starts tracking from the pose the object already carries.
    """
    annotator.descriptor.parameters.pose_update_mode = mode
    object_hypothesis = scene.add_object_hypothesis(annotator=annotator)
    known_pose = object_pose_at_depth(0.4)
    add_pose_annotation(object_hypothesis, known_pose)

    annotator.compute()

    (previous_pose,) = pose_estimator.previous_poses
    numpy.testing.assert_allclose(previous_pose.to_np(), known_pose.to_np())
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotations(object_hypothesis)[-1]),
        pose_estimator.tracked_pose.to_np(),
    )


def test_estimate_all_objects_ignores_existing_pose(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    """
    Estimating all objects estimates anew instead of tracking from an existing pose.
    """
    annotator.descriptor.parameters.pose_update_mode = EstimateAllObjects()
    object_hypothesis = scene.add_object_hypothesis(annotator=annotator)
    add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))

    annotator.compute()

    assert pose_estimator.previous_poses == []
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotations(object_hypothesis)[-1]),
        pose_estimator.estimated_pose.to_np(),
    )


def test_compute_records_its_duration(annotator: FoundationPoseAnnotator):
    annotator.compute()

    assert annotator._times["compute"] > 0


def test_object_of_unknown_class_is_not_posed(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    """
    An object whose class has no configured mesh is neither estimated nor annotated.
    """
    object_hypothesis = scene.add_object_hypothesis(
        annotator=annotator, class_name="unknown"
    )

    annotator.compute()

    assert pose_annotations(object_hypothesis) == []
    assert pose_estimator.estimated_object_observations == []


# %% Pose annotations


def test_pose_annotation_reaches_the_world_through_the_camera_pose_of_the_cas(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    """
    The pose annotation is in the camera frame, so that the CAS's camera pose turns it
    into
    the estimated pose in the world.
    """
    world_T_camera = get_transform_matrix_from_q(
        [0.0, 0.0, numpy.sin(numpy.pi / 4), numpy.cos(numpy.pi / 4)], [1.0, 2.0, 3.0]
    )
    camera_body = Body(name=PrefixedName(name="optical frame"))
    cas = annotator.get_cas()
    camera_model = cas.camera_observation.effective_camera_model
    cas.camera_observation = CameraObservation(
        camera=Camera(
            name=PrefixedName(name="camera"),
            root=camera_body,
            camera_model=camera_model,
        ),
        effective_camera_model=camera_model,
        world_T_camera=HomogeneousTransformationMatrix(
            data=world_T_camera,
            reference_frame=Body(name=PrefixedName(name="world")),
            child_frame=camera_body,
        ),
        timestamp_nanoseconds=0,
    )
    object_hypothesis = scene.add_object_hypothesis(annotator=annotator)

    annotator.compute()

    (pose_annotation,) = pose_annotations(object_hypothesis)
    numpy.testing.assert_allclose(
        annotation_matrix(transform_pose_from_camera_to_world(cas, pose_annotation)),
        world_T_camera @ pose_estimator.estimated_pose.to_np(),
        atol=1e-9,
    )
    assert pose_annotation.source == annotator.get_class_name()


def test_tracked_pose_overwrites_its_annotation_when_configured(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    """
    Configured to update existing annotations, tracking overwrites the tracked one.
    """
    annotator.descriptor.parameters.pose_update_mode = EstimateAndTrack(
        update_existing_pose_annotations=True
    )
    object_hypothesis = scene.add_object_hypothesis(annotator=annotator)
    pose_annotation = add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))

    annotator.compute()

    assert pose_annotations(object_hypothesis) == [pose_annotation]
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotation), pose_estimator.tracked_pose.to_np()
    )


def test_object_without_estimated_pose_gets_no_annotation(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    """
    An object the estimator finds no pose for gets no pose annotation.
    """
    pose_estimator.estimated_pose = None
    object_hypothesis = scene.add_object_hypothesis(annotator=annotator)

    annotator.compute()

    assert pose_annotations(object_hypothesis) == []


# %% Tracking


def test_object_is_tracked_from_its_latest_pose_annotation(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    """
    An object with several pose annotations is tracked from the last one.
    """
    object_hypothesis = scene.add_object_hypothesis(annotator=annotator)
    add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))
    latest_pose = object_pose_at_depth(0.45)
    add_pose_annotation(object_hypothesis, latest_pose)

    annotator.compute()

    (previous_pose,) = pose_estimator.previous_poses
    numpy.testing.assert_allclose(previous_pose.to_np(), latest_pose.to_np())


def test_object_is_tracked_from_the_previous_frame(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    """
    An object matched to one of the previous frame is tracked from that object's pose.
    """
    scene.add_object_hypothesis(annotator=annotator)
    annotator.compute()
    scene.next_frame(annotator=annotator)
    object_hypothesis = scene.add_object_hypothesis(annotator=annotator)

    annotator.compute()

    (previous_pose,) = pose_estimator.previous_poses
    numpy.testing.assert_allclose(
        previous_pose.to_np(), pose_estimator.estimated_pose.to_np()
    )
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotations(object_hypothesis)[0]),
        pose_estimator.tracked_pose.to_np(),
    )


def test_object_is_not_tracked_across_frames_when_disabled(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    """
    With tracking across frames disabled, the next frame's object is estimated anew.
    """
    annotator.descriptor.parameters.pose_update_mode = EstimateAndTrack(
        track_across_frames=False
    )
    scene.add_object_hypothesis(annotator=annotator)
    annotator.compute()
    scene.next_frame(annotator=annotator)
    scene.add_object_hypothesis(annotator=annotator)

    annotator.compute()

    assert pose_estimator.previous_poses == []
    assert len(pose_estimator.estimated_object_observations) == 2


def test_track_leaving_the_mask_is_estimated_anew(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    """
    A tracked pose that leaves the object's mask is discarded and estimated anew.
    """
    pose_estimator.tracked_pose = Pose.from_xyz_quaternion(pos_x=0.5, pos_z=0.5)
    object_hypothesis = scene.add_object_hypothesis(annotator=annotator)
    add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))

    annotator.compute()

    assert len(pose_estimator.previous_poses) == 1
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotations(object_hypothesis)[-1]),
        pose_estimator.estimated_pose.to_np(),
    )


# %% Depth image mask


def test_object_mask_is_placed_at_depth_image_resolution(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    """
    With a depth image at half the color resolution, the mask handed to the estimator
    matches the depth image and covers the object region scaled to it.
    """
    cas = annotator.get_cas()
    color_to_depth_ratio = ColorToDepthRatio(horizontal=0.5, vertical=0.5)
    cas.color_to_depth_ratio = color_to_depth_ratio
    depth_image = cas.get(view_name=CASViews.DEPTH_IMAGE)
    cas.set(view_name=CASViews.DEPTH_IMAGE, value=depth_image[::2, ::2].copy())
    scene.add_object_hypothesis(annotator=annotator)

    annotator.compute()

    (object_observation,) = pose_estimator.estimated_object_observations
    expected_region = color_to_depth_ratio.depth_region(region=scene.object_region)
    assert (
        object_observation.mask.shape == cas.get(view_name=CASViews.DEPTH_IMAGE).shape
    )
    assert object_observation.mask_region == expected_region
    assert (
        object_observation.mask.sum() == expected_region.width * expected_region.height
    )


# %% Visualization


def visualization_of(
    annotator: FoundationPoseAnnotator, frame: VisualizationFrame
) -> PoseVisualization:
    """
    :return: An empty visualization of the current frame of the annotator.
    """
    cas = annotator.get_cas()
    return PoseVisualization(
        color_image=cas.get_copy(view_name=CASViews.COLOR_IMAGE),
        color_to_depth_ratio=cas.color_to_depth_ratio,
        camera_model=annotator.color_and_depth_frame().camera_model,
        frame=frame,
    )


def test_mesh_frame_bounding_box_spans_the_mesh(
    scene: FlatSurfaceScene, annotator: FoundationPoseAnnotator
):
    """
    In the mesh frame, the drawn bounding box has the extent of the mesh.
    """
    shape = annotator.class_meshes.meshes_by_class_name[scene.class_name].shape
    visualization = visualization_of(annotator, VisualizationFrame.MESH)

    bounding_box = visualization.bounding_box(
        shape=shape, pose=object_pose_at_depth(0.5)
    )

    numpy.testing.assert_allclose(bounding_box.extent, shape.mesh.extents)


def test_smallest_bounding_box_frame_uses_the_oriented_box(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
):
    """
    In the smallest bounding box frame, the drawn box is the mesh's oriented box.
    """
    shape = annotator.class_meshes.meshes_by_class_name[scene.class_name].shape
    visualization = visualization_of(
        annotator, VisualizationFrame.SMALLEST_BOUNDING_BOX
    )

    bounding_box = visualization.bounding_box(
        shape=shape, pose=object_pose_at_depth(0.5)
    )

    numpy.testing.assert_allclose(
        bounding_box.extent, shape.mesh.bounding_box_oriented.primitive.extents
    )


def test_nothing_is_drawn_for_an_object_behind_the_camera(
    scene: FlatSurfaceScene, annotator: FoundationPoseAnnotator
):
    """
    An object behind the camera leaves the image untouched.
    """
    shape = annotator.class_meshes.meshes_by_class_name[scene.class_name].shape
    visualization = visualization_of(annotator, VisualizationFrame.MESH)

    visualization.draw(
        shape=shape, pose=object_pose_at_depth(-0.5), label="behind the camera"
    )

    assert not visualization.color_image.any()


def test_axes_are_drawn_in_color_image_pixels_from_the_depth_image_calibration(
    annotator: FoundationPoseAnnotator,
):
    """
    Axes are projected with the color image calibration derived from the depth image's,
    so
    they end at the expected color image pixels.
    """
    color_camera_model = annotator.color_and_depth_frame().camera_model
    color_to_depth_ratio = ColorToDepthRatio(horizontal=0.5, vertical=0.5)
    visualization = PoseVisualization(
        color_image=annotator.get_cas().get_copy(view_name=CASViews.COLOR_IMAGE),
        color_to_depth_ratio=color_to_depth_ratio,
        camera_model=color_camera_model.scaled(
            scale_x=color_to_depth_ratio.horizontal,
            scale_y=color_to_depth_ratio.vertical,
        ),
    )
    depth = 0.5

    visualization.draw_axes(axes_pose=object_pose_at_depth(depth).homogeneous_matrix)

    intrinsic_matrix = color_camera_model.intrinsic_matrix
    center_x, center_y = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    half_axis_x = intrinsic_matrix[0, 0] * visualization.axis_length / depth / 2
    half_axis_y = intrinsic_matrix[1, 1] * visualization.axis_length / depth / 2
    maximum = Color.opencv_maximum_channel_value
    assert visualization.color_image[
        int(center_y), int(center_x + half_axis_x)
    ].tolist() == [0, 0, maximum]
    assert visualization.color_image[
        int(center_y + half_axis_y), int(center_x)
    ].tolist() == [0, maximum, 0]


def test_axes_reaching_behind_the_camera_are_not_drawn(
    annotator: FoundationPoseAnnotator,
):
    """
    Axes that point from the object behind the camera leave the image untouched.
    """
    visualization = visualization_of(annotator, VisualizationFrame.MESH)
    pose_facing_the_camera = Pose.from_xyz_quaternion(
        0.0, 0.0, 0.05, 1.0, 0.0, 0.0, 0.0
    )

    visualization.draw_axes(axes_pose=pose_facing_the_camera.homogeneous_matrix)

    assert not visualization.color_image.any()


def test_posed_objects_are_drawn_and_named_in_the_3d_view(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
):
    """
    Each posed object's geometries reach the 3D view under distinct names.
    """
    annotator.descriptor.parameters.visualize_object_mesh = True
    scene.add_object_hypothesis(annotator=annotator)

    status = annotator.compute()

    output = annotator.get_annotator_output_struct()
    names = [geometry[DrawableGeometryKey.NAME] for geometry in output.geometries]
    assert status == Status.SUCCESS
    assert output.image.any()
    assert len(names) == len(set(names)) == 3
