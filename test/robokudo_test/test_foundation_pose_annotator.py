from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field
from pathlib import Path

import numpy
import pytest
import trimesh
from py_trees.common import Status
from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.datastructures.field_of_view import FieldOfView
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.robot_parts import Camera
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from robokudo.annotators.exceptions import DuplicateCadModelClass
from robokudo.annotators.foundation_pose_annotator import (
    CadModel,
    CadModelSource,
    DrawableGeometryKey,
    FoundationPoseAnnotator,
    ObjectObservation,
    ObjectPoseEstimator,
    PoseUpdateMode,
    PoseVisualization,
    PosedObject,
    PreviousFramePoses,
    VisualizationFrame,
)
from robokudo.annotators.outputs import AnnotatorOutputPerPipelineMap
from robokudo.cas import CASViews
from robokudo.pipeline import Pipeline
from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.camera import CameraObservation
from robokudo.types.cv import ColorToDepthRatio, Point2D, Rect
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.transform import get_transform_matrix_from_q


# %% Scripted pose estimation


@dataclass
class ScriptedPoseEstimator(ObjectPoseEstimator):
    """
    Answers every estimate and track with a preset pose and records what it was asked.
    """

    estimated_pose: numpy.ndarray | None = field(
        default_factory=lambda: object_pose_at_depth(0.5)
    )
    """
    Pose every estimate returns.
    """
    tracked_pose: numpy.ndarray = field(
        default_factory=lambda: object_pose_at_depth(0.6)
    )
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
    estimated_observations: list[ObjectObservation] = field(default_factory=list)
    """
    Observation of every :meth:`estimate_pose` call, in order.
    """
    previous_poses: list[numpy.ndarray] = field(default_factory=list)
    """
    Previous pose of every :meth:`track_pose` call, in order.
    """

    def load_meshes(self, mesh_files: list[Path]) -> None:
        self.loaded_mesh_files.extend(mesh_files)

    def select_mesh(self, mesh_file: Path) -> None:
        self.selected_mesh_files.append(mesh_file)

    def estimate_pose(self, observation: ObjectObservation) -> numpy.ndarray | None:
        self.estimated_observations.append(observation)
        return self.estimated_pose

    def track_pose(
        self, observation: ObjectObservation, previous_pose: numpy.ndarray
    ) -> numpy.ndarray:
        self.previous_poses.append(previous_pose)
        return self.tracked_pose


def object_pose_at_depth(depth: float) -> numpy.ndarray:
    """
    :return: Pose of an object on the optical axis, ``depth`` meters in front of the camera.
    """
    pose = numpy.eye(4)
    pose[2, 3] = depth
    return pose


# %% Scene


@dataclass
class FlatSurfaceScene:
    """
    A camera looking at a flat surface one meter away, with classified objects on it.
    """

    class_name: str = "box"
    """
    Class of the objects the annotator has a CAD model for.
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

    def fill_cas(self, annotator: FoundationPoseAnnotator) -> None:
        """
        Give the CAS of the annotator a new frame of the surface.
        """
        cas = annotator.get_cas()
        world = World()
        root = Body(name=PrefixedName(name="world"))
        with world.modify_world():
            world.add_body(root)
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
        cas.set(CASViews.COLOR_IMAGE, numpy.zeros((*shape, 3), dtype=numpy.uint8))
        cas.set(
            CASViews.DEPTH_IMAGE,
            numpy.full(
                shape,
                annotator.descriptor.parameters.depth_units_per_meter,
                dtype=numpy.uint16,
            ),
        )
        cas.color_to_depth_ratio = ColorToDepthRatio()

    def add_object_hypothesis(
        self, annotator: FoundationPoseAnnotator, class_name: str | None = None
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

    def next_frame(self, annotator: FoundationPoseAnnotator) -> None:
        """
        Start a new CAS in the pipeline of the annotator, holding a new frame.
        """
        annotator.get_parent_pipeline().create_new_cas()
        self.fill_cas(annotator)


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
    tmp_path: Path,
    annotator_output_buffer: AnnotatorOutputPerPipelineMap,
) -> FoundationPoseAnnotator:
    """
    An annotator for the class of the scene, in a pipeline whose CAS holds a frame of
    it.
    """
    descriptor = FoundationPoseAnnotator.Descriptor()
    descriptor.parameters = descriptor.Parameters()
    descriptor.parameters.cad_models = [
        CadModelSource(class_name=scene.class_name, mesh_file=box_mesh_file)
    ]
    descriptor.parameters.rewritten_mesh_directory = tmp_path / "rewritten"
    annotator = FoundationPoseAnnotator(
        descriptor=descriptor, pose_estimator=pose_estimator
    )
    pipeline = Pipeline("FoundationPose")
    pipeline.add_child(annotator)
    pipeline.setup()
    scene.fill_cas(annotator)
    return annotator


def add_pose_annotation(
    object_hypothesis: ObjectHypothesis, pose: numpy.ndarray
) -> PoseAnnotation:
    """
    Give an object hypothesis a pose annotation of an unrotated pose.

    :param pose: Pose whose translation the annotation gets.
    """
    pose_annotation = PoseAnnotation()
    pose_annotation.translation = pose[:3, 3].tolist()
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


# %% CAD model loading


def test_unscaled_single_mesh_is_estimated_from_its_own_file(
    scene: FlatSurfaceScene,
    box_mesh_file: Path,
    tmp_path: Path,
):
    cad_model = CadModel.from_source(
        CadModelSource(class_name=scene.class_name, mesh_file=box_mesh_file), tmp_path
    )

    assert cad_model.estimator_mesh_file == box_mesh_file


def test_scaled_mesh_is_rewritten_in_meters(
    scene: FlatSurfaceScene,
    box_mesh_file: Path,
    tmp_path: Path,
):
    source = CadModelSource(
        class_name=scene.class_name, mesh_file=box_mesh_file, scale_to_meters=0.5
    )

    cad_model = CadModel.from_source(source, tmp_path / "rewritten")

    rewritten_extents = trimesh.load(cad_model.estimator_mesh_file).extents
    source_extents = trimesh.load(box_mesh_file).extents
    assert cad_model.estimator_mesh_file != box_mesh_file
    numpy.testing.assert_allclose(
        rewritten_extents, source_extents * source.scale_to_meters
    )


def test_multi_part_mesh_is_rewritten_as_one_mesh(
    scene: FlatSurfaceScene,
    tmp_path: Path,
):
    parts = [
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)),
        trimesh.creation.box(extents=(0.1, 0.1, 0.1)).apply_translation((0.3, 0, 0)),
    ]
    mesh_file = tmp_path / "parts.glb"
    trimesh.Scene(parts).export(mesh_file)

    cad_model = CadModel.from_source(
        CadModelSource(class_name=scene.class_name, mesh_file=mesh_file),
        tmp_path / "rewritten",
    )

    (rewritten,) = trimesh.load(cad_model.estimator_mesh_file).geometry.values()
    assert len(rewritten.vertices) == sum(len(part.vertices) for part in parts)


def test_single_mesh_scene_is_estimated_from_its_own_file(
    scene: FlatSurfaceScene, tmp_path: Path
):
    mesh_file = tmp_path / "single.glb"
    trimesh.Scene([trimesh.creation.box(extents=(0.1, 0.1, 0.1))]).export(mesh_file)

    cad_model = CadModel.from_source(
        CadModelSource(class_name=scene.class_name, mesh_file=mesh_file),
        tmp_path / "rewritten",
    )

    assert cad_model.estimator_mesh_file == mesh_file


def test_two_cad_models_of_one_class_are_rejected(
    scene: FlatSurfaceScene,
    box_mesh_file: Path,
    pose_estimator: ScriptedPoseEstimator,
    tmp_path: Path,
):
    descriptor = FoundationPoseAnnotator.Descriptor()
    descriptor.parameters = descriptor.Parameters()
    descriptor.parameters.cad_models = [
        CadModelSource(class_name=scene.class_name, mesh_file=box_mesh_file),
        CadModelSource(class_name=scene.class_name, mesh_file=box_mesh_file),
    ]

    with pytest.raises(DuplicateCadModelClass):
        FoundationPoseAnnotator(descriptor=descriptor, pose_estimator=pose_estimator)


def test_estimator_loads_every_cad_model(
    annotator: FoundationPoseAnnotator, pose_estimator: ScriptedPoseEstimator
):
    assert pose_estimator.loaded_mesh_files == [
        cad_model.estimator_mesh_file for cad_model in annotator.cad_models.values()
    ]


# %% Matching across frames


def test_match_takes_the_most_overlapping_object_of_the_class(scene: FlatSurfaceScene):
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

    pose = previous_frame_poses.take_match(scene.class_name, scene.object_region, 0.1)

    assert pose is overlapping.pose
    assert previous_frame_poses.unmatched_objects == [distant]


def test_match_ignores_objects_of_other_classes(scene: FlatSurfaceScene):
    other_class = PosedObject(
        class_name="other",
        pose=object_pose_at_depth(0.5),
        mask_region=scene.object_region,
    )
    previous_frame_poses = PreviousFramePoses(unmatched_objects=[other_class])

    assert (
        previous_frame_poses.take_match(scene.class_name, scene.object_region, 0.1)
        is None
    )


def test_match_requires_the_minimum_overlap(scene: FlatSurfaceScene):
    distant = PosedObject(
        class_name=scene.class_name,
        pose=object_pose_at_depth(0.5),
        mask_region=Rect(pos=Point2D(x=0, y=0), width=8, height=8),
    )
    previous_frame_poses = PreviousFramePoses(unmatched_objects=[distant])

    assert (
        previous_frame_poses.take_match(scene.class_name, scene.object_region, 0.1)
        is None
    )
    assert previous_frame_poses.unmatched_objects == [distant]


# %% Pose update modes


@pytest.mark.parametrize(
    "mode, is_posed",
    [
        (PoseUpdateMode.ESTIMATE_UNPOSED_OBJECTS, True),
        (PoseUpdateMode.TRACK_POSED_OBJECTS, False),
    ],
)
def test_mode_leaves_object_without_pose(
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
    mode: PoseUpdateMode,
    is_posed: bool,
    scene: FlatSurfaceScene,
):
    annotator.descriptor.parameters.pose_update_mode = mode
    object_hypothesis = scene.add_object_hypothesis(annotator)
    if is_posed:
        add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))
    annotations_before = list(object_hypothesis.annotations)

    annotator.compute()

    assert object_hypothesis.annotations == annotations_before
    assert pose_estimator.estimated_observations == []
    assert pose_estimator.previous_poses == []


@pytest.mark.parametrize(
    "mode",
    [
        PoseUpdateMode.ESTIMATE_AND_TRACK,
        PoseUpdateMode.ESTIMATE_UNPOSED_OBJECTS,
        PoseUpdateMode.ESTIMATE_ALL_OBJECTS,
    ],
)
def test_mode_estimates_object_without_pose(
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
    mode: PoseUpdateMode,
    scene: FlatSurfaceScene,
):
    annotator.descriptor.parameters.pose_update_mode = mode
    object_hypothesis = scene.add_object_hypothesis(annotator)

    annotator.compute()

    (pose_annotation,) = pose_annotations(object_hypothesis)
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotation), pose_estimator.estimated_pose
    )
    assert pose_estimator.previous_poses == []


@pytest.mark.parametrize(
    "mode", [PoseUpdateMode.ESTIMATE_AND_TRACK, PoseUpdateMode.TRACK_POSED_OBJECTS]
)
def test_mode_tracks_object_from_its_pose_annotation(
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
    mode: PoseUpdateMode,
    scene: FlatSurfaceScene,
):
    annotator.descriptor.parameters.pose_update_mode = mode
    object_hypothesis = scene.add_object_hypothesis(annotator)
    known_pose = object_pose_at_depth(0.4)
    add_pose_annotation(object_hypothesis, known_pose)

    annotator.compute()

    (previous_pose,) = pose_estimator.previous_poses
    numpy.testing.assert_allclose(previous_pose, known_pose)
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotations(object_hypothesis)[-1]),
        pose_estimator.tracked_pose,
    )


def test_estimate_all_objects_ignores_existing_pose(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    annotator.descriptor.parameters.pose_update_mode = (
        PoseUpdateMode.ESTIMATE_ALL_OBJECTS
    )
    object_hypothesis = scene.add_object_hypothesis(annotator)
    add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))

    annotator.compute()

    assert pose_estimator.previous_poses == []
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotations(object_hypothesis)[-1]),
        pose_estimator.estimated_pose,
    )


def test_object_of_unknown_class_is_not_posed(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    object_hypothesis = scene.add_object_hypothesis(annotator, class_name="unknown")

    annotator.compute()

    assert pose_annotations(object_hypothesis) == []
    assert pose_estimator.estimated_observations == []


# %% Pose annotations


def test_pose_annotation_is_in_the_annotation_frame(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    camera_pose_in_world = get_transform_matrix_from_q(
        [0.0, 0.0, numpy.sin(numpy.pi / 4), numpy.cos(numpy.pi / 4)], [1.0, 2.0, 3.0]
    )
    annotator.descriptor.parameters.camera_pose_in_world = camera_pose_in_world
    object_hypothesis = scene.add_object_hypothesis(annotator)

    annotator.compute()

    (pose_annotation,) = pose_annotations(object_hypothesis)
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotation),
        camera_pose_in_world @ pose_estimator.estimated_pose,
        atol=1e-12,
    )
    assert pose_annotation.source == annotator.get_class_name()


def test_tracked_pose_overwrites_its_annotation_when_configured(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    annotator.descriptor.parameters.update_existing_pose_annotations = True
    object_hypothesis = scene.add_object_hypothesis(annotator)
    pose_annotation = add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))

    annotator.compute()

    assert pose_annotations(object_hypothesis) == [pose_annotation]
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotation), pose_estimator.tracked_pose
    )


def test_object_without_estimated_pose_gets_no_annotation(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    pose_estimator.estimated_pose = None
    object_hypothesis = scene.add_object_hypothesis(annotator)

    annotator.compute()

    assert pose_annotations(object_hypothesis) == []


# %% Tracking


def test_object_is_tracked_from_the_previous_frame(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    scene.add_object_hypothesis(annotator)
    annotator.compute()
    scene.next_frame(annotator)
    object_hypothesis = scene.add_object_hypothesis(annotator)

    annotator.compute()

    (previous_pose,) = pose_estimator.previous_poses
    numpy.testing.assert_allclose(previous_pose, pose_estimator.estimated_pose)
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotations(object_hypothesis)[0]),
        pose_estimator.tracked_pose,
    )


def test_object_is_not_tracked_across_frames_when_disabled(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    annotator.descriptor.parameters.track_across_frames = False
    scene.add_object_hypothesis(annotator)
    annotator.compute()
    scene.next_frame(annotator)
    scene.add_object_hypothesis(annotator)

    annotator.compute()

    assert pose_estimator.previous_poses == []
    assert len(pose_estimator.estimated_observations) == 2


def test_track_leaving_the_mask_is_estimated_anew(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    pose_estimator.tracked_pose = object_pose_at_depth(0.5)
    pose_estimator.tracked_pose[0, 3] = 0.5
    object_hypothesis = scene.add_object_hypothesis(annotator)
    add_pose_annotation(object_hypothesis, object_pose_at_depth(0.4))

    annotator.compute()

    assert len(pose_estimator.previous_poses) == 1
    numpy.testing.assert_allclose(
        annotation_matrix(pose_annotations(object_hypothesis)[-1]),
        pose_estimator.estimated_pose,
    )


# %% Depth image mask


def test_object_mask_is_placed_at_depth_image_resolution(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
    pose_estimator: ScriptedPoseEstimator,
):
    cas = annotator.get_cas()
    color_to_depth_ratio = ColorToDepthRatio(horizontal=0.5, vertical=0.5)
    cas.color_to_depth_ratio = color_to_depth_ratio
    depth_image = cas.get(CASViews.DEPTH_IMAGE)
    cas.set(CASViews.DEPTH_IMAGE, depth_image[::2, ::2].copy())
    scene.add_object_hypothesis(annotator)

    annotator.compute()

    (observation,) = pose_estimator.estimated_observations
    expected_region = color_to_depth_ratio.depth_region(scene.object_region)
    assert observation.mask.shape == cas.get(CASViews.DEPTH_IMAGE).shape
    assert observation.mask_region == expected_region
    assert observation.mask.sum() == expected_region.width * expected_region.height


# %% Visualization


def visualization_of(
    annotator: FoundationPoseAnnotator, frame: VisualizationFrame
) -> PoseVisualization:
    """
    :return: An empty visualization of the current frame of the annotator.
    """
    cas = annotator.get_cas()
    return PoseVisualization(
        color_image=cas.get_copy(CASViews.COLOR_IMAGE),
        color_to_depth_ratio=cas.color_to_depth_ratio,
        camera_model=annotator.rgbd_frame().camera_model,
        frame=frame,
    )


def test_mesh_frame_bounding_box_spans_the_mesh(annotator: FoundationPoseAnnotator):
    (cad_model,) = annotator.cad_models.values()
    visualization = visualization_of(annotator, VisualizationFrame.MESH)

    bounding_box = visualization.bounding_box(cad_model, object_pose_at_depth(0.5))

    numpy.testing.assert_allclose(bounding_box.extent, cad_model.mesh.extents)


def test_smallest_bounding_box_frame_uses_the_oriented_box(
    annotator: FoundationPoseAnnotator,
):
    (cad_model,) = annotator.cad_models.values()
    visualization = visualization_of(
        annotator, VisualizationFrame.SMALLEST_BOUNDING_BOX
    )

    bounding_box = visualization.bounding_box(cad_model, object_pose_at_depth(0.5))

    numpy.testing.assert_allclose(
        bounding_box.extent, cad_model.mesh.bounding_box_oriented.primitive.extents
    )


def test_posed_objects_are_drawn_and_named_in_the_3d_view(
    scene: FlatSurfaceScene,
    annotator: FoundationPoseAnnotator,
):
    annotator.descriptor.parameters.visualize_object_mesh = True
    scene.add_object_hypothesis(annotator)

    status = annotator.compute()

    output = annotator.get_annotator_output_struct()
    names = [geometry[DrawableGeometryKey.NAME] for geometry in output.geometries]
    assert status == Status.SUCCESS
    assert output.image.any()
    assert len(names) == len(set(names)) == 3
