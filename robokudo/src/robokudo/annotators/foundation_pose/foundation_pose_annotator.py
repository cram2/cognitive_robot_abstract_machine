"""
6D pose estimation and tracking of classified object hypotheses whose CAD models are known.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from timeit import default_timer

import cv2
import numpy
from py_trees.common import Status
from typing_extensions import TYPE_CHECKING

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.annotators.foundation_pose.cad_model import CadModel, CadModelSource
from robokudo.annotators.foundation_pose.exceptions import DuplicateCadModelClass
from robokudo.annotators.foundation_pose.pose_estimator import (
    ObjectObservation,
    ObjectPoseEstimator,
    RGBDFrame,
)
from robokudo.annotators.foundation_pose.tracking import (
    PosedObject,
    PreviousFramePoses,
)
from robokudo.annotators.foundation_pose.visualization import (
    PoseVisualization,
    VisualizationFrame,
)
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.transform import (
    get_quaternion_from_transform_matrix,
    get_transform_matrix_from_q,
    get_translation_from_transform_matrix,
)

if TYPE_CHECKING:
    import numpy.typing as numpy_typing


# %% Pose update modes


class PoseUpdateMode(Enum):
    """
    Which objects get a new pose, and whether known poses are tracked or estimated anew.

    An object has a known pose if it holds a :class:`PoseAnnotation` or, with tracking
    across frames, if it matches an object posed in the previous frame.
    """

    ESTIMATE_AND_TRACK = auto()
    """
    Objects with a known pose are tracked, the others are estimated.
    """

    ESTIMATE_UNPOSED_OBJECTS = auto()
    """
    Only objects without a :class:`PoseAnnotation` are estimated.
    """

    ESTIMATE_ALL_OBJECTS = auto()
    """
    Every object is estimated, ignoring the poses it already has.
    """

    TRACK_POSED_OBJECTS = auto()
    """
    Only objects with a known pose are tracked.
    """

    @property
    def tracks(self) -> bool:
        """
        Whether known poses are tracked.
        """
        return self in (
            PoseUpdateMode.ESTIMATE_AND_TRACK,
            PoseUpdateMode.TRACK_POSED_OBJECTS,
        )

    @property
    def estimates_unposed_objects(self) -> bool:
        """
        Whether objects without a known pose are estimated.
        """
        return self is not PoseUpdateMode.TRACK_POSED_OBJECTS


# %% Pose requests


@dataclass(eq=False)
class PoseRequest:
    """
    An object hypothesis whose pose is determined in the current frame.
    """

    object_hypothesis: ObjectHypothesis
    """
    The object hypothesis to pose.
    """

    cad_model: CadModel
    """
    CAD model of the class of :attr:`object_hypothesis`.
    """

    observation: ObjectObservation
    """
    The object hypothesis in the current frame.
    """

    previous_pose: numpy_typing.NDArray[numpy.float64] | None = None
    """
    Known pose of the mesh in the camera frame to track from, or ``None`` to estimate the
    pose.
    """

    pose_annotation: PoseAnnotation | None = None
    """
    Annotation of :attr:`object_hypothesis` that receives the new pose, or ``None`` to add
    a new annotation.
    """


@dataclass(eq=False)
class EstimatedPose:
    """
    The pose determined for a :class:`PoseRequest`.
    """

    request: PoseRequest
    """
    The request the pose was determined for.
    """

    pose: numpy_typing.NDArray[numpy.float64]
    """
    Pose of the mesh in the camera frame.
    """

    @property
    def label(self) -> str:
        """
        Text identifying the posed object in visualizations.
        """
        return (
            f"{self.request.object_hypothesis.id}: {self.request.cad_model.class_name}"
        )


# %% FoundationPose annotator


@dataclass(eq=False)
class FoundationPoseAnnotator(ThreadedAnnotator):
    """
    Estimate and track the 6D pose of classified object hypotheses with masks, using the
    CAD model of their class.

    Each posed object hypothesis gets a :class:`PoseAnnotation` of its mesh.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self) -> None:
                """
                Set the default pose estimation and visualization settings.
                """
                self.cad_models: list[CadModelSource] = []
                """
                Mesh files of the object classes to pose, one per class.
                """

                self.pose_update_mode: PoseUpdateMode = (
                    PoseUpdateMode.ESTIMATE_AND_TRACK
                )
                """
                Which objects get a new pose, and how.
                """

                self.update_existing_pose_annotations: bool = False
                """
                Whether a tracked pose overwrites the :class:`PoseAnnotation` it was
                tracked from instead of being added as a new one.
                """

                self.track_across_frames: bool = True
                """
                Whether objects without a :class:`PoseAnnotation` are tracked from the
                matching object posed in the previous frame.
                """

                self.minimum_track_overlap: float = 0.1
                """
                Intersection over union of the mask regions below which an object does not
                match an object of the previous frame.
                """

                self.maximum_track_center_offset: float = 0.25
                """
                Fraction of the mask region size by which the center of a tracked mesh may
                lie outside the mask region before the pose is estimated anew.
                """

                self.camera_pose_in_world: numpy_typing.NDArray[numpy.float64] = (
                    numpy.eye(4)
                )
                """
                Pose of the camera in the frame of the pose annotations.
                """

                self.depth_units_per_meter: float = 1000.0
                """
                Depth image values per meter.
                """

                self.rewritten_mesh_directory: Path = (
                    Path(tempfile.gettempdir()) / __name__
                )
                """
                Directory for meshes that have to be merged or scaled before estimation.
                """

                self.visualization_frame: VisualizationFrame = VisualizationFrame.MESH
                """
                Frame the axes and the bounding box of a posed object are drawn in.
                """

                self.center_axes_on_bounding_box: bool = False
                """
                Whether the drawn axes start at the center of the smallest oriented
                bounding box instead of the origin of the mesh.
                """

                self.visualize_object_mesh: bool = False
                """
                Whether the posed mesh is shown in the 3D view.
                """

        parameters = Parameters()
        """
        Pose estimation and visualization defaults.
        """

    pose_estimator: ObjectPoseEstimator = field(kw_only=True)
    """
    The model estimating and tracking the poses.
    """

    name: str = __qualname__
    """
    Name of the behaviour-tree node.
    """

    descriptor: Descriptor = field(default_factory=Descriptor)
    """
    Pose estimation and visualization configuration.
    """

    cad_models: dict[str, CadModel] = field(init=False, repr=False)
    """
    CAD models by class name.
    """

    previous_frame_objects: list[PosedObject] = field(
        init=False, default_factory=list, repr=False
    )
    """
    Objects posed in the previous frame.
    """

    def __post_init__(self) -> None:
        """
        Initialize the threaded behaviour, load the CAD models and hand their meshes to the
        pose estimator.

        :raises DuplicateCadModelClass: If a class has more than one CAD model.
        """
        super().__init__(self.name, self.descriptor)
        parameters = self.descriptor.parameters
        self.cad_models = {}
        for source in parameters.cad_models:
            if source.class_name in self.cad_models:
                raise DuplicateCadModelClass(class_name=source.class_name)
            self.cad_models[source.class_name] = CadModel.from_source(
                source, parameters.rewritten_mesh_directory
            )
        self.pose_estimator.load_meshes(
            [cad_model.estimator_mesh_file for cad_model in self.cad_models.values()]
        )

    @catch_and_raise_to_blackboard
    def compute(self) -> Status:
        """
        Pose the classified object hypotheses of the CAS and visualize the poses.

        :return: ``Status.SUCCESS`` after processing the frame.
        """
        start_time = default_timer()
        frame = self.rgbd_frame()
        previous_frame_poses = PreviousFramePoses(
            unmatched_objects=list(self.previous_frame_objects)
        )
        requests = [
            request
            for object_hypothesis in self.get_cas().filter_annotations_by_type(
                ObjectHypothesis
            )
            if (
                request := self.pose_request(
                    object_hypothesis, frame, previous_frame_poses
                )
            )
            is not None
        ]
        estimated_poses = [
            EstimatedPose(request=request, pose=pose)
            for request in requests
            if (pose := self.object_pose(request)) is not None
        ]
        for estimated_pose in estimated_poses:
            self.annotate_pose(estimated_pose)
        self.previous_frame_objects = [
            PosedObject(
                class_name=estimated_pose.request.cad_model.class_name,
                pose=estimated_pose.pose,
                mask_region=estimated_pose.request.observation.mask_region,
            )
            for estimated_pose in estimated_poses
        ]
        if self.descriptor.parameters.global_with_visualization:
            self.visualize(frame, estimated_poses)
        self.feedback_message = (
            f"Posed {len(estimated_poses)} of {len(requests)} objects in "
            f"{default_timer() - start_time:.4f}s"
        )
        return Status.SUCCESS

    def rgbd_frame(self) -> RGBDFrame:
        """
        :return: The color and depth image of the CAS at depth image resolution.
        """
        cas = self.get_cas()
        depth_image = (
            cas.get(CASViews.DEPTH_IMAGE).astype(numpy.float32)
            / self.descriptor.parameters.depth_units_per_meter
        )
        depth_height, depth_width = depth_image.shape[:2]
        color_image = cv2.resize(
            cas.get(CASViews.COLOR_IMAGE), (depth_width, depth_height)
        )
        color_to_depth_ratio = cas.color_to_depth_ratio
        camera_model = cas.require_camera_observation().effective_camera_model.scaled(
            scale_x=color_to_depth_ratio.horizontal,
            scale_y=color_to_depth_ratio.vertical,
        )
        return RGBDFrame(
            rgb_image=cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB),
            depth_image=depth_image,
            camera_model=camera_model,
        )

    def pose_request(
        self,
        object_hypothesis: ObjectHypothesis,
        frame: RGBDFrame,
        previous_frame_poses: PreviousFramePoses,
    ) -> PoseRequest | None:
        """
        Decide whether and how an object hypothesis gets a pose in this frame.

        :param object_hypothesis: The object hypothesis.
        :param frame: The current frame.
        :param previous_frame_poses: Objects of the previous frame not yet matched; a
            matching object is removed.
        :return: The request to pose the object hypothesis, or ``None`` if it gets no pose.
        """
        classifications = [
            annotation
            for annotation in object_hypothesis.annotations
            if isinstance(annotation, Classification)
        ]
        if object_hypothesis.roi.mask is None or not classifications:
            return None
        classification = max(
            classifications, key=lambda candidate: candidate.confidence
        )
        cad_model = self.cad_models.get(classification.classname)
        if cad_model is None:
            return None

        parameters = self.descriptor.parameters
        mode = parameters.pose_update_mode
        observation = ObjectObservation(
            frame=frame, mask=self.depth_image_mask(object_hypothesis, frame)
        )
        pose_annotation = next(
            (
                annotation
                for annotation in object_hypothesis.annotations
                if isinstance(annotation, PoseAnnotation)
            ),
            None,
        )
        if pose_annotation is not None and mode.tracks:
            return PoseRequest(
                object_hypothesis=object_hypothesis,
                cad_model=cad_model,
                observation=observation,
                previous_pose=numpy.linalg.inv(parameters.camera_pose_in_world)
                @ get_transform_matrix_from_q(
                    pose_annotation.rotation, pose_annotation.translation
                ),
                pose_annotation=(
                    pose_annotation
                    if parameters.update_existing_pose_annotations
                    else None
                ),
            )
        if (
            pose_annotation is not None
            and mode is not PoseUpdateMode.ESTIMATE_ALL_OBJECTS
        ):
            return None

        previous_pose = None
        if pose_annotation is None and mode.tracks and parameters.track_across_frames:
            previous_pose = previous_frame_poses.take_match(
                class_name=cad_model.class_name,
                mask_region=observation.mask_region,
                minimum_overlap=parameters.minimum_track_overlap,
            )
        if previous_pose is None and not mode.estimates_unposed_objects:
            return None
        return PoseRequest(
            object_hypothesis=object_hypothesis,
            cad_model=cad_model,
            observation=observation,
            previous_pose=previous_pose,
        )

    def depth_image_mask(
        self, object_hypothesis: ObjectHypothesis, frame: RGBDFrame
    ) -> numpy_typing.NDArray[numpy.bool_]:
        """
        :param object_hypothesis: Object hypothesis whose mask covers its region of
            interest in the color image.
        :param frame: The current frame.
        :return: The mask of the object hypothesis over the whole depth image.
        """
        region = self.get_cas().color_to_depth_ratio.depth_region(
            object_hypothesis.roi.roi
        )
        region_mask = cv2.resize(
            object_hypothesis.roi.mask.astype(numpy.uint8),
            (region.width, region.height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(numpy.bool_)
        mask = numpy.zeros(frame.depth_image.shape[:2], dtype=numpy.bool_)
        region_in_image = mask[
            region.pos.y : region.pos.y + region.height,
            region.pos.x : region.pos.x + region.width,
        ]
        region_in_image[:] = region_mask[
            : region_in_image.shape[0], : region_in_image.shape[1]
        ]
        return mask

    def object_pose(
        self, request: PoseRequest
    ) -> numpy_typing.NDArray[numpy.float64] | None:
        """
        Track the pose of a request from its previous pose if the tracked pose is
        plausible, and estimate it otherwise.

        :param request: The object hypothesis to pose.
        :return: Pose of the mesh in the camera frame, or ``None`` if none was found.
        """
        self.pose_estimator.select_mesh(request.cad_model.estimator_mesh_file)
        if request.previous_pose is not None:
            tracked_pose = self.pose_estimator.track_pose(
                request.observation, request.previous_pose
            )
            if self.is_plausible_track(request, tracked_pose):
                return tracked_pose
            self.rk_logger.debug(
                f"Tracked pose of {request.cad_model.class_name} left its mask, "
                f"estimating it anew."
            )
        return self.pose_estimator.estimate_pose(request.observation)

    def is_plausible_track(
        self, request: PoseRequest, tracked_pose: numpy_typing.NDArray[numpy.float64]
    ) -> bool:
        """
        :param request: The tracked object hypothesis.
        :param tracked_pose: Tracked pose of the mesh in the camera frame.
        :return: Whether the center of the tracked mesh lies in front of the camera and
            projects into the mask region, enlarged by the maximum track center offset.
        """
        center = tracked_pose @ numpy.append(
            request.cad_model.mesh.bounds.mean(axis=0), 1.0
        )
        if center[2] <= 0:
            return False
        camera_model = request.observation.frame.camera_model
        x, y, _ = camera_model.intrinsic_matrix @ (center[:3] / center[2])
        region = request.observation.mask_region
        left, top, right, bottom = region.get_corner_points()
        offset = self.descriptor.parameters.maximum_track_center_offset
        horizontal_margin = region.width * offset
        vertical_margin = region.height * offset
        return (
            left - horizontal_margin <= x <= right + horizontal_margin
            and top - vertical_margin <= y <= bottom + vertical_margin
        )

    def annotate_pose(self, estimated_pose: EstimatedPose) -> None:
        """
        Write an estimated pose, in the frame of the pose annotations, into the pose
        annotation of its request, adding one if the request has none.

        :param estimated_pose: The estimated pose.
        """
        request = estimated_pose.request
        pose_in_world = (
            self.descriptor.parameters.camera_pose_in_world @ estimated_pose.pose
        )
        pose_annotation = request.pose_annotation
        if pose_annotation is None:
            pose_annotation = PoseAnnotation()
            request.object_hypothesis.annotations.append(pose_annotation)
        pose_annotation.source = self.get_class_name()
        pose_annotation.translation = get_translation_from_transform_matrix(
            pose_in_world
        ).tolist()
        pose_annotation.rotation = get_quaternion_from_transform_matrix(
            pose_in_world
        ).tolist()

    def visualize(self, frame: RGBDFrame, estimated_poses: list[EstimatedPose]) -> None:
        """
        Show the posed CAD models in the color image and, with the point cloud of the CAS,
        in the 3D view.

        :param frame: The current frame.
        :param estimated_poses: The poses of this frame.
        """
        cas = self.get_cas()
        parameters = self.descriptor.parameters
        visualization = PoseVisualization(
            color_image=cas.get_copy(CASViews.COLOR_IMAGE),
            color_to_depth_ratio=cas.color_to_depth_ratio,
            camera_model=frame.camera_model,
            frame=parameters.visualization_frame,
            center_axes_on_bounding_box=parameters.center_axes_on_bounding_box,
            show_object_mesh=parameters.visualize_object_mesh,
        )
        for estimated_pose in estimated_poses:
            visualization.draw(
                estimated_pose.request.cad_model,
                estimated_pose.pose,
                estimated_pose.label,
            )
        if cas.cloud is not None:
            visualization.add_geometry(CASViews.CLOUD, cas.cloud)
        output = self.get_annotator_output_struct()
        output.set_image(visualization.color_image)
        output.set_geometries(visualization.geometries)
