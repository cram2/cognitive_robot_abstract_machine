"""
6D pose estimation and tracking of classified object hypotheses whose meshes are known.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto
from functools import cached_property
from pathlib import Path
from timeit import default_timer

import cv2
import numpy
import open3d
import trimesh
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix, Pose
from py_trees.common import Status
from semantic_digital_twin.world_description.geometry import Color, Mesh
from typing_extensions import TYPE_CHECKING, Self

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.annotators.exceptions import DuplicateObjectClassMesh
from robokudo.annotators.object_observation import ColorAndDepthFrame, ObjectObservation
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.cv import ColorToDepthRatio, Rect
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.comparators import RoiComparator
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image
from robokudo.utils.decorators import record_time
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.o3d_helper import create_oriented_bounding_box, trimesh_to_o3d_mesh

if TYPE_CHECKING:
    import numpy.typing as numpy_typing
    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel

    from robokudo.annotators.object_pose_estimator import MeshBasedPoseEstimator


# %% Object class meshes


@dataclass
class ObjectClassMesh:
    """
    The mesh showing the objects of one class.
    """

    class_name: str
    """
    Class name of the :class:`robokudo.types.annotation.Classification` the objects are
    detected with.
    """

    shape: Mesh
    """
    The mesh, whose :attr:`Mesh.scale` brings file coordinates without a declared unit to
    meters.
    """

    @cached_property
    def estimator_mesh_file(self) -> Path:
        """
        File holding exactly the mesh of :attr:`shape`, as one mesh in meters.
        """
        source_file = self.shape.local_file
        source_scene = trimesh.load_scene(source_file, process=False)
        if len(source_scene.graph.nodes_geometry) == 1 and numpy.allclose(
            source_scene.bounds, self.shape.mesh.bounds
        ):
            return source_file
        return Mesh.from_trimesh(self.shape.mesh).local_file


@dataclass
class ObjectClassMeshes:
    """
    The meshes of the object classes to pose, at most one per class.
    """

    meshes_by_class_name: dict[str, ObjectClassMesh] = field(default_factory=dict)
    """
    Meshes by the name of their class.
    """

    @classmethod
    def from_class_meshes(cls, class_meshes: list[ObjectClassMesh]) -> Self:
        """
        :param class_meshes: The meshes.
        :return: The meshes by the name of their class.
        :raises DuplicateObjectClassMesh: If a class has more than one mesh.
        """
        meshes_by_class_name = {}
        for class_mesh in class_meshes:
            if class_mesh.class_name in meshes_by_class_name:
                raise DuplicateObjectClassMesh(class_name=class_mesh.class_name)
            meshes_by_class_name[class_mesh.class_name] = class_mesh
        return cls(meshes_by_class_name=meshes_by_class_name)

    @property
    def estimator_mesh_files(self) -> list[Path]:
        """
        The file of every mesh, to hand to a pose estimator.
        """
        return [
            class_mesh.estimator_mesh_file
            for class_mesh in self.meshes_by_class_name.values()
        ]


# %% Tracking across frames


@dataclass(eq=False)
class PosedObject:
    """
    An object whose pose was determined in a frame.

    Posed objects are equal only to themselves.
    """

    class_name: str
    """
    Class name of the object's mesh.
    """

    pose: Pose
    """
    Pose of the mesh in the camera frame.
    """

    mask_region: Rect
    """
    The smallest rectangle containing the object's mask, in depth image pixels.
    """


@dataclass
class PreviousFramePoses:
    """
    Objects posed in the previous frame that are not yet matched to an object of the
    current frame.
    """

    unmatched_objects: list[PosedObject] = field(default_factory=list)
    """
    Posed objects still available for matching.
    """

    def take_match(
        self, class_name: str, mask_region: Rect, minimum_overlap: float
    ) -> Pose | None:
        """
        Remove and return the pose of the unmatched object of a class whose mask region
        overlaps most with a mask region of the current frame.

        :param class_name: Class name of the object of the current frame.
        :param mask_region: Mask region of the object of the current frame.
        :param minimum_overlap: Intersection over union below which mask regions do not
            match.
        :return: The pose of the matching object, or ``None`` if none overlaps enough.
        """
        candidates = [
            posed_object
            for posed_object in self.unmatched_objects
            if posed_object.class_name == class_name
        ]
        if not candidates:
            return None
        region_comparator = RoiComparator(weight=1.0)
        overlaps = [
            region_comparator.compute_similarity(candidate.mask_region, mask_region)
            for candidate in candidates
        ]
        best_overlap = max(overlaps)
        if best_overlap < minimum_overlap:
            return None
        match = candidates[overlaps.index(best_overlap)]
        self.unmatched_objects.remove(match)
        return match.pose


# %% Visualization


class VisualizationFrame(Enum):
    """
    Frame the axes and the bounding box of a posed mesh are drawn in.
    """

    MESH = auto()
    """
    The frame of the mesh, with the bounding box aligned to its axes.
    """

    SMALLEST_BOUNDING_BOX = auto()
    """
    The frame of the smallest oriented bounding box of the mesh.
    """


class DrawableGeometryKey(StrEnum):
    """
    Keys of a named geometry in the 3D view, as :func:`open3d.visualization.draw` reads
    them.
    """

    NAME = "name"
    """
    Unique name of the geometry.
    """

    GEOMETRY = "geometry"
    """
    The geometry to draw.
    """


@dataclass
class NamedGeometry:
    """
    A geometry of the 3D view with its unique name.
    """

    geometry_name: str
    """
    Unique name of the geometry.
    """

    geometry: open3d.geometry.Geometry3D
    """
    The geometry, in the camera frame at depth image resolution.
    """

    def to_drawable(
        self,
    ) -> dict[DrawableGeometryKey, str | open3d.geometry.Geometry3D]:
        """
        :return: The geometry in the form :func:`open3d.visualization.draw` reads.
        """
        return {
            DrawableGeometryKey.NAME: self.geometry_name,
            DrawableGeometryKey.GEOMETRY: self.geometry,
        }


@dataclass
class PoseVisualization:
    """
    The color image and the 3D geometries showing posed meshes.
    """

    color_image: numpy_typing.NDArray[numpy.uint8]
    """
    Color image in BGR channel order, drawn into.
    """

    color_to_depth_ratio: ColorToDepthRatio
    """
    Size of the depth image relative to :attr:`color_image`.
    """

    camera_model: PinholeCameraModel
    """
    Calibration of the camera at depth image resolution.
    """

    frame: VisualizationFrame = VisualizationFrame.MESH
    """
    Frame the axes and the bounding box are drawn in.
    """

    center_axes_on_bounding_box: bool = False
    """
    Whether the axes start at the center of the smallest oriented bounding box instead of
    the origin of the mesh, when drawn in :attr:`VisualizationFrame.MESH`.
    """

    show_object_mesh: bool = False
    """
    Whether the posed mesh is added to the 3D view.
    """

    object_mesh_color: Color = field(
        default_factory=lambda: Color.from_rgba(red=0.6, green=0.1, blue=0.9)
    )
    """
    Color of the posed mesh in the 3D view.
    """

    axis_length: float = 0.1
    """
    Length of the drawn axes, in meters.
    """

    line_thickness: int = 2
    """
    Thickness of the lines drawn into :attr:`color_image`, in pixels.
    """

    label_font_scale: float = 0.6
    """
    Scale of the label font relative to its base size.
    """

    geometries: list[NamedGeometry] = field(default_factory=list)
    """
    Named geometries of the 3D view.
    """

    def draw(self, shape: Mesh, pose: Pose, label: str) -> None:
        """
        Draw a posed mesh into the color image and add it to the 3D view.

        :param shape: The mesh.
        :param pose: Pose of the mesh in the camera frame.
        :param label: Unique text identifying the object.
        """
        bounding_box = self.bounding_box(shape, pose)
        axes_pose = self.axes_pose(shape, pose)
        self.draw_bounding_box(bounding_box, label)
        self.draw_axes(axes_pose)

        axes = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.axis_length
        )
        axes.transform(axes_pose.to_np())
        self.geometries.append(NamedGeometry(f"{label} bounding box", bounding_box))
        self.geometries.append(NamedGeometry(f"{label} axes", axes))
        if self.show_object_mesh:
            self.geometries.append(
                NamedGeometry(f"{label} mesh", self.posed_mesh(shape, pose))
            )

    def bounding_box(
        self, shape: Mesh, pose: Pose
    ) -> open3d.geometry.OrientedBoundingBox:
        """
        :param shape: The mesh.
        :param pose: Pose of the mesh in the camera frame.
        :return: The bounding box of the posed mesh in :attr:`frame`.
        """
        if self.frame is VisualizationFrame.SMALLEST_BOUNDING_BOX:
            smallest_box = shape.mesh.bounding_box_oriented.primitive
            box_pose = pose.homogeneous_matrix @ HomogeneousTransformationMatrix(
                data=smallest_box.transform
            )
            return create_oriented_bounding_box(smallest_box.extents, box_pose.to_np())
        mesh_box = shape.local_frame_bounding_box
        box_pose = (
            pose.homogeneous_matrix
            @ HomogeneousTransformationMatrix.from_point_rotation_matrix(
                point=mesh_box.center
            )
        )
        return create_oriented_bounding_box(mesh_box.scale.to_np(), box_pose.to_np())

    def axes_pose(self, shape: Mesh, pose: Pose) -> HomogeneousTransformationMatrix:
        """
        :param shape: The mesh.
        :param pose: Pose of the mesh in the camera frame.
        :return: Pose of the drawn axes in the camera frame.
        """
        smallest_box_pose = HomogeneousTransformationMatrix(
            data=shape.mesh.bounding_box_oriented.primitive.transform
        )
        if self.frame is VisualizationFrame.SMALLEST_BOUNDING_BOX:
            return pose.homogeneous_matrix @ smallest_box_pose
        if not self.center_axes_on_bounding_box:
            return pose.homogeneous_matrix
        return pose.homogeneous_matrix @ smallest_box_pose.translation_matrix

    def draw_bounding_box(
        self, bounding_box: open3d.geometry.OrientedBoundingBox, label: str
    ) -> None:
        """
        Draw the edges of a bounding box in front of the camera into the color image,
        labeled above its top left corner.

        :param bounding_box: Bounding box in the camera frame.
        :param label: Text drawn above the bounding box.
        """
        edges = open3d.geometry.LineSet.create_from_oriented_bounding_box(bounding_box)
        corners = numpy.asarray(edges.points)
        for start, end in numpy.asarray(edges.lines):
            self.draw_line(corners[start], corners[end], Color.GREEN())
        corners_in_front = corners[corners[:, 2] > 0]
        if len(corners_in_front) == 0:
            return
        left, top = self.color_pixels(corners_in_front).min(axis=0)
        cv2.putText(
            self.color_image,
            label,
            (int(left), int(top)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.label_font_scale,
            self.blue_green_red_bytes(Color.WHITE()),
            lineType=cv2.LINE_AA,
        )

    def draw_axes(self, axes_pose: HomogeneousTransformationMatrix) -> None:
        """
        Draw the x, y and z axes of a pose in red, green and blue into the color image,
        unless the end of an axis is not in front of the camera.

        :param axes_pose: Pose of the axes in the camera frame.
        """
        pose_matrix = axes_pose.to_np()
        rotation, translation = pose_matrix[:3, :3], pose_matrix[:3, 3]
        axis_end_depths = translation[2] + rotation[2] * self.axis_length
        if translation[2] <= 0 or (axis_end_depths <= 0).any():
            return
        color_camera_model = self.camera_model.scaled(
            scale_x=1 / self.color_to_depth_ratio.horizontal,
            scale_y=1 / self.color_to_depth_ratio.vertical,
        )
        rotation_vector, _ = cv2.Rodrigues(rotation)
        cv2.drawFrameAxes(
            self.color_image,
            color_camera_model.intrinsic_matrix,
            None,
            rotation_vector,
            translation,
            self.axis_length,
            self.line_thickness,
        )

    def draw_line(
        self,
        start: numpy_typing.NDArray[numpy.float64],
        end: numpy_typing.NDArray[numpy.float64],
        color: Color,
    ) -> None:
        """
        Draw a line into the color image, unless an end of it is not in front of the
        camera.

        :param start: Point the line starts at, in the camera frame.
        :param end: Point the line ends at, in the camera frame.
        :param color: Color of the line.
        """
        if start[2] <= 0 or end[2] <= 0:
            return
        start_pixel, end_pixel = self.color_pixels(numpy.array([start, end]))
        cv2.line(
            self.color_image,
            (int(start_pixel[0]), int(start_pixel[1])),
            (int(end_pixel[0]), int(end_pixel[1])),
            self.blue_green_red_bytes(color),
            self.line_thickness,
            cv2.LINE_AA,
        )

    def color_pixels(
        self, points: numpy_typing.NDArray[numpy.float64]
    ) -> numpy_typing.NDArray[numpy.int_]:
        """
        Project points in front of the camera into the color image.

        :param points: Points in the camera frame, one per row.
        :return: The (x, y) color image pixel of each point, one per row.
        """
        projected = points @ self.camera_model.intrinsic_matrix.T
        depth_pixels = projected[:, :2] / projected[:, 2:]
        return self.color_to_depth_ratio.color_pixels(depth_pixels)

    def posed_mesh(self, shape: Mesh, pose: Pose) -> open3d.geometry.TriangleMesh:
        """
        :param shape: The mesh.
        :param pose: Pose of the mesh in the camera frame.
        :return: The mesh in the camera frame, painted in :attr:`object_mesh_color`.
        """
        posed_mesh = trimesh_to_o3d_mesh(shape.mesh)
        posed_mesh.paint_uniform_color(self.object_mesh_color.to_rgb())
        posed_mesh.transform(pose.to_np())
        return posed_mesh

    @staticmethod
    def blue_green_red_bytes(color: Color) -> tuple[int, int, int]:
        """
        :param color: A color.
        :return: The color as OpenCV byte channels in blue, green, red order.
        """
        maximum = Color.opencv_maximum_channel_value
        red, green, blue = color.to_rgb()
        return round(blue * maximum), round(green * maximum), round(red * maximum)


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

    class_mesh: ObjectClassMesh
    """
    Mesh of the class of :attr:`object_hypothesis`.
    """

    object_observation: ObjectObservation
    """
    The object hypothesis in the current frame.
    """

    previous_pose: Pose | None = None
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

    pose: Pose
    """
    Pose of the mesh in the camera frame.
    """

    @property
    def label(self) -> str:
        """
        Text identifying the posed object in visualizations.
        """
        return (
            f"{self.request.object_hypothesis.id}: {self.request.class_mesh.class_name}"
        )


@dataclass(eq=False)
class PoseCandidate:
    """
    An object hypothesis that can be posed, before deciding whether and how.
    """

    object_hypothesis: ObjectHypothesis
    """
    The object hypothesis.
    """

    class_mesh: ObjectClassMesh
    """
    Mesh of the class of :attr:`object_hypothesis`.
    """

    object_observation: ObjectObservation
    """
    The object hypothesis in the current frame.
    """

    pose_annotation: PoseAnnotation | None
    """
    The latest pose annotation the object hypothesis holds, in the camera frame.
    """

    previous_frame_poses: PreviousFramePoses
    """
    Objects of the previous frame not yet matched.
    """

    def to_request(
        self,
        previous_pose: Pose | None = None,
        pose_annotation: PoseAnnotation | None = None,
    ) -> PoseRequest:
        """
        :param previous_pose: Pose to track from, or ``None`` to estimate the pose.
        :param pose_annotation: Annotation receiving the new pose, or ``None`` to add one.
        :return: The request to pose the object hypothesis.
        """
        return PoseRequest(
            object_hypothesis=self.object_hypothesis,
            class_mesh=self.class_mesh,
            object_observation=self.object_observation,
            previous_pose=previous_pose,
            pose_annotation=pose_annotation,
        )


# %% Pose update modes


@dataclass
class PoseUpdateMode(ABC):
    """
    Decides which objects get a new pose, and whether known poses are tracked or
    estimated anew.
    """

    @abstractmethod
    def pose_request(self, candidate: PoseCandidate) -> PoseRequest | None:
        """
        :param candidate: The object hypothesis to decide on; a track from the previous
            frame removes the matching object from it.
        :return: The request to pose it, or ``None`` if it gets no pose.
        """


@dataclass
class EstimateUnposedObjects(PoseUpdateMode):
    """
    Only objects without a :class:`PoseAnnotation` are estimated.
    """

    def pose_request(self, candidate: PoseCandidate) -> PoseRequest | None:
        if candidate.pose_annotation is not None:
            return None
        return candidate.to_request()


@dataclass
class EstimateAllObjects(PoseUpdateMode):
    """
    Every object is estimated, ignoring the poses it already has.
    """

    def pose_request(self, candidate: PoseCandidate) -> PoseRequest | None:
        return candidate.to_request()


@dataclass
class TrackingPoseUpdateMode(PoseUpdateMode, ABC):
    """
    A mode that tracks objects with a known pose: one from their :class:`PoseAnnotation`
    or the pose of the matching object of the previous frame.
    """

    update_existing_pose_annotations: bool = False
    """
    Whether a tracked pose overwrites the :class:`PoseAnnotation` it was tracked from
    instead of being added as a new one.
    """

    track_across_frames: bool = True
    """
    Whether objects without a :class:`PoseAnnotation` are tracked from the matching object
    posed in the previous frame.
    """

    minimum_track_overlap: float = 0.1
    """
    Intersection over union of the mask regions below which an object does not match an
    object of the previous frame.
    """

    def annotation_tracking_request(self, candidate: PoseCandidate) -> PoseRequest:
        """
        :param candidate: An object hypothesis holding a pose annotation.
        :return: The request to track it from that annotation.
        """
        pose_annotation = candidate.pose_annotation
        return candidate.to_request(
            previous_pose=Pose.from_xyz_quaternion(
                *pose_annotation.translation, *pose_annotation.rotation
            ),
            pose_annotation=(
                pose_annotation if self.update_existing_pose_annotations else None
            ),
        )

    def previous_frame_tracking_request(
        self, candidate: PoseCandidate
    ) -> PoseRequest | None:
        """
        :param candidate: An object hypothesis without a pose annotation.
        :return: The request to track it from the matching object of the previous frame,
            or ``None`` if there is none to track from.
        """
        if not self.track_across_frames:
            return None
        previous_pose = candidate.previous_frame_poses.take_match(
            class_name=candidate.class_mesh.class_name,
            mask_region=candidate.object_observation.mask_region,
            minimum_overlap=self.minimum_track_overlap,
        )
        if previous_pose is None:
            return None
        return candidate.to_request(previous_pose=previous_pose)


@dataclass
class EstimateAndTrack(TrackingPoseUpdateMode):
    """
    Objects with a known pose are tracked, the others are estimated.
    """

    def pose_request(self, candidate: PoseCandidate) -> PoseRequest | None:
        if candidate.pose_annotation is not None:
            return self.annotation_tracking_request(candidate)
        return self.previous_frame_tracking_request(candidate) or candidate.to_request()


@dataclass
class TrackPosedObjects(TrackingPoseUpdateMode):
    """
    Only objects with a known pose are tracked.
    """

    def pose_request(self, candidate: PoseCandidate) -> PoseRequest | None:
        if candidate.pose_annotation is not None:
            return self.annotation_tracking_request(candidate)
        return self.previous_frame_tracking_request(candidate)


# %% FoundationPose annotator


@dataclass(eq=False)
class FoundationPoseAnnotator(ThreadedAnnotator):
    """
    Estimate and track the 6D pose of classified object hypotheses with masks, using the
    mesh of their class.

    Each posed object hypothesis gets a :class:`PoseAnnotation` of its mesh, in the camera
    frame.
    """

    class Descriptor(BaseAnnotator.Descriptor):
        class Parameters:
            def __init__(self) -> None:
                """
                Set the default pose estimation and visualization settings.
                """
                self.class_meshes: list[ObjectClassMesh] = []
                """
                Meshes of the object classes to pose, one per class.
                """

                self.pose_update_mode: PoseUpdateMode = EstimateAndTrack()
                """
                Which objects get a new pose, and how.
                """

                self.maximum_track_center_offset: float = 0.25
                """
                Fraction of the mask region size by which the center of a tracked mesh may
                lie outside the mask region before the pose is estimated anew.
                """

                self.depth_units_per_meter: float = 1000.0
                """
                Depth image values per meter.
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

    pose_estimator: MeshBasedPoseEstimator = field(kw_only=True)
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

    class_meshes: ObjectClassMeshes = field(init=False, repr=False)
    """
    Meshes of the object classes to pose.
    """

    previous_frame_objects: list[PosedObject] = field(
        init=False, default_factory=list, repr=False
    )
    """
    Objects posed in the previous frame.
    """

    def __post_init__(self) -> None:
        """
        Initialize the threaded behaviour and hand the meshes to the pose estimator.

        :raises DuplicateObjectClassMesh: If a class has more than one mesh.
        """
        super().__init__(self.name, self.descriptor)
        self.class_meshes = ObjectClassMeshes.from_class_meshes(
            self.descriptor.parameters.class_meshes
        )
        self.pose_estimator.load_meshes(self.class_meshes.estimator_mesh_files)

    @catch_and_raise_to_blackboard
    @record_time
    def compute(self) -> Status:
        """
        Pose the classified object hypotheses of the CAS and visualize the poses.

        :return: ``Status.SUCCESS`` after processing the frame.
        """
        start_time = default_timer()
        frame = self.color_and_depth_frame()
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
                class_name=estimated_pose.request.class_mesh.class_name,
                pose=estimated_pose.pose,
                mask_region=estimated_pose.request.object_observation.mask_region,
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

    def color_and_depth_frame(self) -> ColorAndDepthFrame:
        """
        :return: The color and depth image of the CAS at depth image resolution.
        """
        cas = self.get_cas()
        depth_image = (
            cas.get(CASViews.DEPTH_IMAGE).astype(numpy.float32)
            / self.descriptor.parameters.depth_units_per_meter
        )
        color_image = get_scaled_color_image_for_depth_image(
            cas, cas.get(CASViews.COLOR_IMAGE)
        )
        color_to_depth_ratio = cas.color_to_depth_ratio
        camera_model = cas.require_camera_observation().effective_camera_model.scaled(
            scale_x=color_to_depth_ratio.horizontal,
            scale_y=color_to_depth_ratio.vertical,
        )
        return ColorAndDepthFrame(
            color_image=cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB),
            depth_image=depth_image,
            camera_model=camera_model,
        )

    def pose_request(
        self,
        object_hypothesis: ObjectHypothesis,
        frame: ColorAndDepthFrame,
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
        cas = self.get_cas()
        classifications = cas.filter_by_type(
            Classification, object_hypothesis.annotations
        )
        if object_hypothesis.roi.mask is None or not classifications:
            return None
        classification = max(
            classifications, key=lambda candidate: candidate.confidence
        )
        class_mesh = self.class_meshes.meshes_by_class_name.get(
            classification.classname
        )
        if class_mesh is None:
            return None
        pose_annotations = cas.filter_by_type(
            PoseAnnotation, object_hypothesis.annotations
        )
        return self.descriptor.parameters.pose_update_mode.pose_request(
            PoseCandidate(
                object_hypothesis=object_hypothesis,
                class_mesh=class_mesh,
                object_observation=ObjectObservation(
                    frame=frame, mask=self.depth_image_mask(object_hypothesis, frame)
                ),
                pose_annotation=pose_annotations[-1] if pose_annotations else None,
                previous_frame_poses=previous_frame_poses,
            )
        )

    def depth_image_mask(
        self, object_hypothesis: ObjectHypothesis, frame: ColorAndDepthFrame
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

    def object_pose(self, request: PoseRequest) -> Pose | None:
        """
        Track the pose of a request from its previous pose if the tracked pose is
        plausible, and estimate it otherwise.

        :param request: The object hypothesis to pose.
        :return: Pose of the mesh in the camera frame, or ``None`` if none was found.
        """
        self.pose_estimator.select_mesh(request.class_mesh.estimator_mesh_file)
        if request.previous_pose is not None:
            tracked_pose = self.pose_estimator.track_pose(
                request.object_observation, request.previous_pose
            )
            if self.is_plausible_track(request, tracked_pose):
                return tracked_pose
            self.rk_logger.debug(
                f"Tracked pose of {request.class_mesh.class_name} left its mask, "
                f"estimating it anew."
            )
        return self.pose_estimator.estimate_pose(request.object_observation)

    def is_plausible_track(self, request: PoseRequest, tracked_pose: Pose) -> bool:
        """
        :param request: The tracked object hypothesis.
        :param tracked_pose: Tracked pose of the mesh in the camera frame.
        :return: Whether the center of the tracked mesh lies in front of the camera and
            projects into the mask region, enlarged by the maximum track center offset.
        """
        mesh_center = request.class_mesh.shape.local_frame_bounding_box.center
        center = (tracked_pose.homogeneous_matrix @ mesh_center).to_np()[:3]
        if center[2] <= 0:
            return False
        camera_model = request.object_observation.frame.camera_model
        x, y, _ = camera_model.intrinsic_matrix @ (center / center[2])
        region = request.object_observation.mask_region
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
        Write an estimated pose into the pose annotation of its request, adding one if the
        request has none.

        :param estimated_pose: The estimated pose.
        """
        request = estimated_pose.request
        pose_annotation = request.pose_annotation
        if pose_annotation is None:
            pose_annotation = PoseAnnotation()
            request.object_hypothesis.annotations.append(pose_annotation)
        pose_annotation.source = self.get_class_name()
        pose_annotation.translation = estimated_pose.pose.position.to_np()[:3].tolist()
        pose_annotation.rotation = estimated_pose.pose.orientation.to_np().tolist()

    def visualize(
        self, frame: ColorAndDepthFrame, estimated_poses: list[EstimatedPose]
    ) -> None:
        """
        Show the posed meshes in the color image and, with the point cloud of the CAS, in
        the 3D view.

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
                estimated_pose.request.class_mesh.shape,
                estimated_pose.pose,
                estimated_pose.label,
            )
        if cas.cloud is not None:
            visualization.geometries.append(NamedGeometry(CASViews.CLOUD, cas.cloud))
        output = self.get_annotator_output_struct()
        output.set_image(visualization.color_image)
        output.set_geometries(
            [geometry.to_drawable() for geometry in visualization.geometries]
        )
