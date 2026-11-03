"""
6D pose estimation and tracking of classified object hypotheses whose CAD models are
known.
"""

from __future__ import annotations

import tempfile
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
from py_trees.common import Status
from semantic_digital_twin.world_description.geometry import Color
from typing_extensions import TYPE_CHECKING, Any

from robokudo.annotators.core import BaseAnnotator, ThreadedAnnotator
from robokudo.annotators.exceptions import DuplicateCadModelClass
from robokudo.cas import CASViews
from robokudo.types.annotation import Classification, PoseAnnotation
from robokudo.types.cv import ColorToDepthRatio, Point2D, Rect
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.error_handling import catch_and_raise_to_blackboard
from robokudo.utils.non_maxima_suppression import _iou
from robokudo.utils.transform import (
    get_quaternion_from_transform_matrix,
    get_transform_matrix_from_q,
    get_transform_matrix_from_translation,
    get_translation_from_transform_matrix,
)

if TYPE_CHECKING:
    import numpy.typing as numpy_typing
    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel


# %% Pose estimator interface


@dataclass
class RGBDFrame:
    """
    A color and a depth image of the same resolution, taken by one camera.
    """

    rgb_image: numpy_typing.NDArray[numpy.uint8]
    """
    Color image in RGB channel order, at the resolution of :attr:`depth_image`.
    """

    depth_image: numpy_typing.NDArray[numpy.float32]
    """
    Depth image in meters; pixels without a measurement are zero.
    """

    camera_model: PinholeCameraModel
    """
    Calibration of the camera at the resolution of the images.
    """


@dataclass
class ObjectObservation:
    """
    One object in an RGB-D frame, marked by a mask.
    """

    frame: RGBDFrame
    """
    The frame the object is seen in.
    """

    mask: numpy_typing.NDArray[numpy.bool_]
    """
    Pixels of :attr:`frame` that show the object.
    """

    @cached_property
    def mask_region(self) -> Rect:
        """
        The smallest rectangle containing every pixel of :attr:`mask`.
        """
        left, top, width, height = cv2.boundingRect(self.mask.astype(numpy.uint8))
        return Rect(pos=Point2D(x=left, y=top), width=width, height=height)


@dataclass
class ObjectPoseEstimator(ABC):
    """
    Estimates and tracks the pose of an object in the camera frame, given the mesh of
    the object.

    Poses are 4x4 homogeneous transforms from the frame of the mesh to the camera frame.
    """

    @abstractmethod
    def load_meshes(self, mesh_files: list[Path]) -> None:
        """
        Load meshes so that selecting any of them with :meth:`select_mesh` is immediate.

        :param mesh_files: Mesh files in meters, each holding a single mesh.
        """

    @abstractmethod
    def select_mesh(self, mesh_file: Path) -> None:
        """
        Use a mesh for the following pose estimates and tracks.

        :param mesh_file: Mesh file in meters, holding a single mesh.
        """

    @abstractmethod
    def estimate_pose(
        self, observation: ObjectObservation
    ) -> numpy_typing.NDArray[numpy.float64] | None:
        """
        Estimate the pose of the selected mesh without prior knowledge.

        :param observation: The object to estimate the pose of.
        :return: The pose of the mesh, or ``None`` if the observation holds no depth
            measurement of the object.
        """

    @abstractmethod
    def track_pose(
        self,
        observation: ObjectObservation,
        previous_pose: numpy_typing.NDArray[numpy.float64],
    ) -> numpy_typing.NDArray[numpy.float64]:
        """
        Refine a known pose of the selected mesh to an observation of the object that
        moved a little since.

        :param observation: The object to track.
        :param previous_pose: The pose of the mesh before it moved.
        :return: The refined pose of the mesh.
        """


# %% CAD models


@dataclass
class CadModelSource:
    """
    A mesh file showing the objects of one class.
    """

    class_name: str
    """
    Class name of the :class:`robokudo.types.annotation.Classification` the objects are
    detected with.
    """

    mesh_file: Path
    """
    The mesh file, in any format :func:`trimesh.load` reads.
    """

    scale_to_meters: float = 1.0
    """
    Factor that converts the units of :attr:`mesh_file` to meters.
    """


@dataclass
class CadModel:
    """
    The mesh of an object class, in meters.
    """

    source: CadModelSource
    """
    The mesh file the CAD model is loaded from.
    """

    mesh: trimesh.Trimesh
    """
    The mesh, in meters.
    """

    estimator_mesh_file: Path
    """
    File holding exactly :attr:`mesh`, as one mesh in meters.
    """

    @classmethod
    def from_source(
        cls, source: CadModelSource, rewritten_mesh_directory: Path
    ) -> CadModel:
        """
        Load a mesh file, merging all its meshes into one and converting it to meters.

        :param source: The mesh file to load.
        :param rewritten_mesh_directory: Directory to write the mesh to if it differs from
            the content of :attr:`CadModelSource.mesh_file`.
        :return: The CAD model of the source.
        """
        loaded = trimesh.load(source.mesh_file)
        if isinstance(loaded, trimesh.Trimesh):
            mesh, is_single_mesh = loaded, True
        else:
            mesh, is_single_mesh = loaded.to_mesh(), len(loaded.geometry) == 1
        mesh.apply_scale(source.scale_to_meters)
        estimator_mesh_file = source.mesh_file
        if not is_single_mesh or source.scale_to_meters != 1.0:
            estimator_mesh_file = (
                rewritten_mesh_directory / source.class_name / source.mesh_file.name
            )
            estimator_mesh_file.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(estimator_mesh_file)
        return cls(source=source, mesh=mesh, estimator_mesh_file=estimator_mesh_file)


# %% Tracking across frames


@dataclass(eq=False)
class PosedObject:
    """
    An object whose pose was determined in a frame.

    Posed objects are equal only to themselves.
    """

    class_name: str
    """
    Class name of the object's CAD model.
    """

    pose: numpy_typing.NDArray[numpy.float64]
    """
    Pose of the CAD model in the camera frame, as a 4x4 homogeneous transform.
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
    ) -> numpy_typing.NDArray[numpy.float64] | None:
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
        overlaps = [
            _iou(
                candidate.mask_region.get_corner_points(),
                mask_region.get_corner_points(),
            )
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
    Frame the axes and the bounding box of a posed CAD model are drawn in.
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
class PoseVisualization:
    """
    The color image and the 3D geometries showing posed CAD models.
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

    geometries: list[dict[DrawableGeometryKey, Any]] = field(default_factory=list)
    """
    Named geometries of the 3D view, in the camera frame at depth image resolution.
    """

    def draw(
        self,
        cad_model: CadModel,
        pose: numpy_typing.NDArray[numpy.float64],
        label: str,
    ) -> None:
        """
        Draw a posed CAD model into the color image and add it to the 3D view.

        :param cad_model: The CAD model.
        :param pose: Pose of the mesh in the camera frame.
        :param label: Unique text identifying the object.
        """
        bounding_box = self.bounding_box(cad_model, pose)
        axes_pose = self.axes_pose(cad_model, pose)
        self.draw_bounding_box(bounding_box, label)
        self.draw_axes(axes_pose)

        axes = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.axis_length
        )
        axes.transform(axes_pose)
        self.add_geometry(f"{label} bounding box", bounding_box)
        self.add_geometry(f"{label} axes", axes)
        if self.show_object_mesh:
            self.add_geometry(f"{label} mesh", self.posed_mesh(cad_model, pose))

    def bounding_box(
        self, cad_model: CadModel, pose: numpy_typing.NDArray[numpy.float64]
    ) -> open3d.geometry.OrientedBoundingBox:
        """
        :param cad_model: The CAD model.
        :param pose: Pose of the mesh in the camera frame.
        :return: The bounding box of the posed mesh in :attr:`frame`.
        """
        if self.frame is VisualizationFrame.SMALLEST_BOUNDING_BOX:
            smallest_box = cad_model.mesh.bounding_box_oriented.primitive
            box_pose = pose @ smallest_box.transform
            return open3d.geometry.OrientedBoundingBox(
                box_pose[:3, 3], box_pose[:3, :3], smallest_box.extents
            )
        lower_corner, upper_corner = cad_model.mesh.bounds
        center = pose @ numpy.append((lower_corner + upper_corner) / 2, 1.0)
        return open3d.geometry.OrientedBoundingBox(
            center[:3], pose[:3, :3], upper_corner - lower_corner
        )

    def axes_pose(
        self, cad_model: CadModel, pose: numpy_typing.NDArray[numpy.float64]
    ) -> numpy_typing.NDArray[numpy.float64]:
        """
        :param cad_model: The CAD model.
        :param pose: Pose of the mesh in the camera frame.
        :return: Pose of the drawn axes in the camera frame.
        """
        smallest_box_pose = cad_model.mesh.bounding_box_oriented.primitive.transform
        if self.frame is VisualizationFrame.SMALLEST_BOUNDING_BOX:
            return pose @ smallest_box_pose
        if not self.center_axes_on_bounding_box:
            return pose
        return pose @ get_transform_matrix_from_translation(smallest_box_pose[:3, 3])

    def draw_bounding_box(
        self, bounding_box: open3d.geometry.OrientedBoundingBox, label: str
    ) -> None:
        """
        Draw the edges of a bounding box into the color image, labeled above its top left
        corner.

        :param bounding_box: Bounding box in the camera frame.
        :param label: Text drawn above the bounding box.
        """
        edges = open3d.geometry.LineSet.create_from_oriented_bounding_box(bounding_box)
        pixels = self.color_pixels(numpy.asarray(edges.points))
        for start, end in numpy.asarray(edges.lines):
            self.draw_line(pixels[start], pixels[end], Color.GREEN())
        left, top = pixels.min(axis=0)
        cv2.putText(
            self.color_image,
            label,
            (int(left), int(top)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.label_font_scale,
            self.bgr(Color.WHITE()),
            lineType=cv2.LINE_AA,
        )

    def draw_axes(self, axes_pose: numpy_typing.NDArray[numpy.float64]) -> None:
        """
        Draw the x, y and z axes of a pose in red, green and blue into the color image.

        :param axes_pose: Pose of the axes in the camera frame.
        """
        axis_ends = numpy.vstack([numpy.zeros(3), numpy.eye(3) * self.axis_length])
        homogeneous_ends = numpy.hstack([axis_ends, numpy.ones((len(axis_ends), 1))])
        pixels = self.color_pixels((homogeneous_ends @ axes_pose.T)[:, :3])
        origin = pixels[0]
        for axis_end, color in zip(
            pixels[1:], (Color.RED(), Color.GREEN(), Color.BLUE())
        ):
            self.draw_line(origin, axis_end, color)

    def draw_line(
        self,
        start: numpy_typing.NDArray[numpy.int_],
        end: numpy_typing.NDArray[numpy.int_],
        color: Color,
    ) -> None:
        """
        Draw a line into the color image.

        :param start: Color image pixel the line starts at.
        :param end: Color image pixel the line ends at.
        :param color: Color of the line.
        """
        cv2.line(
            self.color_image,
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1])),
            self.bgr(color),
            self.line_thickness,
            cv2.LINE_AA,
        )

    def color_pixels(
        self, points: numpy_typing.NDArray[numpy.float64]
    ) -> numpy_typing.NDArray[numpy.int_]:
        """
        Project points in the camera frame into the color image.

        :param points: Points in the camera frame, one per row.
        :return: The (x, y) color image pixel of each point, one per row.
        """
        projected = points @ self.camera_model.intrinsic_matrix.T
        depth_pixels = projected[:, :2] / projected[:, 2:]
        return self.color_to_depth_ratio.color_pixels(depth_pixels)

    def posed_mesh(
        self, cad_model: CadModel, pose: numpy_typing.NDArray[numpy.float64]
    ) -> open3d.geometry.TriangleMesh:
        """
        :param cad_model: The CAD model.
        :param pose: Pose of the mesh in the camera frame.
        :return: The mesh in the camera frame, painted in :attr:`object_mesh_color`.
        """
        mesh = open3d.geometry.TriangleMesh(
            open3d.utility.Vector3dVector(cad_model.mesh.vertices),
            open3d.utility.Vector3iVector(cad_model.mesh.faces),
        )
        mesh.paint_uniform_color(self.object_mesh_color.to_rgb())
        mesh.transform(pose)
        mesh.compute_vertex_normals()
        return mesh

    def add_geometry(self, name: str, geometry: open3d.geometry.Geometry3D) -> None:
        """
        Add a named geometry to the 3D view.

        :param name: Unique name of the geometry.
        :param geometry: Geometry in the camera frame.
        """
        self.geometries.append(
            {DrawableGeometryKey.NAME: name, DrawableGeometryKey.GEOMETRY: geometry}
        )

    @staticmethod
    def bgr(color: Color) -> tuple[int, int, int]:
        """
        :param color: A color.
        :return: The color as OpenCV byte channels in BGR order.
        """
        maximum = numpy.iinfo(numpy.uint8).max
        red, green, blue = color.to_rgb()
        return round(blue * maximum), round(green * maximum), round(red * maximum)


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
        return f"{self.request.object_hypothesis.id}: {self.request.cad_model.source.class_name}"


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
                class_name=estimated_pose.request.cad_model.source.class_name,
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
                class_name=cad_model.source.class_name,
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
                f"Tracked pose of {request.cad_model.source.class_name} left its mask, "
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
