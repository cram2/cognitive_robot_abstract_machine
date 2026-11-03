"""
Drawing of posed CAD models into the color image and the 3D view.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum, auto

import cv2
import numpy
import open3d
from semantic_digital_twin.world_description.geometry import Color
from typing_extensions import TYPE_CHECKING, Any

from robokudo.annotators.foundation_pose.cad_model import CadModel
from robokudo.types.cv import ColorToDepthRatio
from robokudo.utils.transform import get_transform_matrix_from_translation

if TYPE_CHECKING:
    import numpy.typing as numpy_typing
    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel


# %% Visualization settings


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


# %% Pose visualization


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
