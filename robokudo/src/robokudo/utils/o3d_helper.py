"""
Open3D helper utilities for Robokudo.

This module provides helper functions for working with Open3D geometries and cameras.
It supports:

* Oriented bounding box operations
* Point cloud manipulation
* Camera intrinsics handling
* 2D-3D projections
* Visualization utilities

The module integrates with:
* Open3D for 3D geometry operations
* OpenCV for image operations
* NumPy for numerical computations
* ROS for camera info conversion
"""

from __future__ import annotations

from robokudo.exceptions import BoundingBoxDepthMissing

# %% Imports
import cv2
import numpy as np
import open3d
import trimesh
from typing_extensions import TYPE_CHECKING
from robokudo.cas import CASViews
from robokudo.io.camera_model_adapters import Open3DCameraModelAdapter
from robokudo.utils.transform import (
    get_transform_matrix,
    get_translation_from_transform_matrix,
    get_rotation_from_transform_matrix,
)

if TYPE_CHECKING:
    import cv2.typing as cv2t
    import numpy.typing as npt
    from robokudo.cas import CAS
    from semantic_digital_twin.datastructures.camera_model import PinholeCameraModel
    from semantic_digital_twin.world_description.geometry import Mesh


def place_bounding_box_on_target(
    input_bounding_box: open3d.geometry.OrientedBoundingBox,
    target_bounding_box: open3d.geometry.OrientedBoundingBox,
) -> open3d.geometry.OrientedBoundingBox:
    """
    Place one oriented bounding box on top of another.

    The boxes share an orientation, and their centers are aligned along the target's
    local vertical axis. Their faces touch along that axis.

    :param input_bounding_box: Bounding box to place
    :param target_bounding_box: Target bounding box to place on
    :return: Transformed input bounding box
    """
    put_on_translation = np.eye(4)
    put_on_translation[2, 3] = (
        input_bounding_box.extent[2] / 2 + target_bounding_box.extent[2] / 2
    )

    return position_bounding_box_relative_to_target(
        input_bounding_box, target_bounding_box, put_on_translation
    )


def position_bounding_box_relative_to_target(
    input_bounding_box: open3d.geometry.OrientedBoundingBox,
    target_bounding_box: open3d.geometry.OrientedBoundingBox,
    transform_matrix: npt.NDArray,
) -> open3d.geometry.OrientedBoundingBox:
    """
    Position a bounding box relative to the target's center and orientation.

    Preserve the input box's extent and replace its pose with the
    supplied relative transform composed with the target's pose.

    :param input_bounding_box: Bounding box whose extent is preserved
    :param target_bounding_box: Reference bounding box
    :param transform_matrix: 4x4 transform matrix relative to target
        center
    :return: Transformed bounding box
    """
    target_transform = get_transform_matrix(
        target_bounding_box.R, target_bounding_box.center
    )
    positioned_transform = np.matmul(target_transform, transform_matrix)

    positioned_bounding_box = open3d.geometry.OrientedBoundingBox(
        center=positioned_transform[:3, 3],
        R=positioned_transform[:3, :3],
        extent=input_bounding_box.extent,
    )
    return positioned_bounding_box


def create_oriented_bounding_box(
    bounding_box_extent: npt.NDArray, transform_matrix: npt.NDArray
) -> open3d.geometry.OrientedBoundingBox:
    """
    Create an oriented bounding box from size and transform.

    :param bounding_box_extent: Side lengths along the box's local axes
    :param transform_matrix: 4x4 numpy array
    :return: Oriented bounding box
    """
    return open3d.geometry.OrientedBoundingBox(
        center=get_translation_from_transform_matrix(transform_matrix),
        R=get_rotation_from_transform_matrix(transform_matrix),
        extent=bounding_box_extent,
    )


def get_2d_corner_points_from_3d_bb(
    cas: CAS, object_bb: open3d.geometry.OrientedBoundingBox
) -> cv2t.Rect:
    """
    Project 3D bounding box corners to 2D image points.

    :param cas: CAS containing camera parameters
    :param object_bb: 3D bounding box to project
    :return: Array of 2D corner points
    """
    # Create the 2D Boundingbox/ROI based on the OBB
    pointcloud_camera_model = cas.pointcloud_camera_model
    k = pointcloud_camera_model.intrinsic_matrix
    corner_points = np.asarray(object_bb.get_box_points())
    uvd = corner_points @ k.T
    if np.any(uvd[:, 2] == 0):
        raise BoundingBoxDepthMissing()
    corner_points_in_2d_x = (uvd[:, 0] / uvd[:, 2]).astype(int)
    corner_points_in_2d_y = (uvd[:, 1] / uvd[:, 2]).astype(int)
    return cas.color_to_depth_ratio.color_pixels(
        np.stack((corner_points_in_2d_x, corner_points_in_2d_y), axis=1)
    )


def project_points_to_image(
    cas: CAS, points_camera: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Project 3D camera-frame points to 2D image pixels and validity mask.

    :param cas: Point cloud calibration and color-to-depth image scale.
    :param points_camera: Point coordinates in the point cloud camera frame, one per
        row.
    :return: Color image pixel coordinates for valid points and a validity mask over all
        input points.
    """
    pointcloud_camera_model = cas.pointcloud_camera_model
    k = pointcloud_camera_model.intrinsic_matrix
    uvd = points_camera @ k.T
    z = uvd[:, 2]
    valid = z > 1e-6
    u = (uvd[valid, 0] / z[valid]).astype(int)
    v = (uvd[valid, 1] / z[valid]).astype(int)

    color_to_depth_ratio = cas.color_to_depth_ratio
    return color_to_depth_ratio.color_pixels(np.stack([u, v], axis=1)), valid


def draw_mesh_wireframe_on_image(
    image: npt.NDArray,
    mesh_shape: Mesh,
    object_transform: npt.NDArray,
    cas: CAS,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> None:
    """
    Draw the mesh wireframe by projecting its edges into the image.
    """
    tm = mesh_shape.mesh.copy()
    tm.apply_transform(mesh_shape.origin.to_np())
    tm.apply_transform(object_transform)

    vertices = np.asarray(tm.vertices)
    points_2d, valid = project_points_to_image(cas, vertices)

    index_map = -np.ones(len(vertices), dtype=int)
    valid_idx = np.where(valid)[0]
    index_map[valid_idx] = np.arange(len(valid_idx))

    for edge in tm.edges_unique:
        a, b = edge
        ia = index_map[a]
        ib = index_map[b]
        if ia < 0 or ib < 0:
            continue
        pt1 = tuple(points_2d[ia])
        pt2 = tuple(points_2d[ib])
        cv2.line(image, pt1, pt2, color, thickness)


def get_2d_bounding_rect_from_3d_bb(
    cas: CAS, object_bb: open3d.geometry.OrientedBoundingBox
) -> cv2t.Rect:
    """
    Get a cv2.boundingRect which represents the space taken by a open3d boundingbox.
    Perspective projection and color-to-depth ratio are taken into account.

    :param cas:
    :param object_bb:
    :return: cv2.boundingRect
    """
    corner_points_in_2d_xy = get_2d_corner_points_from_3d_bb(cas, object_bb)
    # 0: x, 1: y, 2: w, 3: h
    return cv2.boundingRect(corner_points_in_2d_xy)


def draw_bounding_box_wireframe(
    cas: CAS, image: npt.NDArray, bounding_box: open3d.geometry.OrientedBoundingBox
) -> None:
    """
    Draw a projected bounding box wireframe into an image.

    :param cas: CAS containing camera parameters
    :param image: Image to draw on
    :param bounding_box: 3D bounding box to visualize
    """
    projected_corners = get_2d_corner_points_from_3d_bb(cas, bounding_box)
    edges = [
        (0, 1),
        (1, 7),
        (7, 2),
        (2, 0),  # Bottom face edges
        (3, 6),
        (6, 4),
        (4, 5),
        (5, 3),  # Top face edges
        (0, 3),
        (1, 6),
        (2, 5),
        (7, 4),  # Vertical edges
    ]

    for edge in edges:
        start_point = tuple(projected_corners[edge[0]])
        end_point = tuple(projected_corners[edge[1]])
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)


def get_mask_from_pointcloud(
    input_cloud: open3d.geometry.PointCloud,
    ref_image: npt.NDArray,
    camera_intrinsics: PinholeCameraModel,
    mask_scale_factor: float = None,
    crop_to_ref: bool = None,
) -> npt.NDArray:
    """
    Generate a binary mask image by projecting the input_cloud to the ref_image mask.

    :param input_cloud: 3d points which should be projected to the result mask image
    :param ref_image: A reference image for the dimensionality of the mask
    :param camera_intrinsics: Camera intrinsics that have been used to generate
        input_cloud
    :param mask_scale_factor: If your color and depth image size mismatches, you might
        have to scale your depth-based mask image to the dimensions of your color image.
        Please provide that factor here. Example: Kinect has 1280x images were as the
        depth images are only 640x . This would require a scale factor of 2.0
    :param crop_to_ref: This is usually set to True, if mask_scale_factor has been set.
        Crop the result of the scale operation to the size of ref_image (0:height) and
        (0:width)
    :return: A binary mask image for OpenCV
    """
    K = camera_intrinsics.intrinsic_matrix

    cropped_3d_points = np.asarray(input_cloud.points)

    uvd = cropped_3d_points @ K.T

    cropped_x = (uvd[:, 0] / uvd[:, 2]).astype(int)
    cropped_y = (uvd[:, 1] / uvd[:, 2]).astype(int)

    height, width, d = ref_image.shape

    cropped_x = np.clip(cropped_x, 0, width - 1)
    cropped_y = np.clip(cropped_y, 0, height - 1)

    mask = np.zeros((height, width, d), np.uint8)
    mask[cropped_y, cropped_x] = (255, 255, 255)

    if mask_scale_factor:
        mask = cv2.resize(
            mask,
            None,
            fx=mask_scale_factor,
            fy=mask_scale_factor,
            interpolation=cv2.INTER_NEAREST,
        )

    if crop_to_ref:
        mask = mask[0:height, 0:width]

    return mask


def concatenate_clouds(
    clouds: list[open3d.geometry.PointCloud],
) -> open3d.geometry.PointCloud:
    """
    Combine multiple point clouds into one.

    Concatenates points and colors from input clouds.

    :param clouds: List of point clouds to combine
    :return: Combined point cloud
    """
    visualization_cloud = open3d.geometry.PointCloud()

    all_points = np.concatenate([c.points for c in clouds])
    all_colors = np.concatenate([c.colors for c in clouds])
    visualization_cloud.points = open3d.utility.Vector3dVector(all_points)
    visualization_cloud.colors = open3d.utility.Vector3dVector(all_colors)

    return visualization_cloud


def get_cloud_from_rgb_depth_and_mask(
    rgb_image: npt.NDArray,
    depth_image: npt.NDArray,
    mask: npt.NDArray,
    camera_model: PinholeCameraModel,
    depth_truncate: float = 9.0,
    mask_true_value: int = 255,
) -> open3d.geometry.PointCloud:
    """
    Build the colored point cloud of the depth measurements a mask marks.

    The color image, the depth image and the mask must have the same resolution.

    :param rgb_image: Color image in RGB order.
    :param depth_image: Depth image in millimeters.
    :param mask: Mask on the depth image.
    :param camera_model: Projection model at the depth image resolution.
    :param depth_truncate: Distance in meters beyond which measurements are left out.
    :param mask_true_value: Value of the mask pixels whose depth measurements the cloud
        is built from.
    :return: The colored point cloud of the masked depth measurements.
    """
    depth_image = np.where(mask == mask_true_value, depth_image, 0)
    open3d_color_image = open3d.geometry.Image(rgb_image)
    open3d_depth_image = open3d.geometry.Image(depth_image)
    rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(
        open3d_color_image,
        open3d_depth_image,
        convert_rgb_to_intensity=False,
        depth_trunc=depth_truncate,
    )

    return open3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, Open3DCameraModelAdapter.to_intrinsic(camera_model)
    )


def create_line_for_visualization(
    origin: tuple[float, float, float],
    target: tuple[float, float, float],
    color: tuple[float, float, float],
) -> open3d.geometry.LineSet:
    """
    Create a LineSet that you can use to visualize a line between two points.

    :param origin: 3 element list: [x,y,z]
    :param target: 3 element list: [x,y,z]
    :param color: 3 element list: [r,g,b]
    :return: Line segment geometry
    """
    line_endpoints = [origin, target]
    line_indices = [[0, 1]]
    line_set = open3d.geometry.LineSet()
    line_set.points = open3d.utility.Vector3dVector(line_endpoints)
    line_set.lines = open3d.utility.Vector2iVector(line_indices)
    line_set.colors = open3d.utility.Vector3dVector([color])
    return line_set


def create_sphere_from_translation(
    origin: npt.NDArray, color: tuple[float, float, float], radius: float
) -> open3d.geometry.TriangleMesh:
    """
    Create an open3d sphere for visualization.

    :param origin: 3d translation vector
    :param color: 3 element list with entries 0-1 for RGB
    :param radius: radius of the sphere in meters
    :return: open3d sphere mesh
    """
    sphere = open3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(origin)
    return sphere


def trimesh_to_o3d_mesh(mesh: trimesh.Trimesh) -> open3d.geometry.TriangleMesh:
    """
    Convert the standard mesh representation of semantic_digital_twin (trimesh) to a
    type that the O3DVisualizer can show.

    :param mesh: The mesh to convert
    :return: An open3d compatible triangle mesh
    """
    triangle_mesh = open3d.geometry.TriangleMesh()
    triangle_mesh.vertices = open3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    triangle_mesh.triangles = open3d.utility.Vector3iVector(np.asarray(mesh.faces))

    if getattr(mesh, "visual", None) is not None and hasattr(
        mesh.visual, "vertex_colors"
    ):
        colors = np.asarray(mesh.visual.vertex_colors)
        if len(colors) == len(mesh.vertices):
            colors = colors[:, :3]
            if colors.max() > 1.0:
                colors = colors / 255.0
            triangle_mesh.vertex_colors = open3d.utility.Vector3dVector(colors)

    triangle_mesh.compute_vertex_normals()
    return triangle_mesh
