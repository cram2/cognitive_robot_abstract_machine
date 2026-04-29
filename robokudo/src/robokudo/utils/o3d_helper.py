"""Open3D helper utilities for Robokudo.

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

import cv2
import numpy as np
import open3d as o3d
import trimesh
from typing_extensions import TYPE_CHECKING, Tuple, List
from robokudo.cas import CASViews
from robokudo.utils.transform import (
    get_transform_matrix,
    get_translation_from_transform_matrix,
    get_rotation_from_transform_matrix,
)

if TYPE_CHECKING:
    import cv2.typing as cv2t
    import numpy.typing as npt
    from robokudo.cas import CAS
    from semantic_digital_twin.world_description.geometry import Mesh


def put_obb_on_target_obb(
    input_obb: o3d.geometry.OrientedBoundingBox,
    target_obb: o3d.geometry.OrientedBoundingBox,
) -> o3d.geometry.OrientedBoundingBox:
    """Place one oriented bounding box on top of another.

    Places input_obb on top of target_obb by aligning their centers and
    adjusting height based on extents. Assumes positive Z is up.

    :param input_obb: Bounding box to place
    :param target_obb: Target bounding box to place on
    :return: Transformed input bounding box
    """
    put_on_translation = np.eye(4)
    # set the z(height) based on the extents of the input_obb
    # OBB origins are center of bounds
    put_on_translation[2, 3] = input_obb.extent[2] / 2 + target_obb.extent[2] / 2

    return transform_obb_relative_to_obb_center(
        input_obb, target_obb, put_on_translation
    )


def transform_obb_relative_to_obb_center(
    input_obb: o3d.geometry.OrientedBoundingBox,
    target_obb: o3d.geometry.OrientedBoundingBox,
    transform_matrix: npt.NDArray,
) -> o3d.geometry.OrientedBoundingBox:
    """Transform a bounding box relative to another's center.

    :param input_obb: Bounding box to transform
    :param target_obb: Reference bounding box
    :param transform_matrix: 4x4 transform matrix relative to target center
    :return: Transformed bounding box
    """
    target_obb_transform = get_transform_matrix(target_obb.R, target_obb.center)
    input_to_target_transform = np.matmul(target_obb_transform, transform_matrix)

    # Build a new OBB from the input_obb by reusing the size
    # of the input OBB and then use the transformation we computed
    transformed_obb = o3d.geometry.OrientedBoundingBox(
        center=input_to_target_transform[:3, 3],
        R=input_to_target_transform[:3, :3],
        extent=input_obb.extent,
    )
    return transformed_obb


def get_obb_from_size_and_transform(
    bb_size: npt.NDArray, transform_matrix: npt.NDArray
) -> o3d.geometry.OrientedBoundingBox:
    """Create an oriented bounding box from size and transform.

    :param bb_size: Numpy array with 3 elements for BoundingBox size in X,Y,Z
    :param transform_matrix: 4x4 numpy array
    :return: Oriented bounding box
    """
    return o3d.geometry.OrientedBoundingBox(
        center=get_translation_from_transform_matrix(transform_matrix),
        R=get_rotation_from_transform_matrix(transform_matrix),
        extent=bb_size,
    )


def get_2d_corner_points_from_3d_bb(
    cas: CAS, object_bb: o3d.geometry.OrientedBoundingBox
) -> cv2t.Rect:
    """Project 3D bounding box corners to 2D image points.

    :param cas: CAS containing camera parameters
    :param object_bb: 3D bounding box to project
    :return: Array of 2D corner points
    """
    # Create the 2D Boundingbox/ROI based on the OBB
    pc_cam_intrinsics = cas.get(CASViews.PC_CAM_INTRINSIC)
    assert isinstance(pc_cam_intrinsics, o3d.camera.PinholeCameraIntrinsic)
    k = pc_cam_intrinsics.intrinsic_matrix
    corner_points = np.asarray(object_bb.get_box_points())
    uvd = corner_points @ k.T
    if np.any(uvd[:, 2] == 0):
        raise ValueError("Division by zero: Bounding box has zero depth.")
    corner_points_in_2d_x = (uvd[:, 0] / uvd[:, 2]).astype(int)
    corner_points_in_2d_y = (uvd[:, 1] / uvd[:, 2]).astype(int)
    # We have to respect the ratio of the depth and rgb image to properly project back.
    # Invert from RGB-> Depth to Depth->RGB
    color2depth_ratio = cas.get(CASViews.COLOR2DEPTH_RATIO)
    depth2color_ratio = (1 / color2depth_ratio[0], 1 / color2depth_ratio[1])
    x_scaled = (corner_points_in_2d_x * depth2color_ratio[0]).astype(int)
    y_scaled = (corner_points_in_2d_y * depth2color_ratio[1]).astype(int)
    corner_points_in_2d_xy = np.stack((x_scaled, y_scaled), axis=1)
    return corner_points_in_2d_xy


def project_points_to_image(
    cas: CAS, points_cam: npt.NDArray
) -> Tuple[npt.NDArray, npt.NDArray]:
    """Project 3D camera-frame points to 2D image pixels and validity mask."""
    pc_cam_intrinsics = cas.get(CASViews.PC_CAM_INTRINSIC)
    k = pc_cam_intrinsics.intrinsic_matrix
    uvd = points_cam @ k.T
    z = uvd[:, 2]
    valid = z > 1e-6
    u = (uvd[valid, 0] / z[valid]).astype(int)
    v = (uvd[valid, 1] / z[valid]).astype(int)

    color2depth_ratio = cas.get(CASViews.COLOR2DEPTH_RATIO)
    depth2color_ratio = (1 / color2depth_ratio[0], 1 / color2depth_ratio[1])
    u = (u * depth2color_ratio[0]).astype(int)
    v = (v * depth2color_ratio[1]).astype(int)

    return np.stack([u, v], axis=1), valid


def draw_mesh_wireframe_on_image(
    image: npt.NDArray,
    mesh_shape: Mesh,
    object_transform: npt.NDArray,
    cas: CAS,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> None:
    """Draw the mesh wireframe by projecting its edges into the image."""
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
    cas: CAS, object_bb: o3d.geometry.OrientedBoundingBox
) -> cv2t.Rect:
    """
    Get a cv2.boundingRect which represents the space taken by a open3d boundingbox.
    Perspective projection and color2depth ratio are taken into account.

    :param cas:
    :param object_bb:
    :return: cv2.boundingRect
    """
    corner_points_in_2d_xy = get_2d_corner_points_from_3d_bb(cas, object_bb)
    # 0: x, 1: y, 2: w, 3: h
    return cv2.boundingRect(corner_points_in_2d_xy)


def draw_wireframe_of_obb_into_image(
    cas: CAS, image: npt.NDArray, obb: o3d.geometry.OrientedBoundingBox
) -> None:
    """Draw 3D bounding box wireframe on image.

    Projects 3D box edges to image plane and draws lines.

    :param cas: CAS containing camera parameters
    :param image: Image to draw on
    :param obb: 3D bounding box to visualize
    """
    projected_corners = get_2d_corner_points_from_3d_bb(cas, obb)
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
        pt1 = tuple(projected_corners[edge[0]])
        pt2 = tuple(projected_corners[edge[1]])
        cv2.line(image, pt1, pt2, (0, 255, 0), 2)


def get_mask_from_pointcloud(
    input_cloud: o3d.geometry.PointCloud,
    ref_image: npt.NDArray,
    cam_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    mask_scale_factor: float = None,
    crop_to_ref: bool = None,
) -> npt.NDArray:
    """
    Generate a binary mask image by projecting the input_cloud to the ref_image mask

    :param input_cloud: 3d points which should be projected to the result mask image
    :param ref_image: A reference image for the dimensionality of the mask
    :param cam_intrinsics: Camera intrinsics that have been used to generate input_cloud
    :param mask_scale_factor: If your color and depth image size mismatches,
    you might have to scale your depth-based mask image to the dimensions of your color image.
    Please provide that factor here. Example: Kinect has 1280x images were as the depth images are only 640x .
    This would require a scale factor of 2.0
    :param crop_to_ref: This is usually set to True, if mask_scale_factor has been set. Crop the result of the scale
    operation to the size of ref_image (0:height) and (0:width)
    :return: A binary mask image for OpenCV
    """
    K = cam_intrinsics.intrinsic_matrix

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


def scale_o3d_cam_intrinsics(
    cam_intrinsic: o3d.camera.PinholeCameraIntrinsic, scalex: float, scaley: float
) -> o3d.camera.PinholeCameraIntrinsic:
    """Scale camera intrinsics by x and y factors.

    Create and return a new cam intrinsic by scaling an input cam_intrinsic
    based on a scale factor scalex and scaley.
    Scaling will be done by multipling the factors with the relevant properties.
    Example: new_height = height * scaley

    :param cam_intrinsic: Original camera intrinsics
    :param scalex: Scale factor for width/x
    :param scaley: Scale factor for height/y
    :return: Scaled camera intrinsics
    """
    new_cam_intrinsic = o3d.camera.PinholeCameraIntrinsic()

    width = int(cam_intrinsic.width * scalex)
    height = int(cam_intrinsic.height * scaley)
    fx = cam_intrinsic.intrinsic_matrix[0][0] * scalex
    cx = cam_intrinsic.intrinsic_matrix[0][2] * scalex
    fy = cam_intrinsic.intrinsic_matrix[1][1] * scaley
    cy = cam_intrinsic.intrinsic_matrix[1][2] * scaley

    new_cam_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

    return new_cam_intrinsic


def concatenate_clouds(
    clouds: List[o3d.geometry.PointCloud],
) -> o3d.geometry.PointCloud:
    """Combine multiple point clouds into one.

    Concatenates points and colors from input clouds.

    :param clouds: List of point clouds to combine
    :return: Combined point cloud
    """
    visualization_cloud = o3d.geometry.PointCloud()

    all_points = np.concatenate([c.points for c in clouds])
    all_colors = np.concatenate([c.colors for c in clouds])
    visualization_cloud.points = o3d.utility.Vector3dVector(all_points)
    visualization_cloud.colors = o3d.utility.Vector3dVector(all_colors)

    return visualization_cloud


def get_cloud_from_rgb_depth_and_mask(
    rgb_image: npt.NDArray,
    depth_image: npt.NDArray,
    mask: npt.NDArray,
    cam_intrinsics: o3d.camera.PinholeCameraIntrinsic,
    depth_truncate: float = 9.0,
    mask_true_val: int = 255,
) -> o3d.geometry.PointCloud:
    """Create point cloud from RGB-D images and mask.

    Generate a pointcloud from an object mask, rgb and depth image.
    Important: The depth_image will be changed when executing this function. Provide a copy if necessary.

    Constraints:
      - RGB, Depth image and Mask must be of the same resolution.


    :param rgb_image: An image in RGB order. Please note that cv2 normally uses BGR order. So you have to convert first.
    :param depth_image: A depth image
    :param mask: a numpy array of dtype np.uint8
    :param cam_intrinsics: Camera intrinsics for the rgb and depth image
    :param depth_truncate: During cloud creation, ignore points beyond this many meters
    :param mask_true_val: If the 'mask' contains a value equal to 'max_true_val', the corresponding value in 'depth'
    will be included. Otherwise, it will be set to 0.
    :return: Point cloud from masked RGB-D data

    .. warning::
       The depth_image will be modified during processing.
       Provide a copy if the original needs to be preserved.
    """
    # Generate a new depth image by keeping all values of the depth image where the mask is *mask_true_val*.
    # The rest will be set to 0
    depth_image = np.where(mask == mask_true_val, depth_image, 0)
    o3d_color = o3d.geometry.Image(rgb_image)
    o3d_depth = o3d.geometry.Image(depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, convert_rgb_to_intensity=False, depth_trunc=depth_truncate
    )

    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, cam_intrinsics)


def create_line_for_visualization(
    origin: Tuple[float, float, float],
    target: Tuple[float, float, float],
    color: Tuple[float, float, float],
) -> o3d.geometry.LineSet:
    """Create a LineSet that you can use to visualize a line between two points

    :param origin: 3 element list: [x,y,z]
    :param target: 3 element list: [x,y,z]
    :param color: 3 element list: [r,g,b]
    :return: Line segment geometry
    """
    o3d_pointing_points = [origin, target]
    o3d_pointing_lines = [[0, 1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(o3d_pointing_points)
    line_set.lines = o3d.utility.Vector2iVector(o3d_pointing_lines)
    line_set.colors = o3d.utility.Vector3dVector([color])
    return line_set


def create_sphere_from_translation(
    origin: npt.NDArray, color: Tuple[float, float, float], radius: float
) -> o3d.geometry.TriangleMesh:
    """Create an o3d sphere for visualization.

    :param origin: 3d translation vector
    :param color: 3 element list with entries 0-1 for RGB
    :param radius: radius of the sphere in meters
    :return: o3d sphere mesh
    """
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.paint_uniform_color(color)
    sphere.translate(origin)
    return sphere


def trimesh_to_o3d_mesh(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """
    Convert the standard mesh representation of semantic_digital_twin (trimesh) to a type that the
    O3DVisualizer can show.

    :param mesh: The mesh to convert
    :return: An open3d compatible triangle mesh
    """
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))

    if getattr(mesh, "visual", None) is not None and hasattr(
        mesh.visual, "vertex_colors"
    ):
        colors = np.asarray(mesh.visual.vertex_colors)
        if len(colors) == len(mesh.vertices):
            colors = colors[:, :3]
            if colors.max() > 1.0:
                colors = colors / 255.0
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    o3d_mesh.compute_vertex_normals()
    return o3d_mesh
