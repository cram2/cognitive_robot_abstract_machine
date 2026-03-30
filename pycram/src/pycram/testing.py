import logging
import os
import unittest

import numpy as np

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
)
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import OmniDrive


logger = logging.getLogger(__name__)

try:
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )
except ImportError:
    logger.info(
        "Could not import VizMarkerPublisher. This is probably because you are not running ROS."
    )


def setup_world() -> World:
    logger.setLevel(logging.DEBUG)

    pr2_sem_world = URDFParser.from_file(
        "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    ).parse()
    apartment_world = URDFParser.from_file(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "worlds",
            "apartment.urdf",
        )
    ).parse()
    milk_world = STLParser(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", "objects", "milk.stl"
        )
    ).parse()
    cereal_world = STLParser(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "resources",
            "objects",
            "breakfast_cereal.stl",
        )
    ).parse()
    # apartment_world.merge_world(pr2_sem_world)
    apartment_world.merge_world(milk_world)
    apartment_world.merge_world(cereal_world)

    with apartment_world.modify_world():
        pr2_root = pr2_sem_world.get_body_by_name("base_footprint")
        apartment_root = apartment_world.root
        c_root_bf = OmniDrive.create_with_dofs(
            parent=apartment_root, child=pr2_root, world=apartment_world
        )
        apartment_world.merge_world(pr2_sem_world, c_root_bf)
        c_root_bf.origin = HomogeneousTransformationMatrix.from_xyz_rpy(1.5, 2.5, 0)

    apartment_world.get_body_by_name("milk.stl").parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(
            2.37, 2, 1.05, reference_frame=apartment_world.root
        )
    )
    apartment_world.get_body_by_name(
        "breakfast_cereal.stl"
    ).parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
        2.37, 1.8, 1.05, reference_frame=apartment_world.root
    )
    milk_view = Milk(root=apartment_world.get_body_by_name("milk.stl"))
    with apartment_world.modify_world():
        apartment_world.add_semantic_annotation(milk_view)

    return apartment_world


def _make_sine_scan_poses(
    anchor: Pose,
    lanes: int = 6,
    lane_spacing: float = 0.03,
    y_span: float = 0.18,
    amplitude: float = 0.005,
    wiggles: float = 1.0,
    points_per_lane: int = 16,
    lane_axis: str = "z",
) -> list[Pose]:
    """
    Generates a set of 3D poses following a sine wave pattern for scanning purposes.

    This function creates a series of 3D positions and orientations based on a
    sine wave pattern. The generated poses are arranged into multiple parallel
    lanes, with configurable parameters such as lane spacing, sine wave amplitude,
    number of waves (wiggles), and number of points per lane. The lanes can be
    aligned along either the x-axis or z-axis, and the scanning pattern is centered
    around the anchor pose provided.

    :param anchor:
        The starting pose that defines the reference frame, initial position,
        and orientation for the generated poses.

    :param lanes:
        The number of parallel lanes to generate in the scanning pattern. Default
        is 6.

    :param lane_spacing:
        The spacing between adjacent lanes. Default is 0.03.

    :param y_span:
        The range of y-coordinates over which the sine wave pattern is distributed
        in each lane. Default is 0.18.

    :param amplitude:
        The amplitude of the sine wave defining the deviation from the lane center.
        Default is 0.005.

    :param wiggles:
        The number of complete sine wave oscillations in each lane. Default is 1.0.

    :param points_per_lane:
        The number of discrete poses generated per lane. Default is 16.

    :param lane_axis:
        The axis along which the lanes are arranged. Accepted values are "x"
        or "z". Default is "z".

    :return:
        A list of PoseStamped objects, where each object defines a 3D pose in the
        generated scanning pattern.
    """
    x0 = anchor.x
    y0 = anchor.y
    z0 = anchor.z
    q = anchor.to_quaternion()

    y_min = y0 - 0.5 * y_span
    y_max = y0 + 0.5 * y_span
    poses: list[Pose] = []

    if lane_axis not in ("x", "z"):
        raise ValueError(f"lane_axis must be 'x' or 'z', got: {lane_axis}")

    for i in range(lanes):
        yc = np.linspace(y_min, y_max, points_per_lane).flatten()
        if i % 2 == 1:
            yc = yc[::-1]

        phase = 2.0 * np.pi * wiggles * (yc - y_min) / max(y_span, 1e-9)
        wiggle = amplitude * np.sin(phase)
        if lane_axis == "x":
            lane_center = x0 + i * lane_spacing
            xc = lane_center + wiggle
            zc = np.full_like(yc, z0, dtype=float)
        else:
            lane_center = z0 + i * lane_spacing
            zc = lane_center + wiggle
            xc = np.full_like(yc, x0, dtype=float)

        for x, y, z in zip(xc, yc, zc):
            poses.append(
                Pose.from_xyz_quaternion(
                    float(x),
                    float(y),
                    float(z),
                    q.x,
                    q.y,
                    q.z,
                    q.w,
                    reference_frame=anchor.reference_frame,
                )
            )
    return poses
