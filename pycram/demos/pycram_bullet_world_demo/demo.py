import os
import threading
import time
import numpy as np

from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.locations.costmaps import VisibilityCostmap
from pycram.robot_plans.actions.composite.utils.rviz import CameraVisiblePointsRviz
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.transporting import TransportAction
from pycram.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from pycram.tf_transformations import quaternion_multiply
from pycram.utils import get_quaternion_between_two_vectors

from pycram.testing import setup_world
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bowl,
    Spoon,
    Drawer,
    Handle,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Quaternion
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection

world = setup_world()

spoon = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "spoon.stl"
    )
).parse()
bowl = STLParser(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "resources", "objects", "bowl.stl"
    )
).parse()

with world.modify_world():
    world.merge_world_at_pose(
        bowl,
        HomogeneousTransformationMatrix.from_xyz_quaternion(
            2.4, 2.2, 1, reference_frame=world.root
        ),
    )
    connection = FixedConnection(
        parent=world.get_body_by_name("cabinet10_drawer_top"), child=spoon.root
    )
    world.merge_world(spoon, connection)


node = None
executor = None
executor_thread = None
try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
        VizMarkerPublisher,
    )

    rclpy.init()
    node = rclpy.create_node("pycram_bullet_world_demo")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    executor_thread = threading.Thread(
        target=executor.spin, daemon=True, name="pycram-demo-rclpy-executor"
    )
    executor_thread.start()

    TFPublisher(_world=world, node=node)
    VizMarkerPublisher(_world=world, node=node)
except ImportError:
    pass

pr2 = PR2.from_world(world)
context = Context.from_world(world)

with world.modify_world():
    world_reasoner = WorldReasoner(world)
    world_reasoner.reason()
    world.add_semantic_annotations(
        [
            Bowl(root=world.get_body_by_name("bowl.stl")),
            Spoon(root=world.get_body_by_name("spoon.stl")),
        ]
    )
    world.add_semantic_annotation(
        Drawer(
            root=world.get_body_by_name("cabinet10_drawer_top"),
            handle=Handle(root=world.get_body_by_name("handle_cab10_t")),
        )
    )


def look_at(location: Pose, robot_world: World):
    vis = VisibilityCostmap(
        min_height=1,
        max_height=1.3,
        origin=location,
        resolution=0.02,
        width=200,
        height=200,
        world=robot_world,
    )
    return vis


visualize = look_at(
    location=Pose(
        Point3(2.3, 8, 1.25),
        orientation=(Quaternion(z=-0.9995140, w=0.03117147)),
    ),
    robot_world=world,
)

print(visualize.map)

if node is not None:
    camera = pr2.get_default_camera()
    camera_pose = camera.root.global_pose
    forward_axis = np.asarray(
        camera.forward_facing_axis.to_list(), dtype=float
    ).reshape(-1)[:3]
    alignment_quaternion = get_quaternion_between_two_vectors(
        np.array([1.0, 0.0, 0.0], dtype=float),
        forward_axis,
    )
    camera_quaternion = np.asarray(camera_pose.to_quaternion().to_list(), dtype=float)
    aligned_quaternion = quaternion_multiply(camera_quaternion, alignment_quaternion)
    camera_position = np.asarray(camera_pose.to_position().to_list(), dtype=float)[:3]
    raytracer_camera_pose = Pose.from_xyz_quaternion(
        *camera_position,
        *aligned_quaternion,
        reference_frame=camera_pose.reference_frame,
    )

    ray_tracer = RayTracer(world)
    segmentation = ray_tracer.create_segmentation_mask(
        raytracer_camera_pose,
        resolution=128,
    )
    visible_body_ids = {int(idx) for idx in segmentation[segmentation >= 0].flatten()}
    robot_body_ids = {body.index for body in pr2.bodies}
    visible_bodies = [
        ray_tracer.index_to_body[idx]
        for idx in sorted(visible_body_ids)
        if idx not in robot_body_ids and idx in ray_tracer.index_to_body
    ]
    print("Visible bodies from default camera:")
    for body in visible_bodies:
        print(f" - {body.name}")

    CameraVisiblePointsRviz(
        world=world,
        camera_pose=raytracer_camera_pose,
        node=node,
        frame_id=str(world.root.name),
        topic="/debug/camera/visible_points",
        resolution=128,
        point_scale=0.01,
        show_rays=True,
        ray_stride=12,
        ray_alpha=0.15,
        origin_scale=0.04,
        republish_hz=2.0,
    )

try:
    while node is not None and rclpy.ok():
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
finally:
    if executor is not None:
        executor.shutdown()
    if node is not None:
        node.destroy_node()
    if executor_thread is not None:
        executor_thread.join(timeout=2.0)
    if "rclpy" in globals() and rclpy.ok():
        rclpy.shutdown()
