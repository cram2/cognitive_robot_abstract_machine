import threading
import time

import rclpy
import tqdm
from rclpy.executors import SingleThreadedExecutor

from pycram.motion_executor import simulated_robot
from pycram.sage_10k.demos import *
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.orm.ormatic_interface import *  # type: ignore
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection

demo = Sage10kCraftsmanLobbyDemo()


def run_demo(demo: Sage10kAbstractDemo):
    demo.create_world()
    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("test_node")

    with demo.world.modify_world():
        camera_frame = Body(name=PrefixedName("camera_link"))
        robot = demo.world.get_semantic_annotations_by_type(HSRB)[0]
        camera_to_robot_connection = FixedConnection(
            parent=robot.root,
            child=camera_frame,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=-0.28,
                y=0.33,
                z=0,
                yaw=np.pi,
                pitch=np.pi / 4,
                roll=0,
            ),
        )
        demo.world.add_connection(camera_to_robot_connection)
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
    thread.start()
    time.sleep(0.1)

    viz_marker_publisher = VizMarkerPublisher(_world=demo.world, node=node)
    viz_marker_publisher.with_tf_publisher()

    with simulated_robot:
        demo.plan.perform()

    viz_marker_publisher.stop()
    del demo


demos = [
    Sage10kGymDemo,
    Sage10kTVStudioDemo,
    Sage10kCraftsmanLobbyDemo,
    Sage10kTropicalWarehouse,
    Sage10kVaporwave,
    Sage10kEclecticResidence,
    Sage10kSouthwesternStoreDemo,
    Sage10kBrutalistStoreDemo,
    Sage10kAmericanBuffetDemo,
]

# pbar = tqdm.tqdm(recursive_subclasses(Sage10kAbstractDemo))
pbar = tqdm.tqdm([Sage10kEclecticResidence], mininterval=1)
for demo in pbar:
    pbar.set_postfix({"Current Scene": demo.scene_url.name})
    run_demo(demo())
