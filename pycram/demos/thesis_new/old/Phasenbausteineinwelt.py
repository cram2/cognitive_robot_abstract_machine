import os

import rclpy

from demos.thesis.simulation_setup import add_box, BoxSpec
from pycram.robot_plans.actions.composite.thesis_math import WorldTransformFrameProvider
from pycram.robot_plans.actions.composite.thesis_math import (
    build_default_sequence,
    build_container_sequence,
)
from pycram.robot_plans.actions.composite.thesis_math.motion_models import (
    Pose,
    FixedFrameProvider,
)
from demos.thesis_new.src.utils import MotionSequenceRviz
from pycram.robot_plans.actions.composite.thesis_math import (
    try_get_body,
    make_identity_pose_stamped,
    body_local_aabb,
)
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.datastructures.pose import PoseStamped
from pycram.language import SequentialPlan
from pycram.motion_executor import simulated_robot
from pycram.robot_plans import SimpleMoveTCPAction
from pycram.testing import setup_world
from rclpy.duration import Duration as RclpyDuration
from rclpy.time import Time
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bowl
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import FixedConnection


def _setup_world():
    """Create a demo world with a bowl and a box."""
    world = setup_world()

    bowl = STLParser(
        os.path.join(
            os.path.dirname(__file__), "../..", "..", "resources", "objects", "bowl.stl"
        )
    ).parse()
    # for shape in bowl.root.visual.shapes:
    #     shape.scale = Scale(x=1.5, y=1.5, z=1.5)
    # for shape in bowl.root.collision.shapes:
    #     shape.scale = Scale(x=1.5, y=1.5, z=1.5)

    with world.modify_world():
        world.merge_world_at_pose(
            bowl,
            HomogeneousTransformationMatrix.from_xyz_quaternion(
                2.4, 2.2, 1, reference_frame=world.root
            ),
        )

    with world.modify_world():
        box = add_box(
            world, BoxSpec(name="muh_box", scale_xyz=(0.3, 0.1, 0.1)), tf_frame="/map"
        )
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=box,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=-0.9, y=1.0, z=0.95
                ),
            )
        )

    return world


def _print_phase_points(label, points, phase_ids, phases=None, world=None):
    """Print all sampled points with phase ids and optional phase names."""
    print(f"[points] {label}")
    name_map = None
    if phases is not None:
        name_map = {i: getattr(ph, "name", str(i)) for i, ph in enumerate(phases)}
    for i, p in enumerate(points):
        pid = int(phase_ids[i]) if phase_ids is not None else -1
        pname = name_map.get(pid, str(pid)) if name_map is not None else str(pid)
        # print(
        #     f"  phase={pid} name={pname} idx={i} "
        #     f"p=({p[0]:.6f}, {p[1]:.6f}, {p[2]:.6f})"
        # )
    poses = []
    for p in points:
        msg = PoseStamped()
        msg.header.frame_id = world.root
        msg.pose.position.x = p[0]
        msg.pose.position.y = p[1]
        msg.pose.position.z = p[2]
        poses.append(msg)
    return poses


def main():
    """Run the RViz demo for default and bowl-constrained sequences."""
    world = _setup_world()

    rclpy.init()
    node = rclpy.create_node("pycram_demo")

    tf_wrapper = TFWrapper(node=node)
    TFPublisher(node=node, world=world)
    VizMarkerPublisher(world, node)

    tf_wrapper.wait_for_transform(
        "apartment/apartment_root",
        "pr2/base_footprint",
        timeout=RclpyDuration(seconds=1.0),
        time=Time(),
    )

    PR2.from_world(world)
    context = Context.from_world(world)

    with world.modify_world():
        world_reasoner = WorldReasoner(world)
        world_reasoner.reason()
        world.add_semantic_annotations(
            [
                Bowl(root=world.get_body_by_name("bowl.stl")),
            ]
        )

    seq = build_default_sequence()

    prov_world = FixedFrameProvider(Pose())
    _, P_world, id_world = seq.sample(prov_world, dt=0.01)
    # poses = _print_phase_points("world", P_world, id_world, phases=seq.phases)

    rv = MotionSequenceRviz(
        P_world,
        id_world,
        frame_id="apartment/apartment_root",
        topic="phase_sequence_markers",
        node=node,
    )
    rv.publish_once()

    bowl_body = try_get_body(world, "bowl.stl")
    if bowl_body is None:
        print("[info] body 'bowl.stl' not found, skipping object-dependent example.")
        return
    mins, maxs = body_local_aabb(bowl_body, use_visual=False, apply_shape_scale=False)
    print("AABB mins/maxs:", mins, maxs)
    seq_container = build_container_sequence(bowl_body, debug=True)
    prov_container = WorldTransformFrameProvider(
        world=world,
        source_frame=bowl_body,
        root_frame=world.root,
        make_identity_spatial=make_identity_pose_stamped,
    )
    _, P_container, id_container = seq_container.sample(prov_container, dt=0.01)
    poses = _print_phase_points(
        "bowl", P_container, id_container, phases=seq_container.phases, world=world
    )

    print("one pose only" + str(poses[0]))
    plan = SequentialPlan(
        context,
        SimpleMoveTCPAction(target_location=poses[0], arm=Arms.RIGHT),
        # MoveTorsoActionDescription(TorsoState.HIGH)
    )
    with simulated_robot:
        plan.perform()

    rv_container = MotionSequenceRviz(
        P_container,
        id_container,
        frame_id="apartment/apartment_root",
        topic="phase_sequence_markers_bowl",
        node=node,
    )
    rv_container.publish_once()


if __name__ == "__main__":
    main()
