#!/usr/bin/env python3
"""
PR2 Pick-and-Place Demonstration ================================

Demonstrates full pick-and-place manipulation on the PR2 using Giskard
whole-body motion planning and CoraPlex actions.

Manipulation Pipeline:
  1. Park both arms and raise torso to HIGH position.
  2. Open right gripper.
  3. Approach target object (reach grasp pose with right arm).
  4. Close gripper to grasp object.
  5. Lift object vertically above table.
  6. Lower object to target place location.
  7. Open gripper to release object.
  8. Return both arms to parked position.

Execution Modes:
  --mode sim  : Executes against local simulated digital twin (default).
  --mode real : Delegates execution to Giskard action server for hardware/bridge.
  --rviz      : Publishes TF and visualization markers for RViz2.
"""

from __future__ import annotations

import argparse
import logging
import threading
from typing import Optional, Tuple

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from coraplex.datastructures.dataclasses import Context
from coraplex.datastructures.enums import ExecutionType
from coraplex.execution_environment import (
    ExecutionEnvironment,
    real_robot,
    simulated_robot,
)
from coraplex.plans.factories import execute_single
from coraplex.robot_plans.actions.core.robot_body import (
    MoveManipulatorAction,
    MoveTorsoAction,
    ParkArmsAction,
)
from coraplex.robot_plans.motions.gripper import MoveGripperMotion
from coraplex.testing import setup_world
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.definitions import GripperState, TorsoState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.semantic_annotations.semantic_annotations import Cereal
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

import coraplex.alternative_motion_mappings.pr2_motion_mapping  # noqa: F401 -- registers PR2 handlers

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pr2_giskard_pick_place_demo")


def setup_demo_scene() -> Tuple[World, PR2, Body, Cereal, Context]:
    """
    Set up the apartment world with the PR2 and annotated cereal box.
    """
    logger.info("Setting up apartment world and PR2 model...")
    world = setup_world()
    pr2 = PR2.from_world(world)

    cereal = world.get_body_by_name("breakfast_cereal.stl")
    cpos = cereal.global_pose.position

    with world.modify_world():
        cereal_ann = Cereal(root=cereal)
        world.add_semantic_annotation(cereal_ann)
        # Position PR2 base within arm reach of the table and cereal
        pr2.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                float(cpos.x) - 0.65, float(cpos.y), 0.0, reference_frame=world.root
            )
        )

    context = Context.from_world(world)
    logger.info(
        "Scene initialized. Cereal at [%.3f, %.3f, %.3f]",
        float(cpos.x),
        float(cpos.y),
        float(cpos.z),
    )
    return world, pr2, cereal, cereal_ann, context


def run_pick_and_place(
    world: World,
    pr2: PR2,
    cereal: Body,
    cereal_ann: Cereal,
    context: Context,
    execution_env: ExecutionEnvironment,
) -> None:
    """
    Execute the 8-step pick and place sequence.
    """
    grasps = cereal_ann.grasp_candidates()
    if not grasps:
        raise RuntimeError("No valid grasp candidates generated for cereal box.")

    grasp = grasps[0]
    grasp_pose = grasp.world_T_grasp
    cpos = cereal.global_pose.position

    lift_pose = Pose(
        position=Point3(float(cpos.x), float(cpos.y), float(cpos.z) + 0.15),
        orientation=grasp_pose.orientation,
        reference_frame=world.root,
    )

    place_pose = Pose(
        position=Point3(float(cpos.x), float(cpos.y) + 0.20, float(cpos.z) + 0.02),
        orientation=grasp_pose.orientation,
        reference_frame=world.root,
    )

    r_eef = pr2.right_arm.end_effector

    logger.info(
        "Starting pick-and-place sequence (mode: %s)...",
        execution_env.execution_type.name,
    )

    with execution_env:
        # Step 1: Park arms & raise torso
        logger.info("Step 1/8: Parking arms and raising torso...")
        execute_single(
            ParkArmsAction(arms=list(pr2.all_arms)), context=context
        ).plan.perform()
        execute_single(
            MoveTorsoAction(torso_state=TorsoState.HIGH), context=context
        ).plan.perform()

        # Step 2: Open gripper
        logger.info("Step 2/8: Opening right gripper...")
        execute_single(
            MoveGripperMotion(motion=GripperState.OPEN, gripper=r_eef),
            context=context,
        ).plan.perform()

        # Step 3: Reach grasp pose
        logger.info("Step 3/8: Approaching grasp pose...")
        execute_single(
            MoveManipulatorAction(
                target_pose=grasp_pose,
                end_effector=r_eef,
                allow_gripper_collision=True,
            ),
            context=context,
        ).plan.perform()

        # Step 4: Close gripper
        logger.info("Step 4/8: Closing gripper to grasp object...")
        execute_single(
            MoveGripperMotion(motion=GripperState.CLOSE, gripper=r_eef),
            context=context,
        ).plan.perform()

        # Update world attachment in simulation
        if execution_env.execution_type == ExecutionType.SIMULATED:
            with world.modify_world():
                world.move_branch_with_fixed_connection(cereal, r_eef.tool_frame)

        # Step 5: Lift object
        logger.info("Step 5/8: Lifting object...")
        execute_single(
            MoveManipulatorAction(
                target_pose=lift_pose,
                end_effector=r_eef,
                allow_gripper_collision=True,
            ),
            context=context,
        ).plan.perform()

        # Step 6: Lower to place location
        logger.info("Step 6/8: Lowering object to place location...")
        execute_single(
            MoveManipulatorAction(
                target_pose=place_pose,
                end_effector=r_eef,
                allow_gripper_collision=True,
            ),
            context=context,
        ).plan.perform()

        # Step 7: Release object
        logger.info("Step 7/8: Opening gripper to release object...")
        execute_single(
            MoveGripperMotion(motion=GripperState.OPEN, gripper=r_eef),
            context=context,
        ).plan.perform()

        # Detach object in simulation
        if execution_env.execution_type == ExecutionType.SIMULATED:
            with world.modify_world():
                world.move_branch_with_fixed_connection(cereal, world.root)

        # Step 8: Park arms
        logger.info("Step 8/8: Parking both arms...")
        execute_single(
            ParkArmsAction(arms=list(pr2.all_arms)), context=context
        ).plan.perform()

    logger.info("Pick-and-place sequence completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="PR2 Pick-and-Place Demonstration")
    parser.add_argument(
        "--mode",
        choices=["sim", "real"],
        default="sim",
        help="Execution mode: 'sim' for local digital twin simulation, 'real' for Giskard hardware execution (default: sim)",
    )
    parser.add_argument(
        "--rviz",
        action="store_true",
        help="Start ROS 2 node and publish TF / MarkerArray for RViz2",
    )
    args = parser.parse_args()

    ros_node: Optional[Node] = None
    executor: Optional[MultiThreadedExecutor] = None

    if args.rviz or args.mode == "real":
        if not rclpy.ok():
            rclpy.init()
        ros_node = rclpy.create_node("pr2_pick_place_demo")
        executor = MultiThreadedExecutor()
        executor.add_node(ros_node)
        threading.Thread(target=executor.spin, daemon=True).start()
        logger.info("ROS 2 node active: %s", ros_node.get_name())

    world, pr2, cereal, cereal_ann, context = setup_demo_scene()

    if ros_node is not None:
        VizMarkerPublisher(_world=world, node=ros_node)
        logger.info("VizMarkerPublisher active on topic /semworld/viz_marker")

    execution_env = real_robot if args.mode == "real" else simulated_robot

    try:
        run_pick_and_place(
            world=world,
            pr2=pr2,
            cereal=cereal,
            cereal_ann=cereal_ann,
            context=context,
            execution_env=execution_env,
        )
    finally:
        if ros_node is not None:
            ros_node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
