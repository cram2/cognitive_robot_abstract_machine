from krrood.ormatic.utils import drop_database
from pycram.locations.locations import CostmapLocation
from pycram.motion_executor import (
    simulated_robot_without_collision,
)
from pycram.plans.factories import sequential
from pycram.robot_plans.actions.composite.tool_based import CuttingAction
from pycram.robot_plans.actions.core.navigation import NavigateAction
from pycram.robot_plans.actions.core.robot_body import (
    ParkArmsAction,
    MoveTorsoAction,
    SetGripperAction,
)

from pycram.tf_transformations import quaternion_from_euler, quaternion_multiply

# from demos.thesis_new.spawn_random_breads import setup_random_bread_world
from thesis_new.ideas.spawn_breads_luca import setup_random_bread_world

from thesis_new.src.spawn_random_breads import build_cutting_reachability_costmaps
from thesis_new.src.tool_mounts import get_tool_mount_pose_kwargs
from thesis_new.src.world_setup import resolve_robot_name
from demos.thesis_new.src.utils import (
    attach_available_tools,
    update_navigation_costmap_debug_publishers,
    collect_named_targets,
    get_park_arms_argument,
    highlight_current_target,
    set_entity_global_pose,
    setup_experiment_runtime,
    shutdown_experiment_runtime,
    commit_plan_to_db,
)
from demos.thesis_new.src.utils import (
    body_name as _body_name,
    format_attempt_error as _format_attempt_error,
)
import os
import numpy as np
from pycram.datastructures.dataclasses import Context
from pycram.datastructures.enums import Arms
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker


from semantic_digital_twin.adapters.mesh import STLParser

from semantic_digital_twin.datastructures.definitions import TorsoState, GripperState
from semantic_digital_twin.semantic_annotations.semantic_annotations import Knife
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.geometry import Color

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "..", "resources")
)
DEFAULT_BREAD_COLOR = Color(R=0.76, G=0.60, B=0.42)
ACTIVE_BREAD_COLOR = Color(R=0.52, G=0.82, B=0.98)
FAILED_BREAD_COLOR = Color(R=0.95, G=0.20, B=0.20)
SUCCESS_BREAD_COLOR = Color(R=0.62, G=0.92, B=0.62)
session = None


def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def _update_costmap_debug_publishers(node, robot, world, bread, publishers):
    target_pose = bread.global_pose
    occupancy, ring, final_map = build_cutting_reachability_costmaps(
        robot, world, target_pose
    )
    return update_navigation_costmap_debug_publishers(
        node,
        world,
        publishers,
        occupancy,
        ring,
        final_map,
        namespace_prefix="cutting",
    )


def _try_cut(context, bread, arm, tool):
    with simulated_robot_without_collision:
        sequential(
            [
                NavigateAction(
                    Pose(position=Point3(1, 1, 0), reference_frame=context.world.root),
                    teleport=True,
                ),
            ],
            context,
        ).perform()
        pickup_loc = CostmapLocation(
            target=bread.global_pose,
            reachable_arm=arm,
            validate_reachability=False,
            samples=1000,
            context=context,
        )

        # Tries to find a pick-up position for the robot that uses the given arm

    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(get_park_arms_argument(context.world)),
                MoveTorsoAction(TorsoState.HIGH),
                NavigateAction(pickup_loc, True, teleport=True),
            ],
            context,
        ).perform()

    with simulated_robot_without_collision:
        current_plan = sequential(
            [
                CuttingAction(
                    container=bread,
                    arm=arm,
                    tool=tool,
                    technique="saw",
                    pointer_stride=13,
                    num_cuts_x=1,
                ),
            ],
            context,
        )
        current_plan.perform()

    commit_plan_to_db(session, current_plan)


def _rotate_bread_180deg_z(world, bread):
    pose = bread.global_pose
    pos = np.asarray(pose.to_position().to_np(), dtype=float).reshape(-1)[:3]
    quat = np.asarray(pose.to_quaternion().to_np(), dtype=float).reshape(-1)[:4]
    rot_quat = quaternion_from_euler(0.0, 0.0, np.pi)
    new_quat = quaternion_multiply(rot_quat, quat)

    rotated_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=float(pos[0]),
        pos_y=float(pos[1]),
        pos_z=float(pos[2]),
        quat_x=float(new_quat[0]),
        quat_y=float(new_quat[1]),
        quat_z=float(new_quat[2]),
        quat_w=float(new_quat[3]),
        reference_frame=world.root,
    )
    set_entity_global_pose(world, bread, rotated_pose)


def main_cutting(seed=None, robot_name=None, environment_name=None):
    global session
    if session is None:
        session = pycram_sessionmaker()()
        drop_database(session.bind)
        Base.metadata.create_all(session.bind)
        session.commit()
        # print("commited")
    effective_seed = (
        int(seed)
        if seed is not None
        else int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    )
    world = setup_random_bread_world(
        seed=effective_seed,
        robot_name=robot_name,
        environment_name=environment_name,
    )

    node = setup_experiment_runtime(
        world=world,
        node_name="pycram_cut_all_breads_retry",
    )

    resolved_robot_name = resolve_robot_name(robot_name)
    arm_tools = attach_available_tools(
        world,
        _parse_stl,
        mesh_parts=("pycram_object_gap_demo", "big-knife.stl"),
        right_name="knife_right",
        left_name="knife_left",
        right_pose_kwargs=get_tool_mount_pose_kwargs(
            "cut", resolved_robot_name, Arms.RIGHT
        ),
        left_pose_kwargs=get_tool_mount_pose_kwargs(
            "cut", resolved_robot_name, Arms.LEFT
        ),
        tool_cls=Knife,
    )
    breads = collect_named_targets(world, "bread_")

    context = Context.from_world(world)
    context.ros_node = node
    with simulated_robot_without_collision:
        sequential([SetGripperAction(Arms.LEFT, GripperState.CLOSE)], context).perform()

    success_primary = 0
    success_fallback = 0
    success_rotated_right = 0
    success_rotated_left = 0
    failed = 0
    failed_breads = set()
    successful_breads = set()
    debug_costmap_publishers = {}

    with simulated_robot_without_collision:
        sequential(
            [
                ParkArmsAction(get_park_arms_argument(world)),
                MoveTorsoAction(TorsoState.HIGH),
            ],
            context,
        ).perform()
    for bread in breads:
        debug_costmap_publishers = _update_costmap_debug_publishers(
            node, context.robot, world, bread, debug_costmap_publishers
        )
        attempt_failures = []
        attempt_count = 0
        highlight_current_target(
            world,
            breads,
            bread,
            default_color=DEFAULT_BREAD_COLOR,
            active_color=ACTIVE_BREAD_COLOR,
            failed_color=FAILED_BREAD_COLOR,
            success_color=SUCCESS_BREAD_COLOR,
            failed_targets=failed_breads,
            successful_targets=successful_breads,
        )
        bread_name = _body_name(bread)
        arm_attempt_groups = [
            ("primary", arm_tools),
            ("after_rotation", arm_tools),
        ]
        attempt_succeeded = False

        for group_index, (phase_name, current_arm_tools) in enumerate(
            arm_attempt_groups
        ):
            if group_index == 1:
                print(f"[retry] {bread_name}: rotate 180deg around Z and try again")
                _rotate_bread_180deg_z(world, bread)

            for attempt_index, (arm, tool) in enumerate(current_arm_tools):
                is_primary_phase = group_index == 0 and attempt_index == 0
                is_fallback_phase = group_index == 0 and attempt_index > 0
                print(
                    f"[cut] {bread_name}: try {arm.name} arm"
                    + (" after rotation" if group_index == 1 else "")
                )
                _try_cut(context, bread, arm, tool)
                try:
                    attempt_count += 1
                    _try_cut(context, bread, arm, tool)
                    if is_primary_phase:
                        success_primary += 1
                    elif is_fallback_phase:
                        success_fallback += 1
                    elif attempt_index == 0:
                        success_rotated_right += 1
                    else:
                        success_rotated_left += 1
                    successful_breads.add(bread)
                    suffix = (
                        " after rotation"
                        if group_index == 1
                        else (" (fallback)" if attempt_index > 0 else "")
                    )
                    print(f"[ok] {bread_name}: cut with {arm.name} arm{suffix}")
                    attempt_succeeded = True
                    break
                except TimeoutError as exc:
                    attempt_failures.append(
                        f"{arm.name} {phase_name} -> {_format_attempt_error(exc)}"
                    )
                    print(
                        f"[{'retry' if not (group_index == len(arm_attempt_groups) - 1 and attempt_index == len(current_arm_tools) - 1) else 'fail'}] "
                        f"{bread_name}: {arm.name}"
                        + (" after rotation" if group_index == 1 else "")
                        + f" timed out ({type(exc).__name__}: {exc})"
                    )
                except Exception as exc:
                    attempt_failures.append(
                        f"{arm.name} {phase_name} -> {_format_attempt_error(exc)}"
                    )
                    print(
                        f"[{'retry' if not (group_index == len(arm_attempt_groups) - 1 and attempt_index == len(current_arm_tools) - 1) else 'fail'}] "
                        f"{bread_name}: {arm.name}"
                        + (" after rotation" if group_index == 1 else "")
                        + f" failed ({type(exc).__name__}: {exc})"
                    )
            if attempt_succeeded:
                break

        if attempt_succeeded:
            continue

        failed += 1
        failed_breads.add(bread)
        print(f"[fail] {bread_name}: all attempts failed")

    highlight_current_target(
        world,
        breads,
        None,
        default_color=DEFAULT_BREAD_COLOR,
        active_color=ACTIVE_BREAD_COLOR,
        failed_color=FAILED_BREAD_COLOR,
        success_color=SUCCESS_BREAD_COLOR,
        failed_targets=failed_breads,
        successful_targets=successful_breads,
    )

    print("[summary]")
    print(f"  total breads: {len(breads)}")
    print(f"  success primary (RIGHT): {success_primary}")
    print(f"  success fallback (LEFT): {success_fallback}")
    print(f"  success after rotation (RIGHT): {success_rotated_right}")
    print(f"  success after rotation (LEFT): {success_rotated_left}")
    print(f"  failed both arms: {failed}")

    shutdown_experiment_runtime(node)


# if __name__ == "__main__":
#     session = pycram_sessionmaker()()
#     # drop_database(session.bind)
#     Base.metadata.create_all(session.bind)
#     session.commit()
#
#     main_cutting()
