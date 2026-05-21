import os
import numpy as np
import rclpy

from demos.thesis_new.src.utils.demo_utils import build_navigation_costmaps
from pycram.datastructures.dataclasses import Context
from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.tfwrapper import TFWrapper
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
    ShapeSource,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Table
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color, Scale


from pycram.robot_plans.actions.composite.thesis_math import body_local_aabb
from .world_setup import resolve_environment_name, setup_thesis_world

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "..", "resources")
)

GENERIC_SUPPORT_EXCLUDE_KEYWORDS = (
    "fridge",
    "cabinet",
    "drawer",
    "door",
    "oven",
    "dish_washer",
    "dishwasher",
    "handle",
)
PREFERRED_SURFACE_NAMES = (
    "island_countertop",
    "countertop",
    "table_area_main",
    # "coffee_table",
    # "bedside_table",
    # "kitchen_island_surface",
    # "sink_area_surface",
)
ENVIRONMENT_ALLOWED_SURFACE_NAME_KEYWORDS = {
    # Keep ISR bread generation on tables only.
    "isr": ("dinning_room_table", "table_living_room", "table_bedside"),
    "isr-testbed": ("dinning_room_table", "table_living_room", "table_bedside"),
}
MIN_SUPPORT_SURFACE_AREA_M2 = 0.025
MIN_SUPPORT_SURFACE_SPAN_M = 0.18
MAX_SUPPORT_SURFACE_TOP_Z_M = 1.30
MAX_SPAWN_OBJECT_WORLD_Z_M = 1.30
SPAWN_OBSTACLE_VERTICAL_MARGIN_M = 0.03
SPAWN_OBSTACLE_TOP_EPSILON_M = 0.01

# Automatic count model: breads ~= usable_surface_area * BREADS_PER_SQM.
# Keep this tunable when switching to a new environment.
BREADS_PER_SQM = 12.0
MIN_BREADS_PER_SURFACE = 0
MAX_BREADS_PER_SURFACE = 30
STRICT_CLEAN_MODE = True
MIN_BREAD_CLEARANCE_M = 0.03
BREAD_RADIUS_SAFETY_FACTOR = 1.08

# Optional per-surface overrides (set value to force exact count).
SURFACE_COUNT_OVERRIDES = {}
COUNTERTOP_TINT = Color(R=0.82, G=0.70, B=0.55)
DEBUG_SPAWN_FRAMES = True
DEBUG_FORCE_SPAWN_ALL_TARGETS = False
DEBUG_DISABLE_REACHABILITY = True
DEBUG_SPAWN_ACTUAL_POSES = True
MAX_PLACEMENT_ATTEMPTS_PER_OBJECT = 60

CUT_OBJECT_CONFIGS = {
    "bread": {
        "object_label": "bread",
        "object_name_prefix": "bread",
        "mesh_parts": ("pycram_object_gap_demo", "bread.stl"),
        "object_color": Color(R=0.76, G=0.60, B=0.42),
    },
    "cucumber": {
        "object_label": "cucumber",
        "object_name_prefix": "cucumber",
        "mesh_parts": ("pycram_object_gap_demo", "cucumber.stl"),
        "object_color": Color(R=0.26, G=0.66, B=0.30),
    },
    "apple": {
        "object_label": "apple",
        "object_name_prefix": "apple",
        "mesh_parts": ("pycram_object_gap_demo", "apple.stl"),
        "object_color": Color(R=0.84, G=0.18, B=0.16),
    },
}


def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def get_cut_object_config(object_kind="bread"):
    try:
        return CUT_OBJECT_CONFIGS[str(object_kind).lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(CUT_OBJECT_CONFIGS))
        raise ValueError(
            f"Unsupported cut object_kind '{object_kind}'. Supported: {supported}"
        ) from exc


def _set_uniform_scale(body, scale_xyz, color=None):
    scale = Scale(*scale_xyz)
    for shape in body.root.visual.shapes:
        shape.scale = scale
        if color is not None:
            shape.color = color
    for shape in body.root.collision.shapes:
        shape.scale = scale


def _body_name(body):
    maybe_name = getattr(body, "name", None)
    if hasattr(maybe_name, "name"):
        maybe_name = maybe_name.name
    return maybe_name if isinstance(maybe_name, str) else None


def _body_basename(body_or_name):
    name = body_or_name if isinstance(body_or_name, str) else _body_name(body_or_name)
    if not name:
        return None
    return str(name).split("/")[-1]


def _frame_name(frame):
    if frame is None:
        return "None"
    maybe_name = getattr(frame, "name", None)
    if hasattr(maybe_name, "name"):
        maybe_name = maybe_name.name
    if isinstance(maybe_name, str):
        return maybe_name
    return str(frame)


def _surface_like_name(name):
    basename = (_body_basename(name) or "").lower()
    if not basename:
        return False

    if basename.endswith("_surface"):
        return True

    return False


def _surface_geometry_is_usable(body):
    basename = (_body_basename(body) or "").lower()
    if any(skip in basename for skip in GENERIC_SUPPORT_EXCLUDE_KEYWORDS):
        return False

    try:
        mins, maxs = body_local_aabb(body, use_visual=False, apply_shape_scale=True)
    except Exception:
        return False

    extents = maxs - mins
    if not np.all(np.isfinite(extents)):
        return False

    span_x = float(extents[0])
    span_y = float(extents[1])
    top_z = float(maxs[2])
    area = max(0.0, span_x) * max(0.0, span_y)
    if area < MIN_SUPPORT_SURFACE_AREA_M2:
        return False
    if min(span_x, span_y) < MIN_SUPPORT_SURFACE_SPAN_M:
        return False
    if top_z <= 0.2 or top_z > MAX_SUPPORT_SURFACE_TOP_Z_M:
        return False
    return True


def _collect_surfaces_by_geometry(world, seen):
    surfaces = []
    robot_body_ids, robot_body_names = _robot_body_identity_sets(world)
    for body in getattr(world, "bodies", []):
        name = _body_name(body)
        basename = (_body_basename(name) or "").lower()
        if not name or name in seen or not basename:
            continue
        if _is_robot_body(body, name, robot_body_ids, robot_body_names):
            continue
        if any(skip in basename for skip in GENERIC_SUPPORT_EXCLUDE_KEYWORDS):
            continue
        if not _surface_geometry_is_usable(body):
            continue
        surfaces.append(body)
        seen.add(name)
    return surfaces


def _robot_body_identity_sets(world):
    robot_root = None
    robot_bodies = []

    try:
        from semantic_digital_twin.robots.abstract_robot import AbstractRobot
        robot_annotations = [
            a for a in getattr(world, "semantic_annotations", [])
            if isinstance(a, AbstractRobot)
        ]
        if robot_annotations:
            robot = robot_annotations[0]
            robot_root = robot.root
            # Use Robot.bodies directly instead of traversing the graph
            robot_bodies = list(getattr(robot, "bodies", []))

    except Exception:
        pass

    if not robot_bodies and robot_root is not None:
        robot_bodies = [robot_root]

    if not robot_bodies:
        return set(), set()

    robot_ids = set()
    robot_names = set()
    for body in robot_bodies:
        name = _body_name(body)
        if name:
            robot_names.add(name)
        bid = getattr(body, "id", None)
        if bid is not None:
            robot_ids.add(bid)

    robot_ids.discard(None)
    return robot_ids, robot_names


def _is_robot_body(body, name, robot_body_ids, robot_body_names):
    return getattr(body, "id", None) in robot_body_ids or name in robot_body_names


def _world_aabb_for_body(world, body):
    try:
        mins, maxs = body_local_aabb(body, use_visual=False, apply_shape_scale=True)
    except Exception:
        return None

    corners = []
    for x in (mins[0], maxs[0]):
        for y in (mins[1], maxs[1]):
            for z in (mins[2], maxs[2]):
                local_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    roll=0.0,
                    pitch=0.0,
                    yaw=0.0,
                    reference_frame=body,
                )
                try:
                    corners.append(_pose_xyz(world.transform(local_pose, world.root)))
                except Exception:
                    return None
    if not corners:
        return None
    corners = np.asarray(corners, dtype=float)
    if not np.all(np.isfinite(corners)):
        return None
    return corners.min(axis=0), corners.max(axis=0)


def _candidate_overlaps_existing_body(
    *,
    world,
    candidate_world_pose,
    radius,
    support_body,
    robot_body_ids,
    robot_body_names,
):
    x, y, z = _pose_xyz(candidate_world_pose)
    for body in getattr(world, "bodies", []):
        if body is support_body:
            continue
        name = _body_name(body)
        if not name:
            continue
        if _is_robot_body(body, name, robot_body_ids, robot_body_names):
            continue
        aabb = _world_aabb_for_body(world, body)
        if aabb is None:
            continue
        mins, maxs = aabb
        if float(z) < float(mins[2]) - SPAWN_OBSTACLE_VERTICAL_MARGIN_M:
            continue
        if float(z) >= float(maxs[2]) - SPAWN_OBSTACLE_TOP_EPSILON_M:
            continue
        if float(x) < float(mins[0]) - radius:
            continue
        if float(x) > float(maxs[0]) + radius:
            continue
        if float(y) < float(mins[1]) - radius:
            continue
        if float(y) > float(maxs[1]) + radius:
            continue
        return True, name
    return False, ""


def _collect_surface_bodies(world):
    surfaces = []
    seen = set()
    robot_body_ids, robot_body_names = _robot_body_identity_sets(world)

    preferred_basenames = {name.lower() for name in PREFERRED_SURFACE_NAMES}
    for body in getattr(world, "bodies", []):
        name = _body_name(body)
        basename = (_body_basename(name) or "").lower()
        if not name or not basename or basename not in preferred_basenames:
            continue
        if any(skip in basename for skip in GENERIC_SUPPORT_EXCLUDE_KEYWORDS):
            continue
        if _is_robot_body(body, name, robot_body_ids, robot_body_names):
            continue
        if name in seen:
            continue
        if not _surface_geometry_is_usable(body):
            continue
        surfaces.append(body)
        seen.add(name)

    # If explicitly preferred surfaces exist in the scene, keep the sampling scope
    # limited to them instead of also traversing every generic table/counter body.
    if surfaces:
        return surfaces

    for body in getattr(world, "bodies", []):
        name = _body_name(body)
        if not name or name in seen:
            continue

        if _is_robot_body(body, name, robot_body_ids, robot_body_names):
            continue
        basename = (_body_basename(name) or "").lower()
        if any(skip in basename for skip in GENERIC_SUPPORT_EXCLUDE_KEYWORDS):
            continue
        if not _surface_like_name(name):
            continue
        if not _surface_geometry_is_usable(body):
            continue
        surfaces.append(body)
        seen.add(name)

    if surfaces:
        return surfaces

    return _collect_surfaces_by_geometry(world, seen)


def _filter_surfaces_for_environment(surfaces, environment_name):
    normalized_environment = resolve_environment_name(environment_name)
    allowed_keywords = ENVIRONMENT_ALLOWED_SURFACE_NAME_KEYWORDS.get(
        normalized_environment
    )
    if not allowed_keywords:
        return surfaces

    filtered = []
    for surface in surfaces:
        name = (_body_name(surface) or "").lower()
        if any(keyword in name for keyword in allowed_keywords):
            filtered.append(surface)

    if not filtered:
        allowed = ", ".join(sorted(allowed_keywords))
        raise RuntimeError(
            "No allowed support surfaces found for "
            f"{normalized_environment}. Expected one of: {allowed}"
        )

    return filtered


def _iter_visual_shapes(body):
    seen = set()
    for geom_owner in (body, getattr(body, "root", None)):
        if geom_owner is None:
            continue
        geom = getattr(geom_owner, "visual", None)
        if geom is None:
            continue
        for shape in getattr(geom, "shapes", []):
            sid = id(shape)
            if sid in seen:
                continue
            seen.add(sid)
            yield shape


def _tint_surfaces_light_brown(world):
    for body in getattr(world, "bodies", []):
        name = _body_name(body) or ""
        if not _surface_like_name(name):
            continue
        for shape in _iter_visual_shapes(body):
            shape.color = COUNTERTOP_TINT


def _surface_sampling_bounds(surface_body):
    mins, maxs = body_local_aabb(surface_body, use_visual=False, apply_shape_scale=True)
    extents = maxs - mins

    margin_x = min(0.08, max(0.0, 0.25 * extents[0]))
    margin_y = min(0.08, max(0.0, 0.25 * extents[1]))

    lo_x = mins[0] + margin_x
    hi_x = maxs[0] - margin_x
    lo_y = mins[1] + margin_y
    hi_y = maxs[1] - margin_y

    return mins, maxs, lo_x, hi_x, lo_y, hi_y


def _sample_xy(lo_x, hi_x, lo_y, hi_y, mins, maxs, rng):
    if hi_x <= lo_x:
        x_local = 0.5 * (mins[0] + maxs[0])
    else:
        x_local = float(rng.uniform(lo_x, hi_x))
    if hi_y <= lo_y:
        y_local = 0.5 * (mins[1] + maxs[1])
    else:
        y_local = float(rng.uniform(lo_y, hi_y))
    return x_local, y_local


def _base_xy_radius_for_mesh(*mesh_parts):
    """Approximate object XY footprint as a circle radius in local object frame."""
    bread = _parse_stl(*mesh_parts)
    mins, maxs = body_local_aabb(bread.root, use_visual=False, apply_shape_scale=True)
    dx = max(0.0, float(maxs[0] - mins[0]))
    dy = max(0.0, float(maxs[1] - mins[1]))
    return 0.5 * np.hypot(dx, dy)


def _bread_base_xy_radius():
    return _base_xy_radius_for_mesh("pycram_object_gap_demo", "bread.stl")


def _surface_usable_area(lo_x, hi_x, lo_y, hi_y):
    width = max(0.0, float(hi_x - lo_x))
    depth = max(0.0, float(hi_y - lo_y))
    return width * depth


def _count_for_surface(surface_name, area_m2):
    if surface_name in SURFACE_COUNT_OVERRIDES:
        return int(SURFACE_COUNT_OVERRIDES[surface_name])
    raw = int(round(area_m2 * BREADS_PER_SQM))
    return max(MIN_BREADS_PER_SURFACE, min(MAX_BREADS_PER_SURFACE, raw))


def _pose_xyz(spatial_pose):
    position = spatial_pose.to_position()
    if hasattr(position, "is_constant") and position.is_constant():
        xyz = position.to_np()
    elif hasattr(position, "evaluate"):
        xyz = np.asarray(position.evaluate(), dtype=float).reshape(-1)
    else:
        xyz = np.asarray(position.to_np(), dtype=float).reshape(-1)
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def _is_excluded_kitchen_spawn_pose(spatial_pose, *, environment_name=None):
    if resolve_environment_name(environment_name) not in {
        "kitchen",
        "test-kitchen-chat",
    }:
        return False
    x, y, _ = _pose_xyz(spatial_pose)
    return float(np.hypot(x, y)) < 0.75


def _spawn_bread_at_local_pose(
    world, surface_body, bread_name, scale, x_local, y_local, yaw, z_local
):
    return _spawn_object_at_local_pose(
        world=world,
        surface_body=surface_body,
        object_name=bread_name,
        scale=scale,
        x_local=x_local,
        y_local=y_local,
        yaw=yaw,
        z_local=z_local,
        mesh_parts=("pycram_object_gap_demo", "bread.stl"),
        color=Color(R=0.76, G=0.60, B=0.42),
        debug_log=DEBUG_SPAWN_ACTUAL_POSES,
    )


def _spawn_object_at_local_pose(
    *,
    world,
    surface_body,
    object_name,
    scale,
    x_local,
    y_local,
    yaw,
    z_local,
    mesh_parts,
    color,
    debug_log=False,
):
    spawned = _parse_stl(*mesh_parts)
    spawned_root_id = spawned.root.id
    spawned.root.name.name = object_name
    _set_uniform_scale(spawned, (scale, scale, scale), color=color)
    local_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=float(x_local),
        y=float(y_local),
        z=float(z_local),
        roll=0.0,
        pitch=0.0,
        yaw=float(yaw),
        reference_frame=surface_body,
    )
    world_pose = world.transform(local_pose, world.root)
    world.merge_world_at_pose(spawned, world_pose)
    world.update_forward_kinematics()
    if debug_log:
        try:
            spawned_body = world.get_kinematic_structure_entity_by_id(spawned_root_id)
            actual_pose = world.transform(spawned_body.global_transform, world.root)
            print(
                f"[spawn-debug] inserted {object_name} "
                f"target={_pose_xyz(world_pose)} actual={_pose_xyz(actual_pose)}"
            )
        except Exception as exc:
            print(f"[spawn-debug] inserted {object_name} pose lookup failed: {exc}")
    return world_pose


def _sample_random_surface_layout(
    *,
    world,
    seed,
    environment_name,
    object_label,
    object_name_prefix,
    mesh_parts,
    object_color,
    scale_choices,
    base_radius,
    radius_safety_factor,
    min_clearance_m,
    strict_clean_mode,
    z_offset,
    reachability_fn,
    debug_disable_reachability,
    debug_force_spawn_all_targets,
    debug_spawn_actual_poses,
    spawn_objects=True,
    include_world_pose=False,
):
    rng = np.random.default_rng(seed)

    surfaces = _filter_surfaces_for_environment(
        _collect_surface_bodies(world), environment_name
    )
    if not surfaces:
        raise RuntimeError(
            f"No support surfaces found for random {object_label} placement."
        )
    print(f"[spawn] evaluating {len(surfaces)} support surfaces")
    if DEBUG_SPAWN_FRAMES:
        print(f"[spawn-debug] world_root={_frame_name(world.root)}")


    spawn_robot = None
    if not debug_disable_reachability:
        spawn_robot = Context.from_world(world).robot
    placements = []
    surface_plan = []
    created_idx = 0
    reachability_cache = {}
    robot_body_ids, robot_body_names = _robot_body_identity_sets(world)

    with world.modify_world():
        _tint_surfaces_light_brown(world)
        for surface_body in surfaces:
            surface_name = _body_name(surface_body) or "unknown_surface"
            if DEBUG_SPAWN_FRAMES:
                print(
                    "[spawn-debug] surface="
                    f"{surface_name} frame={_frame_name(surface_body)}"
                )
            table = Table(root=surface_body)
            world.add_semantic_annotation(table)
            table.calculate_supporting_surface()
            world.update_forward_kinematics()
            mins, maxs, lo_x, hi_x, lo_y, hi_y = _surface_sampling_bounds(surface_body)
            z_local = float(maxs[2] + z_offset)
            area_m2 = _surface_usable_area(lo_x, hi_x, lo_y, hi_y)
            target_count = _count_for_surface(surface_name, area_m2)
            print(
                f"[spawn] surface={surface_name} area={area_m2:.3f}m^2 target={target_count}"
            )
            occupied_xy = []
            reject_no_points = 0
            reject_clearance = 0
            reject_reachability = 0
            reject_overlap = 0
            for _ in range(target_count):
                scale = float(rng.choice(scale_choices))
                radius = base_radius * scale * radius_safety_factor
                placed = False
                for _attempt in range(MAX_PLACEMENT_ATTEMPTS_PER_OBJECT):
                    lo_x_eff = lo_x + radius
                    hi_x_eff = hi_x - radius
                    lo_y_eff = lo_y + radius
                    hi_y_eff = hi_y - radius

                    if hi_x_eff <= lo_x_eff or hi_y_eff <= lo_y_eff:
                        reject_no_points += 1
                        break

                    x_local, y_local = _sample_xy(
                        lo_x_eff, hi_x_eff, lo_y_eff, hi_y_eff, mins, maxs, rng
                    )

                    if not debug_force_spawn_all_targets:
                        if not all(
                            ((x_local - ox) ** 2 + (y_local - oy) ** 2)
                            >= (
                                radius
                                + orad
                                + (min_clearance_m if strict_clean_mode else 0.015)
                            )
                            ** 2
                            for ox, oy, orad in occupied_xy
                        ):
                            reject_clearance += 1
                            continue

                    yaw = float(rng.uniform(-np.pi, np.pi))
                    candidate_surface_pose = (
                        HomogeneousTransformationMatrix.from_xyz_rpy(
                            x=x_local,
                            y=y_local,
                            z=z_local,
                            roll=0.0,
                            pitch=0.0,
                            yaw=float(yaw),
                            reference_frame=surface_body,
                        )
                    )
                    candidate_world_pose = world.transform(
                        candidate_surface_pose, world.root
                    )
                    if _is_excluded_kitchen_spawn_pose(
                        candidate_world_pose, environment_name=environment_name
                    ):
                        reject_clearance += 1
                        continue
                    candidate_position = candidate_world_pose.to_position()
                    if float(candidate_position.z) > MAX_SPAWN_OBJECT_WORLD_Z_M:
                        reject_clearance += 1
                        continue
                    overlaps_body, overlap_name = _candidate_overlaps_existing_body(
                        world=world,
                        candidate_world_pose=candidate_world_pose,
                        radius=radius,
                        support_body=surface_body,
                        robot_body_ids=robot_body_ids,
                        robot_body_names=robot_body_names,
                    )
                    if overlaps_body:
                        reject_overlap += 1
                        if DEBUG_SPAWN_FRAMES and reject_overlap <= 3:
                            print(
                                "[spawn-debug] rejected overlap "
                                f"surface={surface_name} body={overlap_name} "
                                f"xyz={_pose_xyz(candidate_world_pose)}"
                            )
                        continue
                    if DEBUG_SPAWN_FRAMES and created_idx == 0:
                        print(
                            "[spawn-debug] candidate "
                            f"surface_frame={_frame_name(surface_body)} "
                            f"world_frame={_frame_name(candidate_world_pose.reference_frame)} "
                            f"xyz={_pose_xyz(candidate_world_pose)}"
                        )

                    if (
                        not debug_force_spawn_all_targets
                        and not debug_disable_reachability
                    ):
                        if not reachability_fn(
                            spawn_robot,
                            world,
                            candidate_world_pose,
                            cache=reachability_cache,
                        ):
                            reject_reachability += 1
                            continue

                    created_idx += 1
                    object_name = f"{object_name_prefix}_{created_idx:04d}"
                    if spawn_objects:
                        world_pose = _spawn_object_at_local_pose(
                            world=world,
                            surface_body=surface_body,
                            object_name=object_name,
                            scale=scale,
                            x_local=x_local,
                            y_local=y_local,
                            yaw=yaw,
                            z_local=z_local,
                            mesh_parts=mesh_parts,
                            color=object_color,
                            debug_log=debug_spawn_actual_poses,
                        )
                    else:
                        world_pose = candidate_world_pose
                    occupied_xy.append((x_local, y_local, radius))
                    placement = (
                        object_name,
                        surface_name,
                        scale,
                        _pose_xyz(world_pose),
                    )
                    if include_world_pose:
                        placement = (*placement, world_pose)
                    placements.append(placement)
                    placed = True
                    if placed:
                        break
                if not placed:
                    break
            surface_plan.append((surface_name, area_m2, target_count, len(occupied_xy)))
            print(
                f"[spawn] surface={surface_name} placed={len(occupied_xy)}/{target_count} "
                f"rejects(no_points={reject_no_points}, "
                f"clearance={reject_clearance}, "
                f"overlap={reject_overlap}, "
                f"reachability={reject_reachability})"
            )

    return world, placements, surface_plan


def _spawn_bread_at_surface_point(world, point_on_surface, bread_name, scale, yaw):
    bread = _parse_stl("pycram_object_gap_demo", "bread.stl")
    bread.root.name.name = bread_name
    _set_uniform_scale(
        bread,
        (scale, scale, scale),
        color=Color(R=0.76, G=0.60, B=0.42),
    )
    surface_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=float(point_on_surface.x),
        y=float(point_on_surface.y),
        z=float(point_on_surface.z),
        roll=0.0,
        pitch=0.0,
        yaw=float(yaw),
        reference_frame=point_on_surface.reference_frame,
    )
    world_pose = world.transform(surface_pose, world.root)
    world.merge_world_at_pose(bread, world_pose)
    return world_pose


def build_cutting_reachability_costmaps(robot, world, target_pose):
    """Build the occupancy, ring, and merged heuristic maps for a cutting target."""
    base_bb = robot.base.bounding_box
    base_depth = float(getattr(base_bb, "depth", float("nan")))
    base_width = float(getattr(base_bb, "width", float("nan")))
    obstacle_clearance = 0.25 * (base_depth + base_width)
    if (not np.isfinite(obstacle_clearance)) or obstacle_clearance <= 0.0:
        obstacle_clearance = 0.20

    return build_navigation_costmaps(
        robot,
        world,
        target_pose,
        width=140,
        height=140,
        resolution=0.03,
        ring_std=12,
        ring_distance=0.55,
        obstacle_clearance=obstacle_clearance,
        number_of_samples=30,
    )


def _coarse_reachability_key(target_pose, grid_size=0.10):
    position = target_pose.to_position()
    return (
        int(round(float(position.x) / grid_size)),
        int(round(float(position.y) / grid_size)),
        str(target_pose.reference_frame),
    )


def _is_pose_reachable_for_cutting(robot, world, target_pose, cache=None):
    """
    Coarse reachability gate for world generation.

    Spawning should not do full-resolution navigation planning for every candidate
    object pose; that is too expensive and the actual execution path validates
    navigation again anyway. Use a cached low-resolution probe to reject clearly
    unreachable placements without stalling setup.
    """
    if cache is not None:
        key = _coarse_reachability_key(target_pose)
        if key in cache:
            return cache[key]

    try:
        _, _, final_map = build_navigation_costmaps(
            robot,
            world,
            target_pose,
            width=60,
            height=60,
            resolution=0.05,
            ring_std=8,
            ring_distance=0.55,
            obstacle_clearance=0.20,
            number_of_samples=8,
        )

        next(iter(final_map))
        result = True
    except StopIteration:
        result = False
    except Exception:
        # Scene generation should not hard-fail on planner internals; the actual
        # action execution performs the authoritative reachability check later.
        result = True

    if cache is not None:
        cache[key] = result
    return result


def setup_random_bread_world(
    seed=None, robot_name=None, environment_name=None, object_kind="bread"
):
    object_cfg = get_cut_object_config(object_kind)
    world = setup_thesis_world(robot_name=robot_name, environment_name=environment_name)
    return _sample_random_surface_layout(
        world=world,
        seed=seed,
        environment_name=environment_name,
        object_label=object_cfg["object_label"],
        object_name_prefix=object_cfg["object_name_prefix"],
        mesh_parts=object_cfg["mesh_parts"],
        object_color=object_cfg["object_color"],
        scale_choices=np.array([0.8, 1.0, 1.2, 1.4, 1.6], dtype=float),
        base_radius=_base_xy_radius_for_mesh(*object_cfg["mesh_parts"]),
        radius_safety_factor=BREAD_RADIUS_SAFETY_FACTOR,
        min_clearance_m=MIN_BREAD_CLEARANCE_M,
        strict_clean_mode=STRICT_CLEAN_MODE,
        z_offset=0.03,
        reachability_fn=_is_pose_reachable_for_cutting,
        debug_disable_reachability=DEBUG_DISABLE_REACHABILITY,
        debug_force_spawn_all_targets=DEBUG_FORCE_SPAWN_ALL_TARGETS,
        debug_spawn_actual_poses=DEBUG_SPAWN_ACTUAL_POSES,
    )


def main(seed=None):
    world, placements, surface_plan = setup_random_bread_world(seed=seed)

    rclpy.init()
    node = rclpy.create_node("pycram_random_breads_demo")
    tf_wrapper = TFWrapper(node=node)
    TFPublisher(node=node, _world=world)
    VizMarkerPublisher(_world=world, node=node, shape_source=ShapeSource.COLLISION_ONLY)

    # tf_wrapper.wait_for_transform(
    #     "apartment/apartment_root",
    #     "pr2/base_footprint",
    #     timeout=RclpyDuration(seconds=1.0),
    #     time=Time(),
    # )

    for surface_name, area_m2, target_count, placed_count in surface_plan:
        print(
            f"[surface] {surface_name}: usable_area={area_m2:.3f} m^2 "
            f"target={target_count} placed={placed_count}"
        )

    for bread_name, surface_name, scale, (x, y, z) in placements:
        print(
            f"{bread_name} spawned on {surface_name} at "
            f"({x:.3f}, {y:.3f}, {z:.3f}) scale={scale:.2f}"
        )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
