import os
import re
import numpy as np

from semantic_digital_twin.adapters.mesh import STLParser
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.geometry import Color

from .spawn_random_breads import (
    MAX_SPAWN_OBJECT_WORLD_Z_M,
    _body_basename,
    _body_name,
    _candidate_overlaps_existing_body,
    _is_excluded_kitchen_spawn_pose,
    _pose_xyz,
    _robot_body_identity_sets,
    _sample_random_surface_layout,
    body_local_aabb,
)
from .world_setup import resolve_environment_name, setup_thesis_world
from .utils.demo_utils import (
    build_navigation_costmaps,
)

RESOURCES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..", "..", "resources")
)
BOWL_RADIUS_SAFETY_FACTOR = 1.08
MIN_BOWL_CLEARANCE_M = 0.03
STRICT_CLEAN_MODE = True
DEFAULT_BOWL_COLOR = Color(R=0.78, G=0.80, B=0.86)
DEFAULT_POT_COLOR = Color(R=0.56, G=0.58, B=0.62)
WIPE_TARGET_KEEP_FRACTION = 0.5
WIPE_TARGET_SURFACE_OFFSET_M = 0.005
WIPE_TARGET_RADIUS_M = 0.05
WIPE_TARGET_CLEARANCE_M = 0.02
TABLE_ONLY_WIPE_TARGET_ENVIRONMENTS = {"isr", "isr-testbed"}
VERTICAL_WIPE_SURFACE_CANDIDATES = (
    ("cabinet3_door_top_left", 1, -1),
    ("cabinet3_door_bottom_left", 1, -1),
    ("cabinet9_drawer_top", 1, -1),
    ("cabinet9_drawer_middle", 1, -1),
    ("cabinet9_drawer_bottom", 1, -1),
    ("iai_fridge_door", 1, 1),
    ("oven_area_oven_door", 1, 1),
    ("sink_area_dish_washer_door", 1, 1),
)
VERTICAL_WIPE_TARGETS_PER_SQM = 3.0
MAX_VERTICAL_WIPE_TARGETS_PER_SURFACE = 3
AUTO_CABINET_FRONT_FACE_SIGN = -1
AUTO_CABINET_EXCLUDE_KEYWORDS = (
    "handle",
    "out_fancy",
    "footprint",
    "joint",
    "kitchen_island",
)
AUTO_CABINET_FRONT_RE = re.compile(r"^cabinet\d+($|_(door|drawer)(_|$))")


def _parse_stl(*relative_path_parts):
    return STLParser(os.path.join(RESOURCES_DIR, *relative_path_parts)).parse()


def _container_base_xy_radius(*mesh_parts):
    bowl = _parse_stl(*mesh_parts)
    mins, maxs = body_local_aabb(bowl.root, use_visual=False, apply_shape_scale=True)
    dx = max(0.0, float(maxs[0] - mins[0]))
    dy = max(0.0, float(maxs[1] - mins[1]))
    return 0.5 * np.hypot(dx, dy)


def _bowl_base_xy_radius():
    return _container_base_xy_radius("objects", "bowl.stl")


def _sample_vertical_wipe_targets(
    world, rng, surface_name, count, start_idx, *, face_sign=1, environment_name=None
):
    if "kitchen_island" in str(surface_name).lower():
        return [], []
    try:
        surface_body = world.get_body_by_name(surface_name)
    except Exception:
        return [], []

    mins, maxs = body_local_aabb(surface_body, use_visual=False, apply_shape_scale=True)
    extents = maxs - mins
    y_margin = min(0.08, max(0.0, 0.18 * extents[1]))
    z_margin = min(0.10, max(0.0, 0.18 * extents[2]))
    lo_y = mins[1] + y_margin
    hi_y = maxs[1] - y_margin
    lo_z = mins[2] + z_margin
    hi_z = maxs[2] - z_margin

    placements = []
    occupied_yz = []
    robot_body_ids, robot_body_names = _robot_body_identity_sets(world)
    face_sign = 1 if float(face_sign) >= 0.0 else -1
    door_offset = 0.03
    yz_clearance = 0.18

    for local_idx in range(count):
        for _attempt in range(80):
            if hi_y <= lo_y:
                y_local = 0.5 * (mins[1] + maxs[1])
            else:
                y_local = float(rng.uniform(lo_y, hi_y))
            if hi_z <= lo_z:
                z_local = 0.5 * (mins[2] + maxs[2])
            else:
                z_local = float(rng.uniform(lo_z, hi_z))

            if any(
                ((y_local - oy) ** 2 + (z_local - oz) ** 2) < yz_clearance**2
                for oy, oz in occupied_yz
            ):
                continue

            local_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
                x=float(
                    (maxs[0] + door_offset)
                    if face_sign > 0
                    else (mins[0] - door_offset)
                ),
                y=float(y_local),
                z=float(z_local),
                roll=0.0,
                pitch=face_sign * np.pi / 2,
                yaw=0.0,
                reference_frame=surface_body,
            )
            world_pose = world.transform(local_pose, world.root)
            if _is_excluded_kitchen_spawn_pose(
                world_pose, environment_name=environment_name
            ):
                continue
            if float(world_pose.to_position().z) > MAX_SPAWN_OBJECT_WORLD_Z_M:
                continue
            overlaps_body, _ = _candidate_overlaps_existing_body(
                world=world,
                candidate_world_pose=world_pose,
                radius=WIPE_TARGET_RADIUS_M,
                support_body=surface_body,
                robot_body_ids=robot_body_ids,
                robot_body_names=robot_body_names,
            )
            if overlaps_body:
                continue
            placements.append(
                {
                    "bowl_name": f"wipe_target_{start_idx + local_idx:04d}",
                    "surface_name": str(surface_body.name),
                    "scale": 1.0,
                    "pose_xyz": _pose_xyz(world_pose),
                    "world_pose": world_pose,
                }
            )
            occupied_yz.append((y_local, z_local))
            break

    surface_area = max(0.0, float(hi_y - lo_y)) * max(0.0, float(hi_z - lo_z))
    surface_plan_entry = (
        str(surface_body.name),
        surface_area,
        count,
        len(placements),
    )
    return placements, [surface_plan_entry]


def _is_pose_reachable_for_mixing(robot, world, target_pose, cache=None):
    key = None
    if cache is not None:
        position = target_pose.to_position()
        key = (
            int(round(float(position.x) / 0.10)),
            int(round(float(position.y) / 0.10)),
            str(target_pose.reference_frame),
        )
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
        result = True

    if cache is not None and key is not None:
        cache[key] = result
    return result


def _vertical_wipe_target_count(surface_body):
    try:
        mins, maxs = body_local_aabb(
            surface_body, use_visual=False, apply_shape_scale=True
        )
    except Exception:
        return 1
    extents = maxs - mins
    usable_y = max(0.0, float(extents[1]) - 0.16)
    usable_z = max(0.0, float(extents[2]) - 0.20)
    area_m2 = usable_y * usable_z
    count = int(round(area_m2 * VERTICAL_WIPE_TARGETS_PER_SQM))
    return max(1, min(MAX_VERTICAL_WIPE_TARGETS_PER_SURFACE, count))


def _is_auto_cabinet_front_candidate(body_name):
    basename = (_body_basename(body_name) or "").lower()
    if not basename:
        return False
    if any(keyword in basename for keyword in AUTO_CABINET_EXCLUDE_KEYWORDS):
        return False
    return AUTO_CABINET_FRONT_RE.match(basename) is not None


def _resolve_vertical_wipe_surfaces(world):
    resolved = []
    seen = set()
    for candidate_name, count, face_sign in VERTICAL_WIPE_SURFACE_CANDIDATES:
        for body in getattr(world, "bodies", []):
            body_name = _body_name(body)
            body_basename = _body_basename(body)
            if not body_name or not body_basename:
                continue
            if candidate_name not in (body_name, body_basename):
                continue
            if body_name in seen:
                continue
            resolved.append((body_name, count, face_sign))
            seen.add(body_name)
            break

    for body in getattr(world, "bodies", []):
        body_name = _body_name(body)
        if not body_name or body_name in seen:
            continue
        if not _is_auto_cabinet_front_candidate(body_name):
            continue
        resolved.append(
            (
                body_name,
                _vertical_wipe_target_count(body),
                AUTO_CABINET_FRONT_FACE_SIGN,
            )
        )
        seen.add(body_name)

    return resolved


def _downsample_wipe_targets_by_surface(placements, surface_plan, rng):
    placements_by_surface = {}
    for placement in placements:
        placements_by_surface.setdefault(placement["surface_name"], []).append(
            placement
        )

    selected = []
    selected_counts = {}
    for surface_name, surface_placements in placements_by_surface.items():
        keep_count = max(
            1,
            int(np.ceil(len(surface_placements) * WIPE_TARGET_KEEP_FRACTION)),
        )
        keep_count = min(keep_count, len(surface_placements))
        keep_indices = set(
            rng.choice(len(surface_placements), size=keep_count, replace=False).tolist()
        )
        kept = [
            placement
            for idx, placement in enumerate(surface_placements)
            if idx in keep_indices
        ]
        selected.extend(kept)
        selected_counts[surface_name] = len(kept)

    downsampled_plan = []
    for surface_name, area_m2, _target_count, placed_count in surface_plan:
        kept_count = selected_counts.get(surface_name, 0)
        if kept_count <= 0 and placed_count > 0:
            continue
        downsampled_plan.append((surface_name, area_m2, kept_count, kept_count))

    return selected, downsampled_plan


def _sample_random_bowl_layout(
    world,
    seed=None,
    robot_name=None,
    environment_name=None,
    spawn_bowls=True,
    container_kind="bowl",
    z_offset=0.05,
    scale_choices_override=None,
    base_radius_override=None,
    min_clearance_override=None,
    debug_disable_reachability=False,
):
    mesh_parts = (
        ("pycram_object_gap_demo", "pot_1.stl")
        if str(container_kind).lower() == "pot"
        else ("objects", "bowl.stl")
    )
    scale_choices = (
        np.array([0.8], dtype=float)
        if str(container_kind).lower() == "pot"
        else np.array([0.8, 1.0, 1.2, 1.4, 1.6], dtype=float)
    )
    object_color = (
        DEFAULT_POT_COLOR
        if str(container_kind).lower() == "pot"
        else DEFAULT_BOWL_COLOR
    )
    world, tuple_placements, surface_plan = _sample_random_surface_layout(
        world=world,
        seed=seed,
        environment_name=environment_name,
        object_label="bowl",
        object_name_prefix="bowl",
        mesh_parts=mesh_parts,
        object_color=object_color,
        scale_choices=(
            np.asarray(scale_choices_override, dtype=float)
            if scale_choices_override is not None
            else scale_choices
        ),
        base_radius=(
            float(base_radius_override)
            if base_radius_override is not None
            else _container_base_xy_radius(*mesh_parts)
        ),
        radius_safety_factor=BOWL_RADIUS_SAFETY_FACTOR,
        min_clearance_m=(
            float(min_clearance_override)
            if min_clearance_override is not None
            else MIN_BOWL_CLEARANCE_M
        ),
        strict_clean_mode=STRICT_CLEAN_MODE,
        z_offset=z_offset,
        reachability_fn=_is_pose_reachable_for_mixing,
        debug_disable_reachability=debug_disable_reachability,
        debug_force_spawn_all_targets=False,
        debug_spawn_actual_poses=False,
        spawn_objects=spawn_bowls,
        include_world_pose=not spawn_bowls,
    )
    placements = []
    for placement in tuple_placements:
        bowl_name, surface_name, scale, pose_xyz = placement[:4]
        world_pose = placement[4] if len(placement) > 4 else None
        placements.append(
            {
                "bowl_name": bowl_name,
                "surface_name": surface_name,
                "scale": scale,
                "pose_xyz": pose_xyz,
                "world_pose": world_pose,
            }
        )
    return world, placements, surface_plan


def setup_random_bowl_world(seed=None, robot_name=None, environment_name=None):
    return setup_random_mixing_container_world(
        seed=seed,
        robot_name=robot_name,
        environment_name=environment_name,
        container_kind="bowl",
    )


def setup_random_mixing_container_world(
    seed=None, robot_name=None, environment_name=None, container_kind="bowl"
):
    world, placements, surface_plan = _sample_random_bowl_layout(
        setup_thesis_world(robot_name=robot_name, environment_name=environment_name),
        seed=seed,
        robot_name=robot_name,
        environment_name=environment_name,
        spawn_bowls=True,
        container_kind=container_kind,
    )
    return (
        world,
        [
            (
                placement["bowl_name"],
                placement["surface_name"],
                placement["scale"],
                placement["pose_xyz"],
            )
            for placement in placements
        ],
        surface_plan,
    )


def sample_random_bowl_poses(seed=None, robot_name=None, environment_name=None):
    world, placements, surface_plan = _sample_random_bowl_layout(
        setup_thesis_world(robot_name=robot_name, environment_name=environment_name),
        seed=seed,
        robot_name=robot_name,
        environment_name=environment_name,
        spawn_bowls=False,
        z_offset=WIPE_TARGET_SURFACE_OFFSET_M,
        scale_choices_override=np.array([1.0], dtype=float),
        base_radius_override=WIPE_TARGET_RADIUS_M,
        min_clearance_override=WIPE_TARGET_CLEARANCE_M,
        debug_disable_reachability=True,
    )
    rng = np.random.default_rng(seed)
    placements, surface_plan = _downsample_wipe_targets_by_surface(
        placements, surface_plan, rng
    )

    renamed_placements = []
    for idx, placement in enumerate(placements, start=1):
        renamed = dict(placement)
        renamed["bowl_name"] = f"wipe_target_{idx:04d}"
        renamed_placements.append(renamed)

    extra_targets = []
    extra_surface_plan = []
    if (
        resolve_environment_name(environment_name)
        not in TABLE_ONLY_WIPE_TARGET_ENVIRONMENTS
    ):
        next_idx = len(renamed_placements) + 1
        for surface_name, count, face_sign in _resolve_vertical_wipe_surfaces(world):
            targets, plan_entries = _sample_vertical_wipe_targets(
                world,
                rng,
                surface_name,
                count,
                start_idx=next_idx,
                face_sign=face_sign,
                environment_name=environment_name,
            )
            extra_targets.extend(targets)
            extra_surface_plan.extend(plan_entries)
            next_idx += len(targets)

    return world, renamed_placements + extra_targets, surface_plan + extra_surface_plan
