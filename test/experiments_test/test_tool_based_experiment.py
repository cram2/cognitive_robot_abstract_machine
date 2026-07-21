import json
import math

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    CuttingKnife,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from experiments.tool_based_actions.experiment.configuration import (
    ExperimentConfiguration,
    SpawnRegion,
    ToolBasedTask,
    TrialSpecification,
)
from experiments.tool_based_actions.experiment.results import (
    IncompatibleResultRecord,
    ResultRecorder,
    TargetResult,
)
from experiments.tool_based_actions.experiment.scene import (
    MissingSpawnSurfaces,
    ObjectFootprint,
    ObstacleBox,
    SceneSampler,
    SpawnRegionExhausted,
    SpawnSurface,
    TargetPlacement,
    discover_obstacles,
    discover_spawn_surfaces,
)
from experiments.tool_based_actions.experiment.task_definitions import (
    CuttingTaskDefinition,
    definition_for_task,
)
from experiments.tool_based_actions.experiment.visualization import TargetHighlight

COUNTER = SpawnSurface(
    name="counter",
    region=SpawnRegion(
        minimum_x=2.35, maximum_x=2.55, minimum_y=2.1, maximum_y=3.2, height=1.0
    ),
)
TABLE = SpawnSurface(
    name="table",
    region=SpawnRegion(
        minimum_x=4.7, maximum_x=5.3, minimum_y=3.3, maximum_y=4.7, height=0.75
    ),
)


def _sampler(seed: int = 910001, clearance: float = 0.35) -> SceneSampler:
    return SceneSampler(surfaces=[COUNTER, TABLE], clearance=clearance, seed=seed)


def test_scene_sampler_is_deterministic_per_seed():
    first = _sampler().sample_placements(3, name_prefix="target")
    second = _sampler().sample_placements(3, name_prefix="target")
    assert first == second

    other_seed = _sampler(seed=910002).sample_placements(3, name_prefix="target")
    assert first != other_seed


def test_scene_sampler_respects_surfaces_clearance_and_yaw_range():
    placements = _sampler().sample_placements(3, name_prefix="target")

    surfaces_by_name = {surface.name: surface for surface in [COUNTER, TABLE]}
    for placement in placements:
        surface = surfaces_by_name[placement.surface_name]
        assert surface.region.contains(placement.x, placement.y)
        assert placement.z == surface.region.height
        assert 0.0 <= placement.yaw < 2.0 * math.pi
    for first_index in range(len(placements)):
        for second_index in range(first_index + 1, len(placements)):
            assert placements[first_index].distance_to(placements[second_index]) >= 0.35
    assert [placement.name for placement in placements] == [
        "target_0",
        "target_1",
        "target_2",
    ]


def test_scene_sampler_spreads_targets_over_multiple_surfaces():
    used_surfaces = set()
    for seed in range(1, 21):
        for placement in _sampler(seed=seed).sample_placements(3, name_prefix="target"):
            used_surfaces.add(placement.surface_name)
    assert used_surfaces == {"counter", "table"}


def test_target_count_follows_surface_area_density():
    sampler = _sampler()
    total_area = 0.2 * 1.1 + 0.6 * 1.4

    unclamped = sampler.target_count(
        targets_per_square_meter=12.0, minimum=2, maximum=30
    )
    assert unclamped == round(total_area * 12.0)

    assert sampler.target_count(targets_per_square_meter=12.0, minimum=2, maximum=3) == 3
    assert sampler.target_count(targets_per_square_meter=0.1, minimum=2, maximum=30) == 2


def test_object_footprint_scales_base_radius():
    footprint = ObjectFootprint(
        base_radius=0.1, scale_choices=(0.8, 1.6), safety_factor=1.08
    )
    assert footprint.radius_for_scale(1.6) == pytest.approx(0.1 * 1.6 * 1.08)
    assert footprint.largest_radius() == pytest.approx(0.1 * 1.6 * 1.08)


def test_scene_sampler_keeps_scaled_footprints_inside_surface_bounds():
    footprint = ObjectFootprint(
        base_radius=0.1, scale_choices=(1.0, 1.6), safety_factor=1.0
    )
    sampler = SceneSampler(
        surfaces=[TABLE], clearance=0.0, seed=910001, footprint=footprint
    )

    placements = sampler.sample_placements(3, name_prefix="target")

    for placement in placements:
        assert placement.scale in (1.0, 1.6)
        assert placement.footprint_radius == pytest.approx(0.1 * placement.scale)
        assert TABLE.region.minimum_x + placement.footprint_radius <= placement.x
        assert placement.x <= TABLE.region.maximum_x - placement.footprint_radius
        assert TABLE.region.minimum_y + placement.footprint_radius <= placement.y
        assert placement.y <= TABLE.region.maximum_y - placement.footprint_radius


def test_scene_sampler_places_best_effort_down_to_the_minimum_count():
    footprint = ObjectFootprint(
        base_radius=0.25, scale_choices=(1.0,), safety_factor=1.0
    )
    sampler = SceneSampler(
        surfaces=[TABLE], clearance=0.35, seed=910001, footprint=footprint
    )

    placements = sampler.sample_placements(10, name_prefix="target", minimum_count=2)
    assert 2 <= len(placements) < 10

    with pytest.raises(SpawnRegionExhausted):
        sampler.sample_placements(10, name_prefix="target", minimum_count=10)


def test_scene_sampler_keeps_footprint_clearance_between_large_targets():
    footprint = ObjectFootprint(
        base_radius=0.25, scale_choices=(1.0,), safety_factor=1.0
    )
    sampler = SceneSampler(
        surfaces=[TABLE],
        clearance=0.35,
        seed=910001,
        footprint=footprint,
        footprint_clearance=0.05,
    )

    placements = sampler.sample_placements(2, name_prefix="target")

    assert placements[0].distance_to(placements[1]) >= 0.25 + 0.25 + 0.05


def _table_covering_obstacle(minimum_z: float, maximum_z: float) -> ObstacleBox:
    return ObstacleBox(
        name="blocking_box",
        minimum_x=TABLE.region.minimum_x,
        maximum_x=TABLE.region.maximum_x,
        minimum_y=TABLE.region.minimum_y,
        maximum_y=TABLE.region.maximum_y,
        minimum_z=minimum_z,
        maximum_z=maximum_z,
    )


def test_scene_sampler_rejects_placements_inside_obstacle_bands():
    sampler = SceneSampler(
        surfaces=[TABLE],
        clearance=0.35,
        seed=910001,
        obstacles=[_table_covering_obstacle(minimum_z=0.5, maximum_z=1.2)],
    )
    with pytest.raises(SpawnRegionExhausted):
        sampler.sample_placements(2, name_prefix="target")


def test_scene_sampler_ignores_obstacles_below_the_spawn_height():
    sampler = SceneSampler(
        surfaces=[TABLE],
        clearance=0.35,
        seed=910001,
        obstacles=[_table_covering_obstacle(minimum_z=0.2, maximum_z=0.7)],
    )
    assert len(sampler.sample_placements(2, name_prefix="target")) == 2


def test_scene_sampler_avoids_partial_obstacles():
    obstacle = ObstacleBox(
        name="tray",
        minimum_x=TABLE.region.minimum_x,
        maximum_x=TABLE.region.maximum_x,
        minimum_y=TABLE.region.minimum_y,
        maximum_y=4.0,
        minimum_z=0.5,
        maximum_z=1.2,
    )
    sampler = SceneSampler(
        surfaces=[TABLE], clearance=0.35, seed=910001, obstacles=[obstacle]
    )

    placements = sampler.sample_placements(2, name_prefix="target")

    for placement in placements:
        assert placement.y > 4.0


HIGH_SHELF = SpawnSurface(
    name="high_shelf",
    region=SpawnRegion(
        minimum_x=4.7, maximum_x=5.3, minimum_y=3.3, maximum_y=4.7, height=1.6
    ),
)


def test_scene_sampler_excludes_surfaces_above_the_maximum_spawn_height():
    sampler = SceneSampler(
        surfaces=[TABLE, HIGH_SHELF],
        clearance=0.35,
        seed=910001,
        maximum_spawn_height=1.35,
    )

    placements = sampler.sample_placements(3, name_prefix="target")
    assert {placement.surface_name for placement in placements} == {"table"}

    shelf_only_sampler = SceneSampler(
        surfaces=[HIGH_SHELF], clearance=0.35, seed=910001, maximum_spawn_height=1.35
    )
    with pytest.raises(SpawnRegionExhausted):
        shelf_only_sampler.sample_placements(1, name_prefix="target")


def test_scene_sampler_fails_fast_when_surfaces_cannot_fit_targets():
    with pytest.raises(SpawnRegionExhausted):
        _sampler(clearance=10.0).sample_placements(3, name_prefix="target")


def test_scene_sampler_recovers_from_dead_end_placements():
    tight = SceneSampler(surfaces=[COUNTER], clearance=0.5, seed=1)
    for seed in range(1, 51):
        tight.seed = seed
        placements = tight.sample_placements(2, name_prefix="target")
        assert placements[0].distance_to(placements[1]) >= 0.5


def _world_with_surface_box(name: str) -> World:
    world = World()
    shape_collection = ShapeCollection([Box(scale=Scale(1.0, 2.0, 0.1))])
    body = Body(
        name=PrefixedName(name), collision=shape_collection, visual=shape_collection
    )
    root = Body(name=PrefixedName("world_root"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(body)
        world.add_connection(
            FixedConnection(
                parent=root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    2.0, 3.0, 0.75, reference_frame=root
                ),
            )
        )
    return world


def test_discover_spawn_surfaces_measures_named_bodies():
    world = _world_with_surface_box("table_area_main")
    surfaces = discover_spawn_surfaces(
        world,
        surface_names=("table_area_main", "does_not_exist"),
        margin=0.1,
        height_offset=0.05,
    )

    assert len(surfaces) == 1
    region = surfaces[0].region
    assert region.minimum_x == pytest.approx(1.6)
    assert region.maximum_x == pytest.approx(2.4)
    assert region.minimum_y == pytest.approx(2.1)
    assert region.maximum_y == pytest.approx(3.9)
    assert region.height == pytest.approx(0.85)


def test_discover_spawn_surfaces_raises_without_any_match():
    world = _world_with_surface_box("shelf")
    with pytest.raises(MissingSpawnSurfaces):
        discover_spawn_surfaces(
            world, surface_names=("island_countertop",), margin=0.1, height_offset=0.05
        )


def test_discover_obstacles_measures_collision_bodies_and_skips_excluded_names():
    world = _world_with_surface_box("side_table")

    obstacles = discover_obstacles(world, excluded_body_names=set())

    assert [obstacle.name for obstacle in obstacles] == ["side_table"]
    obstacle = obstacles[0]
    assert obstacle.minimum_x == pytest.approx(1.5)
    assert obstacle.maximum_x == pytest.approx(2.5)
    assert obstacle.minimum_y == pytest.approx(2.0)
    assert obstacle.maximum_y == pytest.approx(4.0)
    assert obstacle.minimum_z == pytest.approx(0.7)
    assert obstacle.maximum_z == pytest.approx(0.8)

    assert discover_obstacles(world, excluded_body_names={"side_table"}) == []


def test_obstacle_box_blocks_only_within_its_vertical_band():
    obstacle = ObstacleBox(
        name="crate",
        minimum_x=0.0,
        maximum_x=1.0,
        minimum_y=0.0,
        maximum_y=1.0,
        minimum_z=0.5,
        maximum_z=1.0,
    )

    assert obstacle.blocks(x=0.5, y=0.5, z=0.75, radius=0.0)
    assert obstacle.blocks(x=1.05, y=0.5, z=0.75, radius=0.1)
    assert not obstacle.blocks(x=1.2, y=0.5, z=0.75, radius=0.1)
    assert not obstacle.blocks(x=0.5, y=0.5, z=1.05, radius=0.0)
    assert not obstacle.blocks(x=0.5, y=0.5, z=0.4, radius=0.0)


def test_cutting_targets_have_a_mesh_footprint():
    footprint = CuttingTaskDefinition().target_footprint(
        scale_choices=(0.8, 1.0), safety_factor=1.08
    )
    assert footprint.base_radius > 0.02
    assert footprint.scale_choices == (0.8, 1.0)
    assert footprint.safety_factor == pytest.approx(1.08)


def test_task_definitions_forward_the_pointer_stride():
    for task in ToolBasedTask:
        definition = definition_for_task(task, pointer_stride=7)
        assert definition.pointer_stride == 7


def test_cutting_actions_carry_the_definition_pointer_stride():
    world = _world_with_surface_box("counter")
    definition = CuttingTaskDefinition(pointer_stride=7)
    placement = TargetPlacement(
        name="bread_target",
        surface_name="counter",
        x=2.0,
        y=3.0,
        z=0.9,
        yaw=0.0,
        scale=1.0,
        footprint_radius=0.1,
    )
    target = definition.spawn_target(world, placement)
    tool = CuttingKnife(root=world.get_body_by_name("counter"))

    action = definition.build_action(target, tool)

    assert action.pointer_stride == 7


def test_target_highlight_dyes_and_restores_the_target_colors():
    world = _world_with_surface_box("counter")
    body = world.get_body_by_name("counter")
    original_colors = [shape.color for shape in body.visual.shapes]

    highlight = TargetHighlight(world=world, body=body)
    with highlight:
        for shape in body.visual.shapes:
            assert shape.color == highlight.color
    assert [shape.color for shape in body.visual.shapes] == original_colors


def test_target_highlight_restores_colors_when_the_action_fails():
    world = _world_with_surface_box("counter")
    body = world.get_body_by_name("counter")
    original_colors = [shape.color for shape in body.visual.shapes]

    with pytest.raises(RuntimeError):
        with TargetHighlight(world=world, body=body):
            raise RuntimeError("action failed")
    assert [shape.color for shape in body.visual.shapes] == original_colors


def test_target_highlight_ignores_pose_only_targets():
    world = _world_with_surface_box("counter")
    with TargetHighlight(world=world, body=None):
        pass


def test_spawned_targets_carry_the_placement_scale():
    world = _world_with_surface_box("counter")
    placement = TargetPlacement(
        name="bread_target",
        surface_name="counter",
        x=2.0,
        y=3.0,
        z=0.9,
        yaw=0.0,
        scale=1.4,
        footprint_radius=0.1,
    )

    target = CuttingTaskDefinition().spawn_target(world, placement)

    for shape in target.body.collision.shapes:
        assert shape.scale.x == pytest.approx(1.4)
        assert shape.scale.y == pytest.approx(1.4)
        assert shape.scale.z == pytest.approx(1.4)


def test_trial_grid_is_the_task_seed_product():
    configuration = ExperimentConfiguration(
        tasks=(ToolBasedTask.CUTTING, ToolBasedTask.WIPING),
        seeds=(1, 2, 3),
    )
    specifications = configuration.build_trial_specifications()

    assert len(specifications) == 6
    assert {specification.task for specification in specifications} == {
        ToolBasedTask.CUTTING,
        ToolBasedTask.WIPING,
    }
    assert len({specification.identifier for specification in specifications}) == 6


def test_configuration_json_roundtrip():
    configuration = ExperimentConfiguration(
        tasks=(ToolBasedTask.MIXING,),
        seeds=(7,),
        surface_names=("island_countertop",),
        scale_choices=(0.9, 1.1),
        full_body_motion=False,
        tool_path_pointer_stride=5,
        collision_avoidance=False,
    )
    assert ExperimentConfiguration.from_json(configuration.to_json()) == configuration


def _result(trial_identifier: str, target_name: str, success: bool) -> TargetResult:
    return TargetResult(
        trial_identifier=trial_identifier,
        task=ToolBasedTask.CUTTING,
        seed=1,
        robot_name="pr2",
        environment_name="apartment",
        target_name=target_name,
        target_x=2.4,
        target_y=2.2,
        target_yaw=1.57,
        target_scale=1.2,
        surface_name="island_countertop",
        success=success,
        duration=1.5,
        failure_reason=None if success else "MotionDidNotFinish: goal not reached",
    )


def test_result_recorder_roundtrip_and_resume(tmp_path):
    recorder = ResultRecorder(results_file=tmp_path / "results.jsonl")
    recorder.record(_result("cutting:apartment:pr2:1", "cutting_1_0", True))
    recorder.record(_result("cutting:apartment:pr2:1", "cutting_1_1", False))

    results = recorder.load_results()
    assert len(results) == 2
    assert results[0].success is True
    assert results[0].surface_name == "island_countertop"
    assert results[0].target_scale == pytest.approx(1.2)
    assert results[1].failure_reason.startswith("MotionDidNotFinish")

    completed_specification = TrialSpecification(
        task=ToolBasedTask.CUTTING,
        seed=1,
        robot_name="pr2",
        environment_name="apartment",
    )
    pending_specification = TrialSpecification(
        task=ToolBasedTask.CUTTING,
        seed=2,
        robot_name="pr2",
        environment_name="apartment",
    )
    assert recorder.is_completed(completed_specification)
    assert not recorder.is_completed(pending_specification)


def test_result_recorder_rejects_records_from_an_older_schema(tmp_path):
    legacy_record = {
        "trial_identifier": "cutting:apartment:pr2:910001",
        "task": "cutting",
        "seed": 910001,
        "robot_name": "pr2",
        "environment_name": "apartment",
        "target_name": "cutting_910001_0",
        "target_x": 2.4,
        "target_y": 2.1,
        "success": True,
        "duration": 9.7,
        "failure_reason": None,
    }
    results_file = tmp_path / "results.jsonl"
    results_file.write_text(json.dumps(legacy_record) + "\n", encoding="utf-8")
    recorder = ResultRecorder(results_file=results_file)

    with pytest.raises(IncompatibleResultRecord) as error_information:
        recorder.load_results()
    assert "target_yaw" in str(error_information.value)
    assert "surface_name" in str(error_information.value)


def test_result_recorder_is_empty_without_file(tmp_path):
    recorder = ResultRecorder(results_file=tmp_path / "missing.jsonl")
    assert recorder.load_results() == []
    assert recorder.completed_trial_identifiers() == set()
