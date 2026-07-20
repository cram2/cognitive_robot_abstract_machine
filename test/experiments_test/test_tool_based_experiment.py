import math

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
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
    ResultRecorder,
    TargetResult,
)
from experiments.tool_based_actions.experiment.scene import (
    MissingSpawnSurfaces,
    SceneSampler,
    SpawnRegionExhausted,
    SpawnSurface,
    discover_spawn_surfaces,
)

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


def test_scene_sampler_target_count_is_reproducible_and_bounded():
    counts = {_sampler().sample_target_count(2, 3) for _ in range(5)}
    assert len(counts) == 1
    assert counts.pop() in (2, 3)


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
        tasks=(ToolBasedTask.MIXING,), seeds=(7,), surface_names=("island_countertop",)
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


def test_result_recorder_is_empty_without_file(tmp_path):
    recorder = ResultRecorder(results_file=tmp_path / "missing.jsonl")
    assert recorder.load_results() == []
    assert recorder.completed_trial_identifiers() == set()
