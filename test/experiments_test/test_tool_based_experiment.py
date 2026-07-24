import json
import subprocess
from datetime import timedelta

import pytest
from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Bread
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

from experiments.tool_based_actions.experiment.configuration import (
    ExperimentConfiguration,
    TrialSpecification,
)
from experiments.tool_based_actions.experiment.exceptions import (
    IncompatibleResultRecord,
    InvalidSpawnRegion,
    MissingSpawnSurfaces,
    SpawnRegionExhausted,
)
from experiments.tool_based_actions.experiment.results import (
    FailureDescription,
    ResultRecorder,
    TargetResult,
    TaskReliability,
)
from experiments.tool_based_actions.experiment import run_experiment
from experiments.tool_based_actions.experiment.run_experiment import ExperimentRunner
from experiments.tool_based_actions.experiment.scene import (
    PlanarPose,
    SceneSpawner,
    SpawnRegion,
    annotate_spawn_surfaces,
)
from experiments.tool_based_actions.experiment.task_definitions import (
    CuttingTaskDefinition,
    MixingTaskDefinition,
    PouringTaskDefinition,
    WipingTaskDefinition,
    tasks_by_name,
)
from experiments.tool_based_actions.experiment.visualization import TargetHighlight

SURFACE_MINIMUM_X = 1.5
SURFACE_MAXIMUM_X = 2.5
SURFACE_MINIMUM_Y = 2.0
SURFACE_MAXIMUM_Y = 4.0
SURFACE_TOP_Z = 0.8


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


def _spawner(world: World, seed: int = 910001, **spawner_arguments) -> SceneSpawner:
    surfaces = annotate_spawn_surfaces(world, ["table_area_main", "does_not_exist"])
    return SceneSpawner(world=world, surfaces=surfaces, seed=seed, **spawner_arguments)


def test_spawn_region_rejects_bounds_without_points():
    with pytest.raises(InvalidSpawnRegion):
        SpawnRegion(
            minimum_x=2.0, maximum_x=1.0, minimum_y=0.0, maximum_y=1.0, height=1.0
        )


def test_spawn_region_contains_points_inside_its_bounds():
    region = SpawnRegion(
        minimum_x=0.0, maximum_x=1.0, minimum_y=0.0, maximum_y=1.0, height=1.0
    )
    assert region.contains(0.5, 0.5)
    assert not region.contains(1.5, 0.5)


def test_annotate_spawn_surfaces_computes_supporting_surface_regions():
    world = _world_with_surface_box("table_area_main")

    surfaces = annotate_spawn_surfaces(world, ["table_area_main", "does_not_exist"])

    assert len(surfaces) == 1
    assert surfaces[0].root.name.name == "table_area_main"
    assert surfaces[0].supporting_surface is not None


def test_annotate_spawn_surfaces_raises_without_any_match():
    world = _world_with_surface_box("shelf")
    with pytest.raises(MissingSpawnSurfaces):
        annotate_spawn_surfaces(world, ["island_countertop"])


def test_scene_spawner_is_deterministic_per_seed():
    first = _spawner(_world_with_surface_box("table_area_main")).spawn_targets(
        WipingTaskDefinition(), count=3, minimum_count=3, name_prefix="target"
    )
    second = _spawner(_world_with_surface_box("table_area_main")).spawn_targets(
        WipingTaskDefinition(), count=3, minimum_count=3, name_prefix="target"
    )
    assert [target.placement for target in first] == [
        target.placement for target in second
    ]

    other_seed = _spawner(
        _world_with_surface_box("table_area_main"), seed=910002
    ).spawn_targets(WipingTaskDefinition(), count=3, minimum_count=3, name_prefix="target")
    assert [target.placement for target in first] != [
        target.placement for target in other_seed
    ]


def test_scene_spawner_keeps_placements_on_the_surface_with_edge_clearance():
    edge_clearance = 0.2
    spawner = _spawner(
        _world_with_surface_box("table_area_main"), edge_clearance=edge_clearance
    )

    targets = spawner.spawn_targets(
        WipingTaskDefinition(), count=5, minimum_count=5, name_prefix="target"
    )

    for target in targets:
        placement = target.placement
        assert placement.surface_name == "table_area_main"
        assert SURFACE_MINIMUM_X + edge_clearance <= placement.pose.x
        assert placement.pose.x <= SURFACE_MAXIMUM_X - edge_clearance
        assert SURFACE_MINIMUM_Y + edge_clearance <= placement.pose.y
        assert placement.pose.y <= SURFACE_MAXIMUM_Y - edge_clearance
        assert placement.z == pytest.approx(SURFACE_TOP_Z, abs=0.01)
    assert [target.placement.name for target in targets] == [
        f"target_{index}" for index in range(5)
    ]


def test_target_count_follows_surface_area_density():
    spawner = _spawner(_world_with_surface_box("table_area_main"))
    surface_area = 1.0 * 2.0

    unclamped = spawner.target_count(
        targets_per_square_meter=12.0, minimum=2, maximum=30
    )
    assert unclamped == round(surface_area * 12.0)

    assert spawner.target_count(targets_per_square_meter=12.0, minimum=2, maximum=3) == 3
    assert spawner.target_count(targets_per_square_meter=0.1, minimum=2, maximum=30) == 2


def test_scene_spawner_places_mesh_targets_on_the_surface():
    world = _world_with_surface_box("table_area_main")
    spawner = _spawner(world, scale_choices=[1.4])

    targets = spawner.spawn_targets(
        CuttingTaskDefinition(), count=2, minimum_count=2, name_prefix="bread"
    )

    surface = spawner.surfaces[0]
    for target in targets:
        assert target.body is not None
        assert target.placement.scale == pytest.approx(1.4)
        assert target.placement.z > SURFACE_TOP_Z
        assert any(placed.root is target.body for placed in surface.objects)
        for shape in target.body.collision.shapes:
            assert shape.scale.x == pytest.approx(1.4)
    assert targets[0].placement.distance_to(targets[1].placement) > 0.0


def test_scene_spawner_raises_when_the_surfaces_are_full():
    world = _world_with_surface_box("table_area_main")
    spawner = _spawner(world)
    _cover_whole_surface(world, spawner)

    with pytest.raises(SpawnRegionExhausted):
        spawner.spawn_targets(
            WipingTaskDefinition(), count=1, minimum_count=1, name_prefix="target"
        )


def test_scene_spawner_removes_targets_it_cannot_place():
    world = _world_with_surface_box("table_area_main")
    spawner = _spawner(world)
    _cover_whole_surface(world, spawner)

    with pytest.raises(SpawnRegionExhausted):
        spawner.spawn_targets(
            CuttingTaskDefinition(), count=1, minimum_count=1, name_prefix="bread"
        )
    assert all(body.name.name != "bread_0" for body in world.bodies)


def _cover_whole_surface(world: World, spawner: SceneSpawner) -> None:
    """
    Register an object covering the whole spawn surface, so no free space remains.
    """
    cover_shapes = ShapeCollection([Box(scale=Scale(2.0, 4.0, 0.1))])
    cover = Body(name=PrefixedName("cover"), collision=cover_shapes, visual=cover_shapes)
    surface = spawner.surfaces[0]
    with world.modify_world():
        world.add_kinematic_structure_entity(cover)
        world.add_connection(
            FixedConnection(
                parent=world.root,
                child=cover,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    2.0, 3.0, 0.85, reference_frame=world.root
                ),
            )
        )
        cover_annotation = Bread(root=cover)
        world.add_semantic_annotations([cover_annotation])
        surface.add_object(cover_annotation)


def test_tasks_by_name_lists_every_task_definition():
    assert tasks_by_name() == {
        "cutting": CuttingTaskDefinition,
        "mixing": MixingTaskDefinition,
        "pouring": PouringTaskDefinition,
        "wiping": WipingTaskDefinition,
    }


def test_task_definitions_forward_the_pointer_stride():
    for task in tasks_by_name().values():
        definition = task(pointer_stride=7)
        assert definition.pointer_stride == 7


def test_cutting_actions_carry_the_definition_pointer_stride():
    world = _world_with_surface_box("table_area_main")
    definition = CuttingTaskDefinition(pointer_stride=7)
    target = _spawner(world).spawn_targets(
        definition, count=1, minimum_count=1, name_prefix="bread"
    )[0]
    tool = Bread(root=world.get_body_by_name("table_area_main"))

    action = definition.build_action(target, tool)

    assert action.pointer_stride == 7
    assert action.object_to_cut is target.body


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


def test_trial_grid_is_the_task_seed_product():
    configuration = ExperimentConfiguration(
        tasks=[CuttingTaskDefinition, WipingTaskDefinition],
        seeds=[1, 2, 3],
    )
    specifications = configuration.build_trial_specifications()

    assert len(specifications) == 6
    assert {specification.task for specification in specifications} == {
        CuttingTaskDefinition,
        WipingTaskDefinition,
    }
    assert len({specification.identifier for specification in specifications}) == 6


def test_default_configuration_runs_every_task():
    assert set(ExperimentConfiguration().tasks) == set(tasks_by_name().values())


def test_configuration_json_roundtrip():
    configuration = ExperimentConfiguration(
        tasks=[MixingTaskDefinition],
        seeds=[7],
        surface_names=["island_countertop"],
        scale_choices=[0.9, 1.1],
        surface_edge_clearance=0.2,
        full_body_motion=False,
        tool_path_pointer_stride=5,
        collision_avoidance=False,
        trial_timeout=timedelta(minutes=45),
    )
    assert from_json(to_json(configuration)) == configuration


def _result(trial_identifier: str, target_name: str, success: bool) -> TargetResult:
    return TargetResult(
        trial_identifier=trial_identifier,
        task=CuttingTaskDefinition,
        seed=1,
        robot_name="pr2",
        environment_name="apartment",
        target_name=target_name,
        target_pose=PlanarPose(x=2.4, y=2.2, yaw=1.57),
        target_scale=1.2,
        surface_name="island_countertop",
        duration=timedelta(seconds=1.5),
        failure=(
            None
            if success
            else FailureDescription(
                exception_type=RuntimeError, message="goal not reached"
            )
        ),
    )


def test_result_recorder_roundtrip_and_resume(tmp_path):
    recorder = ResultRecorder(results_file=tmp_path / "results.jsonl")
    recorder.record(_result("cutting:apartment:1", "cutting_1_0", True))
    recorder.record(_result("cutting:apartment:1", "cutting_1_1", False))

    results = recorder.load_results()
    assert len(results) == 2
    assert results[0].success is True
    assert results[0].surface_name == "island_countertop"
    assert results[0].target_scale == pytest.approx(1.2)
    assert results[0].target_pose == PlanarPose(x=2.4, y=2.2, yaw=1.57)
    assert results[0].duration == timedelta(seconds=1.5)
    assert results[1].success is False
    assert results[1].failure == FailureDescription(
        exception_type=RuntimeError, message="goal not reached"
    )

    completed_specification = TrialSpecification(
        task=CuttingTaskDefinition,
        seed=1,
        environment_name="apartment",
    )
    pending_specification = TrialSpecification(
        task=CuttingTaskDefinition,
        seed=2,
        environment_name="apartment",
    )
    assert recorder.is_completed(completed_specification)
    assert not recorder.is_completed(pending_specification)


def test_failure_description_captures_type_and_message():
    failure = FailureDescription.from_exception(ValueError("bad pose"))
    assert failure.exception_type is ValueError
    assert failure.message == "bad pose"


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
    assert "target_pose" in str(error_information.value)
    assert "target_x" in str(error_information.value)


def test_task_reliability_aggregates_results_per_task():
    results = [
        _result("cutting:apartment:pr2:1", "cutting_1_0", True),
        _result("cutting:apartment:pr2:1", "cutting_1_1", False),
        _result("cutting:apartment:pr2:2", "cutting_2_0", True),
    ]

    rows = TaskReliability.from_results(results)

    assert rows == [TaskReliability(task="cutting", successes=2, total=3)]
    assert TaskReliability.get_column_names() == ["task", "successes", "total"]
    assert rows[0].get_column_values() == ["cutting", 2, 3]


def test_result_recorder_is_empty_without_file(tmp_path):
    recorder = ResultRecorder(results_file=tmp_path / "missing.jsonl")
    assert recorder.load_results() == []
    assert recorder.completed_trial_identifiers() == set()


def _campaign_configuration(tmp_path) -> ExperimentConfiguration:
    return ExperimentConfiguration(
        tasks=[CuttingTaskDefinition],
        seeds=[1, 2],
        results_file=tmp_path / "results.jsonl",
    )


def test_experiment_runner_skips_completed_trials_and_builds_trial_commands(
    tmp_path, monkeypatch
):
    configuration = _campaign_configuration(tmp_path)
    ResultRecorder(configuration.results_file).record(
        _result("cutting:apartment:1", "cutting_1_0", True)
    )
    executed_commands = []

    def record_command(command, timeout):
        executed_commands.append(command)
        return subprocess.CompletedProcess(command, returncode=0)

    monkeypatch.setattr(subprocess, "run", record_command)
    ExperimentRunner(configuration=configuration).run()

    assert len(executed_commands) == 1
    command = executed_commands[0]
    assert command[1:3] == [
        "-m",
        "experiments.tool_based_actions.experiment.single_trial",
    ]
    assert command[command.index("--task") + 1] == "cutting"
    assert command[command.index("--seed") + 1] == "2"
    assert command[command.index("--configuration-json") + 1] == json.dumps(
        to_json(configuration)
    )


def test_experiment_runner_survives_timeouts_and_failing_trials(tmp_path, monkeypatch):
    configuration = _campaign_configuration(tmp_path)
    outcomes = iter(
        [
            subprocess.TimeoutExpired(cmd="trial", timeout=1.0),
            subprocess.CompletedProcess([], returncode=3),
        ]
    )
    attempted_commands = []

    def flaky_run(command, timeout):
        attempted_commands.append(command)
        outcome = next(outcomes)
        if isinstance(outcome, subprocess.TimeoutExpired):
            raise outcome
        return outcome

    monkeypatch.setattr(subprocess, "run", flaky_run)
    ExperimentRunner(configuration=configuration).run()

    assert len(attempted_commands) == 2


def test_experiment_runner_summarizes_recorded_results(tmp_path, monkeypatch):
    configuration = _campaign_configuration(tmp_path)
    recorder = ResultRecorder(configuration.results_file)
    recorder.record(_result("cutting:apartment:1", "cutting_1_0", True))
    recorder.record(_result("cutting:apartment:2", "cutting_2_0", False))

    def unexpected_run(command, timeout):
        raise AssertionError("No trial should run when all trials are completed.")

    monkeypatch.setattr(subprocess, "run", unexpected_run)
    logged_messages = []
    monkeypatch.setattr(
        run_experiment.logger,
        "info",
        lambda message, *arguments: logged_messages.append(message % arguments),
    )
    ExperimentRunner(configuration=configuration).run()

    assert "cutting: 1/2 targets succeeded" in logged_messages


def test_parse_configuration_applies_command_line_overrides(tmp_path):
    results_file = tmp_path / "results.jsonl"

    configuration = run_experiment.parse_configuration(
        [
            "--tasks",
            "cutting",
            "wiping",
            "--seeds",
            "5",
            "7",
            "--results-file",
            str(results_file),
            "--trial-timeout-seconds",
            "120",
        ]
    )

    assert configuration.tasks == [CuttingTaskDefinition, WipingTaskDefinition]
    assert configuration.seeds == [5, 7]
    assert configuration.results_file == results_file
    assert configuration.trial_timeout == timedelta(seconds=120)


def test_parse_configuration_keeps_the_defaults_without_arguments():
    assert run_experiment.parse_configuration([]) == ExperimentConfiguration()
