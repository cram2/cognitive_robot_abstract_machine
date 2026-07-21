# Tool-Based Action Experiment

A reproducible simulation campaign that measures how reliably the PR2 performs
tool-based actions (cutting, mixing, pouring, wiping) on randomly generated scenes
in the apartment environment.

An experiment is a grid of trials: every configured task runs once per seed. Each
trial spawns a seeded random scene, acts on every spawned target, and records one
result per target. Rerunning a trial specification reproduces the exact same scene.

## Running

```bash
python -m experiments.tool_based_actions.experiment.run_experiment
```

There are no command line arguments; everything is driven by
`ExperimentConfiguration` in `configuration.py`. Every trial runs in its own
subprocess with a wall-clock timeout, so a crashing or hanging simulation never
takes the campaign down. Trials that already have recorded results are skipped, so
a killed campaign resumes where it left off.

A single trial can also be run directly:

```bash
python -m experiments.tool_based_actions.experiment.single_trial \
    --task cutting --seed 910001 --configuration-json "$(python -c '
from experiments.tool_based_actions.experiment.configuration import ExperimentConfiguration
print(ExperimentConfiguration().to_json())')"
```

## Modules

| Module | Responsibility |
|---|---|
| `configuration.py` | `ExperimentConfiguration`: the full campaign configuration, JSON-serializable for trial subprocesses. |
| `run_experiment.py` | Campaign orchestrator: runs the trial grid in isolated subprocesses, skips completed trials, prints a summary. |
| `single_trial.py` | One trial: spawn the seeded scene, act on every target, record results. |
| `scene.py` | Seeded random scene sampling: surfaces, obstacles, footprint-aware placements. |
| `task_definitions.py` | Per-task scene and action construction (tool, target mesh, tool action). |
| `results.py` | JSON-lines result recording and resume queries. |
| `visualization.py` | RViz publishing and the blue highlight of the currently approached target. |

## Scene generation

Targets are spawned on the support surfaces named in the configuration
(`island_countertop`, `countertop`, `table_area_main` by default). The sampler is
seeded — the same seed always yields the same scene — and enforces:

- **Density-based counts:** the desired target count is
  `usable surface area × targets_per_square_meter`, clamped to
  `[minimum_targets_per_trial, maximum_targets_per_trial]`. Placement is
  best-effort: if the surfaces cannot hold the desired count, the trial keeps what
  fits, as long as at least the minimum is placed.
- **Size variation:** every target gets a uniform scale drawn from
  `scale_choices`; the mesh is spawned at that scale and the scale is recorded
  with the result.
- **Footprint awareness:** the target's footprint radius is measured from its
  mesh and inflated by `footprint_safety_factor`. Placements keep their whole
  footprint on the surface, and two targets keep
  `max(target_clearance, radius_a + radius_b + footprint_clearance)` apart.
- **Obstacle avoidance:** placements keep clear of the bounding box of every
  other collidable body in the world (robot excluded), so nothing spawns inside
  sinks, faucets, or furniture. Surfaces above `maximum_spawn_height` are
  excluded.

## Trial execution

Per trial, the robot is first brought into its parked posture **without**
collision avoidance, because it may spawn with its outstretched arms in contact
with the environment. Every target is then processed with:

1. Close gripper, park arms, torso high.
2. Navigate to a base pose around the target. The base pose is **underspecified**:
   it is a free variable whose domain is a costmap location (ring at arm's reach ∧
   collision-free occupancy) resolved at runtime. The tool actions themselves are
   fully specified.
3. Perform the tool action.

While a target is processed it is dyed blue in RViz and restored afterwards
(`TargetHighlight`).

Two execution behaviors are configurable:

- `full_body_motion` (default True): the base drives along during tool motions
  instead of staying fixed at the navigated pose, so targets at the edge of reach
  stay achievable.
- `collision_avoidance` (default True): motions avoid collisions with the
  environment. The tool motions exempt the acting arm's manipulator **including
  the mounted tool**, so the tool can touch its target. A real collision of any
  other body part aborts the action and is recorded as a failure
  (`CollisionViolatedError`).

## Results

Results are appended to `records/tool_based_experiment_results.jsonl`, one JSON
object per target:

```json
{"trial_identifier": "cutting:apartment:pr2:910001", "task": "cutting",
 "seed": 910001, "robot_name": "pr2", "environment_name": "apartment",
 "target_name": "cutting_910001_0", "target_x": 5.0, "target_y": 3.7,
 "target_yaw": 1.2, "target_scale": 1.4, "surface_name": "table_area_main",
 "success": true, "duration": 26.4, "failure_reason": null}
```

`failure_reason` holds a compact description (exception type and message) when the
action failed. A trial counts as completed for resume purposes as soon as it has
at least one recorded result. Records from an older schema make loading fail with
`IncompatibleResultRecord`; archive or delete the results file to start a fresh
campaign.

## Visualization

`single_trial` publishes the world, its tf tree, and closest-point collision
results to RViz (topic `/semworld/viz_marker`). In RViz, add a MarkerArray plugin
on that topic, set the durability policy to `TRANSIENT_LOCAL`, and use the tf
root as fixed frame.

## Key configuration values

| Field | Default | Meaning |
|---|---|---|
| `tasks` | all four | Tasks in the grid. |
| `seeds` | `(910001, 910002, 910003)` | Seeds per task. |
| `targets_per_square_meter` | `12.0` | Target density on the surfaces. |
| `minimum/maximum_targets_per_trial` | `2` / `30` | Clamp on the density-based count. |
| `scale_choices` | `(0.8, 1.0, 1.2, 1.4, 1.6)` | Uniform target scales. |
| `target_clearance` | `0.35` | Minimum center distance between targets. |
| `footprint_clearance` | `0.03` | Minimum free gap between target footprints. |
| `maximum_spawn_height` | `1.35` | Highest usable surface top. |
| `full_body_motion` | `True` | Base drives along during tool motions. |
| `collision_avoidance` | `True` | Avoid collisions, tool and gripper exempt. |
| `tool_path_pointer_stride` | `10` | Keep every Nth tool path waypoint. |
| `trial_timeout` | `3600.0` | Wall-clock limit per trial process. |

Cutting parameters (technique, number of cuts, slice thickness) are set in
`CuttingTaskDefinition.build_action` in `task_definitions.py`; the analogous
parameters of the other tasks live in their respective definitions.
