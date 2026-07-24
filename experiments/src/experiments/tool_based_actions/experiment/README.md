# Tool-Based Action Experiment

A reproducible simulation campaign that measures how reliably the PR2 performs
tool-based actions (cutting, mixing, pouring, wiping) on randomly generated scenes
in the apartment environment.

An experiment is a grid of trials: every configured task runs once per seed. Each
trial runs in its own subprocess with a wall-clock timeout, spawns a seeded random
scene, acts on every spawned target, and appends one result per target to the
results file. Trials that already have recorded results are skipped, so a killed
campaign resumes where it left off, and rerunning a trial specification reproduces
the exact same scene.

## Running

```bash
python -m experiments.tool_based_actions.experiment.run_experiment --help
```

All defaults live in `ExperimentConfiguration` in `configuration.py`; the command
line only overrides them.

## Visualization

`single_trial` publishes the world, its tf tree, and closest-point collision
results to RViz (topic `/semworld/viz_marker`). In RViz, add a MarkerArray plugin
on that topic, set the durability policy to `TRANSIENT_LOCAL`, and use the tf
root as fixed frame. The currently approached target is dyed blue.
