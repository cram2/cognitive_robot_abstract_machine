# Trial workflow: notes and requirements

Working notes for the branch `trial-workflow` (draft PR cram2#1040, "Find out by trials
what makes a behaviour succeed"). Read this first in a new session. Delete this file
before the pull request is marked ready for review.

## The process this formalizes

Working on RoboCasa tasks with the PandaOmron in MuJoCo, the same loop kept repeating:

1. We do a task step by step until we hit a wall: a step fails and no single fix
   explains why.
2. We reach the state where the failure shows once, check that it really is that state,
   and store it as a starting world through ormatic, so every trial starts from exactly
   it.
3. We declare what the behaviour may vary by as a dataclass of numbers, bound it, and let
   a probabilistic model draw parameters (EQL statement, `ProbabilisticBackend`, a
   `ModelRegistry`): first uniformly, later from a Gaussian mixture around the
   parameters that worked.
4. Many processes try the draws from the starting world at once (48 workers on neem-4),
   and every trial is stored as an episode through ormatic: parameters, measured result,
   the stage that failed, and the recorded plans and simulated moments.
5. We analyse which parameters change the success rate, adopt the winning values into
   the action's defaults, and rerun the whole task.

Two studies on the branch `tomsch420:robocasa-experiments` followed this loop by hand:

- Picking a broccoli up from a sink (`experiments.grasp_learning.pick_up_trials`): run 1
  drew uniformly, 2 of 937 trials lifted; run 2 drew from a Gaussian mixture around
  those two, 10 of 959 lifted. From RoboCasa's real spawn, 45 of 972 lifted; a check of
  the causal question showed the distance to the nearest centre predicts lifting, the
  base pose error does not.
- Letting go of the broccoli on a plate (`experiments.grasp_learning.release_trials`,
  commit f0510f4af5): 867 trials from a starting world with the broccoli held and
  resting on the plate. Only opening the fingers to 98 % or more releases it (17 of 17
  when also waiting at least 1 s); finger force and retreat direction do not matter. With
  that release the whole task succeeds in 3 of 6 runs from the spawn.

Both studies wrote the same structure twice (parameter space with a hand written
statement, performer, runner, command line), and everything around them was throwaway
scripts: making the starting world, launching workers, analysing. Those scripts cost
most of the time and caused most of the mistakes.

## What the pull request already has

The package `krrood/krrood-probabilistic/src/krrood/parametrization/trials/`, with tests
in `test/krrood_test/test_underspecified_knowledge/test_trials.py` against mimic classes
in `test/krrood_test/dataset/throwing_trials.py` (throws mapped through the ORM
interface of the krrood tests, see `test/krrood_test/conftest.py`), and the page
`krrood/doc/trials.rst`:

- `records`: `TrialParameters` and `TrialMeasurements` (numbers, nested ones keyed by
  path), `Trial[TParameters, TMeasurements]` with `succeeded` and `starting_state_id`,
  `InformedField` (which action field a parameter informs; `TrialParameters.informed_fields`).
- `parameter_spaces`: `ParameterSpace` (statement derived from the parameters),
  `NearCentresRegistry` (`from_successful_trials`), and the sources a campaign tries:
  `DrawnParameters` (one seed, drawn in the main process) and `GivenParameters`.
- `campaigns`: `TrialPerformer`, `TrialStore` (`prepare` runs once before workers),
  `TrialSetup`, `TrialCampaign` (`repetitions`, `pilot_trials` that raise
  `OutcomeDidNotVaryInPilot` when all come out the same).
- `workers`: `WorkerPool` (workers sized by free memory and `memory_budget`; SIGINT or
  SIGTERM to the campaign's process lets every worker finish its trial and stop) and
  `ParallelJob`.
- `database`: `ObjectDatabase` (tables, sessions, storing, reading back by id) and
  `DatabaseTrialStore` (trials as episodes of their starting state, read back by it).
- `starting_states`: `StartingStateRecipe` and `StateReacher`; `reach_and_store` attempts
  in parallel and keeps every state `is_wanted` accepts.
- `analysis`: `TrialAnalysis` (success rate, effects of parameters and measurements with
  a chi-square `p_value`, success rate per distinct parameters, `adopted_values`) and
  `CausalTrialAnalysis` (a JPT over the trials; `causes_of_success` ranks every number
  through `ProbabilisticBackend.rank_causes`, measurements adjusted for the parameters).

## Requirements still to realize

Requirements 1 to 8 are realized as listed above. Open points from realizing them:

- The pilot refuses a campaign whose pilot trials all failed, which also stops a
  behaviour that succeeds only rarely (2 of 937 in the first grasp run). Such a campaign
  runs with `pilot_trials=0`, or more pilot trials.
- The p value comes from a chi-square test of the ranges against the outcome, not from a
  Mann Whitney test: a Mann Whitney test misses an effect where failures lie on both
  sides of the successes, such as the steep angles of a throw. A logistic fit is not
  part of it.
- `WorkerPool` reads free memory as `SC_AVPHYS_PAGES`, which leaves out the page cache,
  so it sizes conservatively; it does not cap the memory a worker actually takes.
- `CausalTrialAnalysis` fits a plain JPT; `StratifiedLearning` overfits continuous causes
  (one partition per row). Its `region` is the best region the tree found, a sub range of
  where success is likely.
- **Follow up, not in this pull request:** move `pick_up_trials` and `release_trials` on
  `robocasa-experiments` onto these classes (their stored data must keep reading
  back), then use them for the next study: the fingers that sometimes stay shut when
  opening in the sink before the lift.

## Working environment

- The repository rules are in `AGENTS.md`; krrood must not import other workspace
  packages (only `random_events` and `probabilistic_model`), so mimic classes for tests go
  into `test/krrood_test/dataset`.
- The local `.venv` has the old krrood layout and a stale `random_events_lib`, so it
  cannot run main. Tests ran in container `tom-cram-parametrized-trials` on neem-4
  (clone `/raid/users/tom_sch/repos/cram-parametrized-trials` mounted at `/opt/cram`,
  environment synced with `uv sync --extra dev --active --inexact`). Run tests with
  `CRAM_ORM_BUILD=never bash /entrypoint.sh python3 -m pytest ...`, and `krrood format`
  and `krrood check` through
  `python3 -c "import sys; from krrood.development_tools.command_line import CommandLine; sys.argv=['krrood']+sys.argv[1:]; CommandLine.main()"`.
  That clone borrows objects from a host path, so git does not work inside the
  container (`krrood check --since` fails there; use `--all-lines`), and its `origin`
  is cram2, not the fork. Edit and commit in a local checkout of the fork, copy the
  krrood sources and tests to the clone with rsync to test, format and check there, and
  copy the formatted files back before committing.
- Cap a test run's memory with `ulimit -v`; 16 GB of address space is too little for
  the krrood tests to start (`MemoryError`), 96 GB is enough.
