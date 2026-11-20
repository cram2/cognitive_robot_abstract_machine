# Trial workflow: notes and requirements

Working notes for the branch `trial-workflow` (draft PR cram2#1039, "Find out by trials
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

`krrood/krrood-probabilistic/src/krrood/parametrization/trials.py`, with tests in
`test/krrood_test/test_underspecified_knowledge/test_trials.py` against mimic classes in
`test/krrood_test/dataset/throwing_trials.py`, and the page `krrood/doc/trials.rst`:

- `TrialParameters`: a dataclass of numbers, nested ones included;
  `values_by_attribute_path()` keys every number by its path.
- `ParameterSpace[TParameters]`: derives the statement's pattern from the parameter
  dataclass; a subclass writes only `conditions` (`conditions_within` for bounds) and may
  filter draws with `accepts`.
- `NearCentresRegistry`: an equal Gaussian mixture around given parameters, with a
  standard deviation of `spread` times each parameter's allowed range.
- `TrialPerformer`, `TrialStore`, `TrialSetup`: try one trial, keep it, and describe
  what a worker builds (pickled to each worker process; `performing()` builds what cannot
  be pickled, such as a simulation).
- `TrialCampaign`: spawned worker processes, a draw seed each, every trial kept as soon as
  it is tried, refusal to start when the workers would need more memory than is free.
- `TrialAnalysis`: success rate, the effect of each parameter as success rates over
  ranges of equally many trials (strongest first), and the successful parameters.

## Requirements still to realize

These come from what went wrong or was done by hand in the two studies.

1. **Starting worlds as a recipe.** A class for "reach this state, check it is the one
   wanted, store it": the steps to get there, a predicate on the reached state (for
   example "broccoli touched by the fingers and touching the plate"), and storing through
   ormatic. Reaching it can fail and physics varies, so it should try in parallel and keep
   every state that passes the check.
2. **Trials stored as episodes through ormatic.** A generic store for any trial dataclass
   (to_dao, add, commit), with reading back by starting world. Generic dataclasses through
   ormatic are untested: prove a bound subclass such as `Trial[ThrowParameters,
   ThrowResult]` round trips in a krrood test first.
3. **Check the outcome before scaling up.** The release study first measured RoboCasa's
   judgement, which can never pass there: it needs the gripper 25 cm from the object and
   the hand moved at most 15 cm. A campaign should run a few pilot trials and refuse to
   scale up when an outcome cannot vary by construction, or at least report it.
4. **Record what actually happened, not only what was asked for.** The causal question
   needed the base pose reached, finger separation and contacts as variables (mediators);
   they were not stored for the grasps. Results should hold measured mediators next to
   the outcome.
5. **Repetition.** MuJoCo runs from an identical start differ. Promising parameters
   should be retried several times so a success rate, not a single outcome, is learned.
6. **Analysis through the probabilistic models.** Beyond the success rates per range:
   fit a probabilistic circuit of parameters and outcome, ask it for the region of the
   highest success probability, and use `krrood.parametrization.causal_search` and
   `causal_relation` where the question is causal. Report effects together with how
   sure they are (the studies used Mann Whitney tests and a logistic fit by hand).
7. **Closing the loop.** A campaign's successes become the next campaign's centres
   (`NearCentresRegistry`), and the winning values become the defaults of the action
   they belong to; say which action field a parameter informs.
8. **Memory and processes.** 48 whole kitchen workers took about 220 GB and nearly
   exhausted neem-4; stopping workers by hand killed more than intended. The campaign
   should size its workers from a memory budget and stop cleanly, keeping finished
   trials.
9. **Follow up, not in this pull request:** move `pick_up_trials` and `release_trials` on
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
  That clone borrows objects from a host path, so commit on the host, not in the
  container, after `chown -R 1004:1004 /opt/cram/.git` from inside it.
