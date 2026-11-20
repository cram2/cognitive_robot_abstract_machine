Finding What Makes a Behaviour Succeed
======================================

When a behaviour fails in a way no single fix explains, the module
:mod:`krrood.parametrization.trials` turns the failure into an experiment:

1. Reach the state the failure shows in and keep it, so every trial starts from it: a
   :class:`~krrood.parametrization.trials.starting_states.StartingStateRecipe` states the
   steps that lead to it and how to tell the state reached is the one wanted, such as an
   object held by the fingers and touching its target. Reaching it can fail and physics
   varies, so its
   :meth:`~krrood.parametrization.trials.starting_states.StartingStateRecipe.reach_and_store`
   attempts it in several workers at once and keeps every state that is the one wanted.
2. Declare what the behaviour may vary by as a dataclass of numbers, a
   :class:`~krrood.parametrization.trials.records.TrialParameters`, and bound it with a
   :class:`~krrood.parametrization.trials.parameter_spaces.ParameterSpace`. Declare
   what is measured while a trial runs, such as the pose actually reached, as a
   :class:`~krrood.parametrization.trials.records.TrialMeasurements`, and keep both with
   the outcome in a :class:`~krrood.parametrization.trials.records.Trial`.
3. Try many draws at once with a
   :class:`~krrood.parametrization.trials.campaigns.TrialCampaign`, which keeps every
   trial in a :class:`~krrood.parametrization.trials.campaigns.TrialStore` as soon as it
   is tried.
4. Read with a :class:`~krrood.parametrization.trials.analysis.TrialAnalysis` which
   parameters and measurements change the success rate, then draw the next campaign near
   the successful trials with a
   :class:`~krrood.parametrization.trials.parameter_spaces.NearCentresRegistry`.

The parameters may nest dataclasses; every number becomes a parameter, keyed by its path
of attributes such as ``release.height``. A space states only its conditions, reaching
each parameter from the pattern it is given:

.. code-block:: python

    @dataclass(kw_only=True)
    class ThrowSpace(ParameterSpace[ThrowParameters]):
        angle: Bounds[float] = field(default_factory=lambda: Bounds(lower=0.0, upper=1.5))
        height: Bounds[float] = field(default_factory=lambda: Bounds(lower=1.0, upper=2.0))

        def conditions(self, *, parameters):
            return [
                *self.conditions_within(value=parameters.angle, bounds=self.angle),
                *self.conditions_within(value=parameters.release.height, bounds=self.height),
            ]

A :class:`~krrood.parametrization.trials.campaigns.TrialSetup` says how a worker tries and keeps its
trials. It is sent to every worker process, so it builds what cannot be pickled, such as
a simulation, inside :meth:`~krrood.parametrization.trials.campaigns.TrialSetup.performing`.
A :class:`~krrood.parametrization.trials.database.DatabaseTrialStore` keeps every trial
through ormatic as an episode of the stored state it started from, and reads back the
trials of one starting state:

.. code-block:: python

    setup = ThrowingSetup(database_url="sqlite:///throws.sqlite", starting_state_id=1)
    campaign = TrialCampaign(
        setup=setup,
        parameter_source=DrawnParameters(
            parameter_space=ThrowSpace(),
            model_registry=UniformPriorRegistry(),
            number_of_draws=160,
        ),
        pilot_trials=20,
        worker_pool=WorkerPool(
            number_of_workers=48, memory_per_worker=4 * 10**9, memory_budget=120 * 10**9
        ),
    )
    campaign.run()

    analysis = TrialAnalysis(trials=setup.create_store().stored_trials())
    [strongest, *_] = analysis.effects_on_success()
    registry = NearCentresRegistry(centres=analysis.successful_parameters(), spread=0.1)

The :class:`~krrood.parametrization.trials.workers.WorkerPool` runs only as many workers as
fit into the free working memory and its budget, here 30. Stop a campaign by
interrupting or terminating its own process, never its workers: every worker finishes
its current trial and stops, and every finished trial stays kept.

The first :attr:`~krrood.parametrization.trials.campaigns.TrialCampaign.pilot_trials`
are tried before any worker starts. When they all come out the same, the campaign stops
with :class:`~krrood.parametrization.trials.exceptions.OutcomeDidNotVaryInPilot`: an
outcome that cannot succeed by construction would make every analysis empty.

Every :class:`~krrood.parametrization.trials.analysis.EffectOnSuccess` says with its
:attr:`~krrood.parametrization.trials.analysis.EffectOnSuccess.p_value` how likely so
different success rates arise by chance. Where the question is causal, a
:class:`~krrood.parametrization.trials.analysis.CausalTrialAnalysis` fits a joint
probability tree to the trials and searches every parameter and measurement as a cause
of success through the probabilistic backend. Each
:class:`~krrood.parametrization.trials.analysis.CauseOfSuccess` gives the region of its
values that makes success most likely and how likely success is when it is set there.
A measurement, such as the pose actually reached, is searched adjusted for the
parameters, so a mediator is told apart from the parameters it depends on:

.. code-block:: python

    [strongest, *_] = CausalTrialAnalysis(trials=trials).causes_of_success()
    strongest.attribute_path  # "measurements.reached_angle"
    strongest.region  # Bounds(lower=0.60, upper=0.79)

A simulation run twice from the same state need not end the same. A campaign tries each
parameters :attr:`~krrood.parametrization.trials.campaigns.TrialCampaign.repetitions`
times, and
:meth:`~krrood.parametrization.trials.analysis.TrialAnalysis.success_rates_of_parameters`
gives the success rate of each. To retry the most promising parameters of an earlier
campaign, give them as
:class:`~krrood.parametrization.trials.parameter_spaces.GivenParameters`.

.. note::
    A pilot cannot tell an outcome that never passes from one that passes rarely. When
    it stops, check the judgement of success on a state where it must pass before trying
    more pilot trials.
