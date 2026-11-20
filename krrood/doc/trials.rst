Finding What Makes a Behaviour Succeed
======================================

When a behaviour fails in a way no single fix explains, the package
:mod:`krrood.parametrization.trials` turns the failure into an experiment:

1. Reach the state the failure shows in and keep it, so every trial starts from it: a
   :class:`~krrood.parametrization.trials.starting_states.StartingStateRecipe` states the
   steps that lead to it and how to tell the state reached is the one wanted, such as an
   object held by the fingers and touching its target. Reaching it can fail and physics
   varies, so its
   :meth:`~krrood.parametrization.trials.starting_states.StartingStateRecipe.reach_and_store`
   attempts it in several workers at once and keeps every state that is the one wanted.
2. Declare what the behaviour may vary by as a dataclass, a
   :class:`~krrood.parametrization.trials.records.TrialParameters`, and bound it with a
   :class:`~krrood.parametrization.trials.parameter_spaces.ParameterSpace`. Declare what
   is measured while a trial runs, such as the pose actually reached, as a
   :class:`~krrood.parametrization.trials.records.TrialMeasurements`, and keep both with
   the outcome in a :class:`~krrood.parametrization.trials.records.Trial`.
3. Try many draws at once with a
   :class:`~krrood.parametrization.trials.campaigns.TrialCampaign`, which keeps every
   trial in a :class:`~krrood.parametrization.trials.campaigns.TrialStore` as soon as it
   is tried.
4. Learn a :class:`~krrood.parametrization.trials.models.TrialModel` of the trials and
   ask it, with the queries of the entity query language, what made them succeed.
5. Close the loop: draw the next campaign near the successful trials, and adopt the
   values that worked best as the defaults of the action they belong to.

A space states only its conditions, reaching each parameter, nested ones included, from
the pattern it is given:

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

A :class:`~krrood.parametrization.trials.campaigns.TrialSetup` says how a worker tries and
keeps its trials. It is sent to every worker process, so it builds what cannot be
pickled, such as a simulation, inside
:meth:`~krrood.parametrization.trials.campaigns.TrialSetup.performing`. A
:class:`~krrood.parametrization.trials.database.DatabaseTrialStore` keeps every trial
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
            number_of_workers=48,
            memory_per_worker=4 * 10**9,
            memory=BudgetedMemory(memory=HostMemory(), budget_bytes=120 * 10**9),
        ),
    )
    campaign.run()
    trials = setup.create_store().stored_trials()

The :class:`~krrood_compute_resources.workers.WorkerPool` runs only as many workers as fit
into the free memory it is given, here at most 30 within a budget of 120 GB. Stop a
campaign by interrupting or terminating its own process, never its workers: every worker
finishes its current trial and stops, and every finished trial stays kept.

The first :attr:`~krrood.parametrization.trials.campaigns.TrialCampaign.pilot_trials`
are tried before any worker starts. When they all come out the same, the campaign stops
with :class:`~krrood.parametrization.trials.exceptions.OutcomeDidNotVaryInPilot`: an
outcome that cannot succeed by construction would make every analysis empty.

.. note::
    A pilot cannot tell an outcome that never passes from one that passes rarely. When
    it stops, check the judgement of success on a state where it must pass before trying
    more pilot trials.

A simulation run twice from the same state need not end the same. A campaign tries each
parameters :attr:`~krrood.parametrization.trials.campaigns.TrialCampaign.repetitions`
times; to retry the most promising parameters of an earlier campaign, give them as
:class:`~krrood.parametrization.trials.parameter_spaces.GivenParameters`. The success
rate of each parameters is a query of the trials:

.. code-block:: python

    throw = variable(Throw, domain=trials)
    success_rate = average(throw.succeeded)
    set_of(throw.parameters.angle, success_rate).grouped_by(throw.parameters.angle)

What made the trials succeed
----------------------------

A :class:`~krrood.parametrization.trials.models.TrialModel` learns a joint probability
tree of the features of the trials and the causal structure between them, in which
parameters come before measurements and measurements before the outcome. Its
:attr:`~krrood.parametrization.trials.models.TrialModel.registry` answers queries of
probabilities, and its
:attr:`~krrood.parametrization.trials.models.TrialModel.causal_backend` causal queries. A
causal query that marks no cause ranks only the features whose edge to the outcome
survives the independence tests of the structure learner, so a feature that does not
change the outcome is left out. A measurement is adjusted for the parameters it depends
on, so a mediator is told apart from them:

.. code-block:: python

    model = TrialModel(trials=trials)
    throw = variable(Throw, domain=trials)
    probability_of(throw.succeeded == True, throw.measurements.reached_angle < 0.6).first(
        backend=ProbabilisticBackend(model_registry=model.registry)
    )

    landing = a(Throw)(
        parameters=a(ThrowParameters)(angle=..., speed=..., release=a(ReleasePoint)(height=...)),
        measurements=a(ThrowMeasurements)(reached_angle=...),
        succeeded=...,
    )
    landing.causes_effect(landing.succeeded == True)
    [strongest, *_] = model.causal_backend.rank_causes(expression=landing)

Closing the loop
----------------

:meth:`~krrood.parametrization.trials.models.TrialModel.registry_near_successes` fits a
:class:`~probabilistic_model.learning.row_kernel_mixture.RowKernelMixture` to the
parameters of the trials that succeeded, as wide as the parameters of all trials vary,
and draws the next campaign from it. A space declares in
:attr:`~krrood.parametrization.trials.parameter_spaces.ParameterSpace.informed_fields`
which field of which action each parameter informs, and
:meth:`~krrood.parametrization.trials.parameter_spaces.ParameterSpace.adopted_values`
gives the values of the best parameters for them, next to the default each would
replace:

.. code-block:: python

    @dataclass(kw_only=True)
    class ThrowSpace(ParameterSpace[ThrowParameters]):
        @classproperty
        def informed_fields(cls):
            parameters = variable(ThrowParameters, domain=[])
            action = variable(ThrowBall, domain=[])
            return [InformedField(parameter=parameters.angle, informed=action.aim_angle)]

    next_source = DrawnParameters(
        parameter_space=ThrowSpace(),
        model_registry=model.registry_near_successes(),
        number_of_draws=160,
    )
    for adopted in ThrowSpace().adopted_values(parameters=best):
        print(adopted.informed_field.current_default, "->", adopted.value)
