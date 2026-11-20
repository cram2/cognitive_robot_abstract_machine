Finding What Makes a Behaviour Succeed
======================================

When a behaviour fails in a way no single fix explains, the module
:mod:`krrood.parametrization.trials` turns the failure into an experiment:

1. Reach the state the failure shows in once and keep it, so every trial starts from it.
2. Declare what the behaviour may vary by as a dataclass of numbers, a
   :class:`~krrood.parametrization.trials.TrialParameters`, and bound it with a
   :class:`~krrood.parametrization.trials.ParameterSpace`.
3. Try many draws at once with a :class:`~krrood.parametrization.trials.TrialCampaign`,
   which keeps every trial in a :class:`~krrood.parametrization.trials.TrialStore` as soon
   as it is tried.
4. Read with a :class:`~krrood.parametrization.trials.TrialAnalysis` which parameters
   change the success rate, then draw the next campaign near the successful trials with a
   :class:`~krrood.parametrization.trials.NearCentresRegistry`.

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

A :class:`~krrood.parametrization.trials.TrialSetup` says how a worker tries and keeps its
trials. It is sent to every worker process, so it builds what cannot be pickled, such as
a simulation, inside :meth:`~krrood.parametrization.trials.TrialSetup.performing`:

.. code-block:: python

    campaign = TrialCampaign(
        setup=ThrowingSetup(path=Path("throws.jsonl")),
        number_of_workers=8,
        trials_per_worker=20,
        memory_per_worker=4 * 10**9,
    )
    campaign.run()

    analysis = TrialAnalysis(outcomes=outcomes_read_from_the_store)
    [strongest, *_] = analysis.parameter_effects()
    registry = NearCentresRegistry(centres=analysis.successful_parameters(), spread=0.1)

.. note::
    Check on a few trials that the measured outcome can come out both ways before
    starting a large campaign: an outcome that cannot succeed by construction makes every
    analysis empty.
