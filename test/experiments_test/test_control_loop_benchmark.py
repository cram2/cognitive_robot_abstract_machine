import json

import pytest

from experiments.control_loop_experiments.benchmark import (
    ControlLoopBenchmarkResult,
    MeasurementProcessFailedError,
    NoProfilesMeasuredError,
    PhaseBreakdownResult,
)
from experiments.control_loop_experiments.scenarios import (
    CartesianGoalScenario,
    LongSequenceScenario,
    PlotterMode,
    ScenarioRunner,
)
from experiments.experiment_definitions import MeanAndStandardDeviation, Unit
from experiments.control_loop_experiments.control_loop_profiler import (
    CallTreeProfile,
    PhaseSamples,
)

# %% profiles built by hand, so the aggregation can be checked against known numbers


def make_profile(
    cycle_durations: list[float],
    qp_solve_durations: list[float],
    compile_duration: float = 0.5,
) -> CallTreeProfile:
    """
    A profile of a cycle that does nothing but solve a qp, with the given measurements.
    """
    cycle_path = ("control_cycle",)
    qp_solve_path = ("control_cycle", "qp_solve")
    exclusive = [
        cycle - qp_solve for cycle, qp_solve in zip(cycle_durations, qp_solve_durations)
    ]
    return CallTreeProfile(
        scenario_name="measured_motion",
        control_dt=0.05,
        wall_time=sum(cycle_durations),
        compile_duration=compile_duration,
        phases={
            cycle_path: PhaseSamples(
                path=cycle_path,
                inclusive_durations=cycle_durations,
                exclusive_durations=exclusive,
            ),
            qp_solve_path: PhaseSamples(
                path=qp_solve_path,
                inclusive_durations=qp_solve_durations,
                exclusive_durations=qp_solve_durations,
            ),
        },
    )


# %% carrying a profile across the subprocess boundary


class TestProfileSerialization:
    """
    The sweep reads its measurements back from a subprocess, so a profile has to survive
    a round trip through json.
    """

    def test_phases_survive_the_round_trip(self):
        profile = make_profile([0.02, 0.04], [0.01, 0.03])

        restored = CallTreeProfile.from_json(json.loads(json.dumps(profile.to_json())))

        assert restored == profile

    def test_phases_are_keyed_by_path_again(self):
        """
        The phases are stored as a list because json cannot key by path, so the keys
        have to be rebuilt as the tuples that index them.
        """
        profile = make_profile([0.02, 0.04], [0.01, 0.03])

        restored = CallTreeProfile.from_json(json.loads(json.dumps(profile.to_json())))

        assert set(restored.phases) == {
            ("control_cycle",),
            ("control_cycle", "qp_solve"),
        }
        assert restored.control_cycle.inclusive_durations == [0.02, 0.04]


# %% aggregating the repetitions of one configuration


class TestRepetitionAggregation:
    """
    Repeating a measurement only says something if the spread between the repetitions is
    reported.
    """

    def test_measurements_are_aggregated_over_the_repetitions(self):
        profiles = [
            make_profile([0.01, 0.03], [0.005, 0.015], compile_duration=0.4),
            make_profile([0.02, 0.04], [0.005, 0.015], compile_duration=0.6),
        ]

        result = ControlLoopBenchmarkResult.from_profiles(PlotterMode.PLAIN, profiles)

        assert result.scenario_name == "measured_motion"
        assert result.plotter_mode is PlotterMode.PLAIN
        # cycle means are 20 ms and 30 ms, so the budget of 50 ms is 40% and 60% used
        assert result.cycle_mean == MeanAndStandardDeviation.from_measurements(
            [0.02, 0.03], unit=Unit.SECONDS
        ).to(Unit.MILLISECONDS)
        assert result.budget_utilization == MeanAndStandardDeviation.from_measurements(
            [0.4, 0.6]
        )
        assert result.compile_duration == MeanAndStandardDeviation.from_measurements(
            [0.4, 0.6], unit=Unit.SECONDS
        ).to(Unit.MILLISECONDS)
        assert result.control_cycles == MeanAndStandardDeviation.from_measurements(
            [2, 2]
        )

    def test_maximum_of_every_repetition_is_kept(self):
        profiles = [
            make_profile([0.01, 0.09], [0.005, 0.015]),
            make_profile([0.01, 0.03], [0.005, 0.015]),
        ]

        result = ControlLoopBenchmarkResult.from_profiles(PlotterMode.DEBUG, profiles)

        assert result.cycle_maximum == MeanAndStandardDeviation.from_measurements(
            [0.09, 0.03], unit=Unit.SECONDS
        ).to(Unit.MILLISECONDS)

    def test_a_configuration_that_was_never_measured_is_rejected(self):
        with pytest.raises(NoProfilesMeasuredError):
            ControlLoopBenchmarkResult.from_profiles(PlotterMode.PLAIN, [])

    def test_every_measurement_becomes_a_column(self):
        assert ControlLoopBenchmarkResult.get_column_names() == [
            "scenario_name",
            "plotter_mode",
            "control_cycles",
            "cycle_mean",
            "cycle_p95",
            "cycle_maximum",
            "budget_utilization",
            "cycles_per_second",
            "compile_duration",
        ]


# %% what a failed measurement reports


class TestFailureReporting:
    """
    A benchmark that fails after minutes of measuring has to say which configuration
    broke and what to run to see why.
    """

    def test_missing_repetitions_name_the_plotter_mode(self):
        error = NoProfilesMeasuredError(plotter_mode=PlotterMode.DEBUG)

        assert error.error_message() == (
            "No repetition was measured under the debug plotter mode, so there is "
            "nothing to aggregate."
        )

    def test_failed_measurement_names_the_scenario_and_the_return_code(self):
        error = MeasurementProcessFailedError(
            scenario_name="long_sequence", return_code=3
        )

        assert error.error_message() == (
            'Measuring "long_sequence" failed with return code 3.'
        )

    def test_the_suggestion_is_part_of_what_is_raised(self):
        """
        The suggestion only helps if it reaches the traceback, which the base class
        composes into the message at construction time.
        """
        error = MeasurementProcessFailedError(
            scenario_name="long_sequence", return_code=3
        )

        assert str(error) == (
            f"{error.error_message()}\nSuggestion: {error.suggest_correction()}"
        )


# %% breaking one measurement down into its phases


class TestPhaseBreakdown:
    """
    A phase table is read to decide what to optimize, so it has to name the slowest
    branch first and attribute the time to the phase that spent it.
    """

    def test_phases_are_reported_below_the_cycle_that_called_them(self):
        profile = make_profile([0.02, 0.04], [0.015, 0.025])

        rows = PhaseBreakdownResult.table_for(profile).experiments

        assert [row.phase for row in rows] == ["control_cycle", "  qp_solve"]

    def test_time_is_split_between_a_phase_and_its_children(self):
        profile = make_profile([0.02, 0.04], [0.015, 0.025])

        cycle, qp_solve = PhaseBreakdownResult.table_for(profile).experiments

        # the cycle took 60 ms in total, of which 40 ms went into solving the qp
        assert cycle.inclusive_mean_milliseconds == 30.0
        assert cycle.exclusive_mean_milliseconds == 10.0
        assert qp_solve.inclusive_mean_milliseconds == 20.0
        assert qp_solve.share_of_cycle == pytest.approx(2 / 3, abs=1e-4)


# %% measured runtime


@pytest.fixture()
def cartesian_goal_profile(init_rospy) -> CallTreeProfile:
    return ScenarioRunner(plotter_mode=PlotterMode.PLAIN).run(CartesianGoalScenario())


@pytest.fixture()
def long_sequence_profile(init_rospy) -> CallTreeProfile:
    return ScenarioRunner(plotter_mode=PlotterMode.PLAIN).run(LongSequenceScenario())


@pytest.mark.slow
class TestPhaseAccounting:
    """
    A phase table that loses time cannot be used to decide what to optimize.
    """

    def test_measured_phases_account_for_the_whole_cycle(
        self, cartesian_goal_profile: CallTreeProfile
    ):
        control_cycle = cartesian_goal_profile.control_cycle
        children = cartesian_goal_profile.children_of(control_cycle.path)
        accounted = control_cycle.exclusive_total + sum(
            child.inclusive_total for child in children
        )

        assert accounted == pytest.approx(control_cycle.inclusive_total, rel=1e-9)


@pytest.mark.slow
class TestControlCycleBudget:
    """
    A cycle that overruns its slot makes the robot run slower than the controller was
    discretized for, and the pacer does not catch that up.
    """

    def test_cycle_stays_within_the_control_budget(
        self, long_sequence_profile: CallTreeProfile
    ):
        assert long_sequence_profile.control_cycle.inclusive_percentile(95) < (
            long_sequence_profile.control_dt
        )

    def test_motion_is_long_enough_to_judge_the_distribution(
        self, long_sequence_profile: CallTreeProfile
    ):
        assert long_sequence_profile.control_cycles >= 100
