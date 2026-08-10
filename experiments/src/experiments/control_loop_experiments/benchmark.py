"""
Measure how long the control loop of Giskard needs per cycle.

Every measurement runs in a subprocess of its own, so that worlds, symbol graphs and ros
nodes of one scenario cannot slow down the next one. Call the module without arguments to
measure everything::

    python -m experiments.control_loop_experiments.benchmark

.. note::
    The scenarios need the ``iai_pr2_description``, ``iai_kitchen`` and ``iai_apartment``
    packages on the ros package path.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from experiments.control_loop_experiments.scenarios import (
    BENCHMARK_SCENARIOS,
    IsolatedBenchmarkSession,
    PlotterMode,
    ScenarioRunner,
)
from experiments.experiment_definitions import (
    ExperimentResult,
    ExperimentsTable,
    MeanAndStandardDeviation,
    TypstRenderer,
    Unit,
)
from experiments.control_loop_experiments.control_loop_profiler import (
    CONTROL_CYCLE_PHASE,
    CallTreeProfile,
    PhasePath,
)
from krrood.exceptions import DataclassException

# %% one configuration of the sweep


@dataclass
class ControlLoopBenchmarkResult(ExperimentResult):
    """
    What the repetitions of one scenario under one plotter mode measured.
    """

    scenario_name: str
    """
    Which motion was measured.
    """

    plotter_mode: PlotterMode
    """
    Whether the post goal plotters recorded the motion while it was measured.
    """

    control_cycles: MeanAndStandardDeviation
    """
    How many cycles the motion took.
    """

    cycle_mean: MeanAndStandardDeviation
    """
    How long an average control cycle took.
    """

    cycle_p95: MeanAndStandardDeviation
    """
    How long the slower cycles took, ignoring the worst twentieth.
    """

    cycle_maximum: MeanAndStandardDeviation
    """
    How long the slowest cycle took.
    """

    budget_utilization: MeanAndStandardDeviation
    """
    Share of the control budget an average cycle used; ``1.0`` means the loop is exactly
    fast enough.
    """

    cycles_per_second: MeanAndStandardDeviation
    """
    How many cycles the loop sustains per second of cycle time.
    """

    compile_duration: MeanAndStandardDeviation
    """
    How long turning the motion statechart into a controller took.
    """

    @classmethod
    def from_profiles(
        cls, plotter_mode: PlotterMode, profiles: List[CallTreeProfile]
    ) -> ControlLoopBenchmarkResult:
        """
        Aggregate the repetitions of one configuration into a single row.

        :raises NoProfilesMeasuredError: If not a single repetition was measured.
        """
        if not profiles:
            raise NoProfilesMeasuredError(plotter_mode=plotter_mode)
        cycles = [profile.control_cycle for profile in profiles]
        return cls(
            scenario_name=profiles[0].scenario_name,
            plotter_mode=plotter_mode,
            control_cycles=MeanAndStandardDeviation.from_measurements(
                [profile.control_cycles for profile in profiles]
            ),
            cycle_mean=MeanAndStandardDeviation.from_measurements(
                [cycle.inclusive_mean for cycle in cycles], unit=Unit.SECONDS
            ).to(Unit.MILLISECONDS),
            cycle_p95=MeanAndStandardDeviation.from_measurements(
                [cycle.inclusive_percentile(95) for cycle in cycles],
                unit=Unit.SECONDS,
            ).to(Unit.MILLISECONDS),
            cycle_maximum=MeanAndStandardDeviation.from_measurements(
                [cycle.inclusive_maximum for cycle in cycles], unit=Unit.SECONDS
            ).to(Unit.MILLISECONDS),
            budget_utilization=MeanAndStandardDeviation.from_measurements(
                [profile.budget_utilization for profile in profiles]
            ),
            cycles_per_second=MeanAndStandardDeviation.from_measurements(
                [profile.cycles_per_second for profile in profiles]
            ),
            compile_duration=MeanAndStandardDeviation.from_measurements(
                [profile.compile_duration for profile in profiles], unit=Unit.SECONDS
            ).to(Unit.MILLISECONDS),
        )


@dataclass
class NoProfilesMeasuredError(DataclassException):
    """
    Raised when a configuration is aggregated that was never measured.
    """

    plotter_mode: PlotterMode
    """
    The plotter mode whose repetitions are missing.
    """

    def error_message(self) -> str:
        return (
            f"No repetition was measured under the {self.plotter_mode} plotter mode, "
            f"so there is nothing to aggregate."
        )

    def suggest_correction(self) -> str:
        return "Measure at least one repetition before aggregating."


# %% where one measurement spent its time


@dataclass
class PhaseBreakdownResult(ExperimentResult):
    """
    What one phase of the control cycle cost, as one row of the call tree.
    """

    phase: str
    """
    Name of the phase, indented by how deep it sits below the control cycle.
    """

    calls_per_cycle: float
    """
    How often the phase ran in an average cycle.
    """

    inclusive_mean_milliseconds: float
    """
    How long a call took, including the phases it called.
    """

    exclusive_mean_milliseconds: float
    """
    How long the phase itself cost per cycle, excluding the phases it called.
    """

    inclusive_p95_milliseconds: float
    """
    How long the slower calls took, ignoring the worst twentieth.
    """

    inclusive_maximum_milliseconds: float
    """
    How long the slowest call took.
    """

    share_of_cycle: float
    """
    Share of the total cycle time that went into this phase.
    """

    @classmethod
    def table_for(cls, profile: CallTreeProfile) -> ExperimentsTable:
        """
        Break one measurement down into its phases, slowest branch first at every level.
        """
        return ExperimentsTable(cls._rows_for(profile, (CONTROL_CYCLE_PHASE,)))

    @classmethod
    def _rows_for(
        cls, profile: CallTreeProfile, path: PhasePath
    ) -> List[PhaseBreakdownResult]:
        """
        The row of the phase at the given path, followed by everything it called.
        """
        samples = profile.phases[path]
        rows = [
            cls(
                phase="  " * (len(path) - 1) + path[-1],
                calls_per_cycle=round(samples.call_count / profile.control_cycles, 2),
                inclusive_mean_milliseconds=round(samples.inclusive_mean * 1000, 3),
                exclusive_mean_milliseconds=round(
                    samples.exclusive_total / profile.control_cycles * 1000, 3
                ),
                inclusive_p95_milliseconds=round(
                    samples.inclusive_percentile(95) * 1000, 3
                ),
                inclusive_maximum_milliseconds=round(
                    samples.inclusive_maximum * 1000, 3
                ),
                share_of_cycle=round(
                    samples.inclusive_total / profile.control_cycle.inclusive_total, 4
                ),
            )
        ]
        children = sorted(
            profile.children_of(path), key=lambda child: -child.inclusive_total
        )
        for child in children:
            rows.extend(cls._rows_for(profile, child.path))
        return rows


# %% driving the measurements


@dataclass
class BenchmarkMultiRun:
    """
    Runs every requested configuration in a subprocess and collects the results.
    """

    scenario_names: List[str]
    """
    The motions that are measured.
    """

    plotter_modes: List[PlotterMode]
    """
    The plotter modes every motion is measured under.
    """

    repeats: int
    """
    How often every configuration is measured.
    """

    target_frequency: float
    """
    Frequency the controller is discretized for, in hertz.
    """

    results: List[ControlLoopBenchmarkResult] = field(default_factory=list)
    """
    The configurations measured so far.
    """

    def execute(self) -> None:
        """
        Measure every configuration and aggregate its repetitions.
        """
        for scenario_name in self.scenario_names:
            for plotter_mode in self.plotter_modes:
                profiles = [
                    self._measure_in_subprocess(scenario_name, plotter_mode, repetition)
                    for repetition in range(self.repeats)
                ]
                self.results.append(
                    ControlLoopBenchmarkResult.from_profiles(plotter_mode, profiles)
                )

    def _measure_in_subprocess(
        self, scenario_name: str, plotter_mode: PlotterMode, repetition: int
    ) -> CallTreeProfile:
        """
        Run one measurement in a process of its own and read its result.

        :raises MeasurementProcessFailedError: If the measuring process did not finish.
        """
        with tempfile.NamedTemporaryFile(suffix=".json") as result_file:
            command = [
                sys.executable,
                "-m",
                "experiments.control_loop_experiments.benchmark",
                "--scenario",
                scenario_name,
                "--plotter-mode",
                plotter_mode.value,
                "--target-frequency",
                str(self.target_frequency),
                "--write-result-to",
                result_file.name,
            ]
            print(
                f"measuring {scenario_name} ({plotter_mode}, run {repetition})",
            )
            completed = subprocess.run(command)
            if completed.returncode != 0:
                raise MeasurementProcessFailedError(
                    scenario_name=scenario_name, return_code=completed.returncode
                )
            return CallTreeProfile.from_json(
                json.loads(Path(result_file.name).read_text())
            )

    def render_figure(self) -> str:
        """
        Render every configuration as one row, so the spread between them is visible.
        """
        return TypstRenderer(ExperimentsTable(self.results)).render_figure(
            "Control cycle time per scenario, as mean and standard deviation over "
            f"{self.repeats} repetitions."
        )


@dataclass
class MeasurementProcessFailedError(DataclassException):
    """
    Raised when the subprocess measuring a scenario did not finish successfully.
    """

    scenario_name: str
    """
    The motion whose measurement was attempted.
    """

    return_code: int
    """
    The code the measuring process exited with.
    """

    def error_message(self) -> str:
        return (
            f'Measuring "{self.scenario_name}" failed with return code '
            f"{self.return_code}."
        )

    def suggest_correction(self) -> str:
        return (
            f"Measure it on its own with --scenario {self.scenario_name} to see what "
            f"the process reported."
        )


# %% command line


def measure_one_scenario(arguments: argparse.Namespace) -> None:
    """
    Measure a single scenario and write the profile to the requested file.
    """
    scenario = BENCHMARK_SCENARIOS[arguments.scenario]()
    python_profiler = None if arguments.profile_to is None else cProfile.Profile()
    runner = ScenarioRunner(
        plotter_mode=PlotterMode(arguments.plotter_mode),
        target_frequency=arguments.target_frequency,
        python_profiler=python_profiler,
    )
    with IsolatedBenchmarkSession():
        profile = runner.run(scenario)
        if python_profiler is not None:
            statistics = pstats.Stats(python_profiler)
            statistics.dump_stats(arguments.profile_to)
            statistics.sort_stats("tottime").print_stats(40)
    print(
        TypstRenderer(PhaseBreakdownResult.table_for(profile)).render_figure(
            f"Where one control cycle of {profile.scenario_name} spent its time."
        )
    )
    if arguments.write_result_to is not None:
        Path(arguments.write_result_to).write_text(json.dumps(profile.to_json()))


def run_sweep(arguments: argparse.Namespace) -> None:
    """
    Measure every requested configuration and print the comparison.
    """
    plotter_modes = {
        "debug": [PlotterMode.DEBUG],
        "plain": [PlotterMode.PLAIN],
        "both": [PlotterMode.PLAIN, PlotterMode.DEBUG],
    }[arguments.plotters]
    sweep = BenchmarkMultiRun(
        scenario_names=arguments.scenarios,
        plotter_modes=plotter_modes,
        repeats=arguments.repeats,
        target_frequency=arguments.target_frequency,
    )
    sweep.execute()
    print()
    print(sweep.render_figure())


def parse_arguments() -> argparse.Namespace:
    """
    Describe what the benchmark can measure.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(BENCHMARK_SCENARIOS),
        choices=list(BENCHMARK_SCENARIOS),
        help="which motions to measure",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="how often to measure every configuration",
    )
    parser.add_argument(
        "--plotters",
        default="both",
        choices=["debug", "plain", "both"],
        help="whether the post goal plotters record while measuring",
    )
    parser.add_argument(
        "--target-frequency",
        type=float,
        default=20.0,
        help="frequency the controller is discretized for, in hertz",
    )
    parser.add_argument(
        "--write-result-to", default=None, help="file the measurements are stored in"
    )
    parser.add_argument(
        "--scenario",
        default=None,
        choices=list(BENCHMARK_SCENARIOS),
        help="measure only this scenario, in this process",
    )
    parser.add_argument(
        "--plotter-mode",
        default=PlotterMode.PLAIN.value,
        choices=[mode.value for mode in PlotterMode],
        help="whether the post goal plotters record, only together with --scenario",
    )
    parser.add_argument(
        "--profile-to",
        default=None,
        help="file the python profile is stored in, only together with --scenario",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.scenario is not None:
        measure_one_scenario(arguments)
        return
    run_sweep(arguments)


if __name__ == "__main__":
    main()
