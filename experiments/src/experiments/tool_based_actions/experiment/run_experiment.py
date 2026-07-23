"""
Campaign orchestrator of the tool-based action experiment.

Runs every trial of the configured grid in an isolated subprocess with a wall-clock
timeout, skips trials that already have recorded results, and prints a summary at the
end. All configuration defaults live in :class:`ExperimentConfiguration`; the command
line only overrides them.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from krrood.adapters.json_serializer import to_json
from typing_extensions import Any, Dict, List

from experiments.experiment_definitions import ExperimentsTable, TypstRenderer
from experiments.tool_based_actions.experiment.configuration import (
    ExperimentConfiguration,
    TrialSpecification,
)
from experiments.tool_based_actions.experiment.results import (
    ResultRecorder,
    TaskReliability,
)
from experiments.tool_based_actions.experiment.task_definitions import tasks_by_name

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRunner:
    """
    Executes a campaign of isolated trials with resume support.
    """

    configuration: ExperimentConfiguration = field(
        default_factory=ExperimentConfiguration
    )
    """
    The campaign configuration.
    """

    def run(self) -> None:
        """
        Run all trials of the grid that do not have recorded results yet.
        """
        recorder = ResultRecorder(self.configuration.results_file)
        specifications = self.configuration.build_trial_specifications()
        completed = recorder.completed_trial_identifiers()

        for index, specification in enumerate(specifications, start=1):
            if specification.identifier in completed:
                logger.info(
                    "[%d/%d] skipping completed trial %s",
                    index,
                    len(specifications),
                    specification.identifier,
                )
                continue
            logger.info(
                "[%d/%d] running trial %s",
                index,
                len(specifications),
                specification.identifier,
            )
            self._run_isolated(specification)

        self._log_summary(recorder)

    def _run_isolated(self, specification: TrialSpecification) -> None:
        """
        Run one trial in its own process, bounded by the configured timeout.

        :param specification: The trial to run.
        """
        command = self._trial_command(specification)
        try:
            completed_process = subprocess.run(
                command, timeout=self.configuration.trial_timeout.total_seconds()
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "Trial %s hit the %s timeout; continuing with the next trial.",
                specification.identifier,
                self.configuration.trial_timeout,
            )
            return
        if completed_process.returncode:
            logger.error(
                "Trial %s exited with code %d; continuing with the next trial.",
                specification.identifier,
                completed_process.returncode,
            )

    def _trial_command(self, specification: TrialSpecification) -> List[str]:
        """
        :param specification: The trial to run.
        :return: The subprocess command executing the trial.
        """
        return [
            sys.executable,
            "-m",
            "experiments.tool_based_actions.experiment.single_trial",
            "--task",
            specification.task.task_name(),
            "--seed",
            str(specification.seed),
            "--configuration-json",
            json.dumps(to_json(self.configuration)),
        ]

    def _log_summary(self, recorder: ResultRecorder) -> None:
        """
        Log per-task success counts over everything recorded so far.

        :param recorder: The recorder holding the campaign's results.
        """
        results = recorder.load_results()
        if not results:
            logger.info("No results recorded.")
            return
        rows = TaskReliability.from_results(results)
        for row in rows:
            logger.info(
                "%s: %d/%d targets succeeded", row.task, row.successes, row.total
            )
        table = TypstRenderer(ExperimentsTable(rows)).render_table()
        logger.info("Result table:\n%s", table)
        logger.info("Results file: %s", self.configuration.results_file)


def parse_configuration(argument_list: List[str]) -> ExperimentConfiguration:
    """
    Build the campaign configuration from command line overrides.

    :param argument_list: The command line arguments, without the program name.
    :return: The configuration with all given overrides applied.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(tasks_by_name()),
        help="Tasks to run; all of them by default.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Seeds to run every task with.",
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        help="File the trial results are appended to.",
    )
    parser.add_argument(
        "--trial-timeout-seconds",
        type=float,
        help="Wall-clock limit for a single trial process.",
    )
    arguments = parser.parse_args(argument_list)

    overrides: Dict[str, Any] = {}
    if arguments.tasks is not None:
        overrides["tasks"] = [tasks_by_name()[name] for name in arguments.tasks]
    if arguments.seeds is not None:
        overrides["seeds"] = arguments.seeds
    if arguments.results_file is not None:
        overrides["results_file"] = arguments.results_file
    if arguments.trial_timeout_seconds is not None:
        overrides["trial_timeout"] = timedelta(
            seconds=arguments.trial_timeout_seconds
        )
    return ExperimentConfiguration(**overrides)


def main() -> None:
    """
    Run the experiment campaign described by the command line arguments.
    """
    logging.basicConfig(level=logging.INFO)
    ExperimentRunner(configuration=parse_configuration(sys.argv[1:])).run()


if __name__ == "__main__":
    main()
