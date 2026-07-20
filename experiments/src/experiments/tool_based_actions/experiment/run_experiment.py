"""
Campaign orchestrator of the tool-based action experiment.

Runs every trial of the configured grid in an isolated subprocess with a wall-clock
timeout, skips trials that already have recorded results, and prints a summary at the
end. Press Run — there are no command line arguments.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field

from typing_extensions import List

from experiments.tool_based_actions.experiment.configuration import (
    ExperimentConfiguration,
    TrialSpecification,
)
from experiments.tool_based_actions.experiment.results import ResultRecorder

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
                command, timeout=self.configuration.trial_timeout
            )
        except subprocess.TimeoutExpired:
            logger.error(
                "Trial %s hit the %.0f s timeout; continuing with the next trial.",
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
            specification.task.value,
            "--seed",
            str(specification.seed),
            "--configuration-json",
            self.configuration.to_json(),
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
        successes_by_task = Counter(
            result.task.value for result in results if result.success
        )
        totals_by_task = Counter(result.task.value for result in results)
        for task_name in sorted(totals_by_task):
            logger.info(
                "%s: %d/%d targets succeeded",
                task_name,
                successes_by_task.get(task_name, 0),
                totals_by_task[task_name],
            )
        logger.info("Results file: %s", self.configuration.results_file)


def main() -> None:
    """
    Run the default experiment campaign.
    """
    logging.basicConfig(level=logging.INFO)
    ExperimentRunner().run()


if __name__ == "__main__":
    main()
