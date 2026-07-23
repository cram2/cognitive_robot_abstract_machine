"""
Result recording for the tool-based action experiment.

Results are stored as JSON lines so runs can append concurrently and a crashed campaign
can resume from what is already on disk. Per-target results aggregate into
:class:`TaskReliability` rows, an
:class:`~experiments.experiment_definitions.ExperimentResult`, for rendering the campaign
summary as a table for a paper.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import MISSING, dataclass, fields
from datetime import timedelta
from pathlib import Path

from krrood.adapters.exceptions import JSON_TYPE_NAME
from krrood.adapters.json_serializer import from_json, to_json
from typing_extensions import Any, Dict, List, Optional, Set, Type

from experiments.experiment_definitions import ExperimentResult
from experiments.tool_based_actions.experiment.configuration import TrialSpecification
from experiments.tool_based_actions.experiment.exceptions import (
    IncompatibleResultRecord,
)
from experiments.tool_based_actions.experiment.scene import PlanarPose
from experiments.tool_based_actions.experiment.task_definitions import (
    ToolTaskDefinition,
)


@dataclass(frozen=True)
class FailureDescription:
    """
    The recordable description of a failed action.

    Holds the exception type and its message instead of the exception object itself, so
    failures stay serializable regardless of what the raised exception carries.
    """

    exception_type: Type[Exception]
    """
    Type of the exception that aborted the action.
    """

    message: str
    """
    Message of the exception that aborted the action.
    """

    @classmethod
    def from_exception(cls, error: Exception) -> FailureDescription:
        """
        :param error: The exception that aborted the action.
        :return: The recordable description of the failure.
        """
        return cls(exception_type=type(error), message=str(error))


@dataclass(frozen=True)
class TargetResult:
    """
    The outcome of one tool action on one target of a trial.

    A plain dataclass, so :func:`krrood.adapters.json_serializer.to_json` persists it as
    one JSON line without custom serialization code; aggregated into
    :class:`TaskReliability` rows for the campaign summary table.
    """

    trial_identifier: str
    """
    Identifier of the trial the target belongs to.
    """

    task: Type[ToolTaskDefinition]
    """
    The task that was performed.
    """

    seed: int
    """
    Seed of the trial's scene.
    """

    robot_name: str
    """
    Name of the robot that performed the action.
    """

    environment_name: str
    """
    Name of the environment the action ran in.
    """

    target_name: str
    """
    Name of the target the action acted on.
    """

    target_pose: PlanarPose
    """
    Position and orientation of the target in the world's XY plane.
    """

    target_scale: float
    """
    Uniform scale factor the target was spawned with.
    """

    surface_name: str
    """
    Name of the surface the target was spawned on.
    """

    duration: timedelta
    """
    Wall-clock duration of the action.
    """

    failure: Optional[FailureDescription] = None
    """
    Description of the failure that aborted the action, or None on success.
    """

    @property
    def success(self) -> bool:
        """
        :return: True if the action completed without an error.
        """
        return self.failure is None

    @classmethod
    def _validate_record_matches_schema(cls, record: Dict[str, Any]) -> None:
        """
        Check that a deserialized record carries exactly the fields of this class.

        :param record: The deserialized JSON record.
        :raises IncompatibleResultRecord: If required fields are missing or unknown
            fields are present.
        """
        payload_fields = record.keys() - {JSON_TYPE_NAME}
        field_names = {field.name for field in fields(cls)}
        required_field_names = {
            field.name
            for field in fields(cls)
            if field.default is MISSING and field.default_factory is MISSING
        }
        missing_fields = sorted(required_field_names - payload_fields)
        unexpected_fields = sorted(payload_fields - field_names)
        if missing_fields or unexpected_fields:
            raise IncompatibleResultRecord(missing_fields, unexpected_fields)


@dataclass
class TaskReliability(ExperimentResult):
    """
    One per-task reliability row of the campaign summary, renderable as a paper table
    via :class:`~experiments.experiment_definitions.ExperimentsTable` and
    :class:`~experiments.experiment_definitions.TypstRenderer`.
    """

    task: str
    """
    Name of the task the row aggregates.
    """

    successes: int
    """
    Number of targets the task succeeded on.
    """

    total: int
    """
    Total number of targets attempted for the task.
    """

    @classmethod
    def from_results(cls, results: List[TargetResult]) -> List[TaskReliability]:
        """
        :param results: The target results to aggregate.
        :return: One reliability row per task, ordered by task name.
        """
        successes = Counter(
            result.task.task_name() for result in results if result.success
        )
        totals = Counter(result.task.task_name() for result in results)
        return [
            cls(task=name, successes=successes.get(name, 0), total=totals[name])
            for name in sorted(totals)
        ]


@dataclass
class ResultRecorder:
    """
    Appends target results to a JSON lines file and answers resume queries.
    """

    results_file: Path
    """
    The file results are appended to.
    """

    def record(self, result: TargetResult) -> None:
        """
        Append one result to the results file.

        :param result: The result to persist.
        """
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        with self.results_file.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(to_json(result)) + "\n")

    def load_results(self) -> List[TargetResult]:
        """
        :return: All results recorded so far, oldest first.
        """
        if not self.results_file.exists():
            return []
        lines = self.results_file.read_text(encoding="utf-8").splitlines()
        return [self._load_record(line) for line in lines if line.strip()]

    @staticmethod
    def _load_record(line: str) -> TargetResult:
        """
        :param line: One JSON line from the results file.
        :return: The deserialized result.
        :raises IncompatibleResultRecord: If the line was written by an older schema.
        """
        record = json.loads(line)
        TargetResult._validate_record_matches_schema(record)
        return from_json(record)

    def completed_trial_identifiers(self) -> Set[str]:
        """
        :return: Identifiers of trials that already have at least one recorded
            result.
        """
        return {result.trial_identifier for result in self.load_results()}

    def is_completed(self, specification: TrialSpecification) -> bool:
        """
        :param specification: The trial to check.
        :return: True if the trial already has recorded results.
        """
        return specification.identifier in self.completed_trial_identifiers()
