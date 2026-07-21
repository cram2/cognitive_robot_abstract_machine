"""
Result recording for the tool-based action experiment.

Results are stored as JSON lines so runs can append concurrently and a crashed campaign
can resume from what is already on disk.
"""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path

from typing_extensions import Any, Dict, List, Optional, Set

from experiments.tool_based_actions.experiment.configuration import (
    ToolBasedTask,
    TrialSpecification,
)


class IncompatibleResultRecord(Exception):
    """
    Raised when a stored result line does not match the current
    :class:`TargetResult` schema, typically because the results file was written
    by an older version of the experiment.
    """

    def __init__(self, missing_fields: List[str], unexpected_fields: List[str]):
        super().__init__(
            f"Result record does not match the current TargetResult schema: "
            f"missing fields {missing_fields}, unexpected fields "
            f"{unexpected_fields}. Archive or delete the results file to start "
            f"a fresh campaign."
        )


@dataclass(frozen=True)
class TargetResult:
    """
    The outcome of one tool action on one target of a trial.
    """

    trial_identifier: str
    """
    Identifier of the trial the target belongs to.
    """

    task: ToolBasedTask
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

    target_x: float
    """
    X coordinate of the target in the world frame.
    """

    target_y: float
    """
    Y coordinate of the target in the world frame.
    """

    target_yaw: float
    """
    Rotation in radians of the target around the world Z axis.
    """

    target_scale: float
    """
    Uniform scale factor the target was spawned with.
    """

    surface_name: str
    """
    Name of the surface the target was spawned on.
    """

    success: bool
    """
    True if the action completed without an error.
    """

    duration: float
    """
    Wall-clock duration of the action in seconds.
    """

    failure_reason: Optional[str] = None
    """
    Compact description of the failure, or None on success.
    """

    def to_json_line(self) -> str:
        """
        :return: This result serialized as one JSON line.
        """
        record = asdict(self)
        record["task"] = self.task.value
        return json.dumps(record)

    @classmethod
    def from_json_line(cls, line: str) -> TargetResult:
        """
        :param line: One JSON line produced by :meth:`to_json_line`.
        :return: The deserialized result.
        """
        record = json.loads(line)
        cls._validate_record_matches_schema(record)
        record["task"] = ToolBasedTask(record["task"])
        return cls(**record)

    @classmethod
    def _validate_record_matches_schema(cls, record: Dict[str, Any]) -> None:
        """
        Check that a deserialized record carries exactly the fields of this class.

        :param record: The deserialized JSON record.
        :raises IncompatibleResultRecord: If required fields are missing or unknown
            fields are present.
        """
        field_names = {field.name for field in fields(cls)}
        required_field_names = {
            field.name
            for field in fields(cls)
            if field.default is MISSING and field.default_factory is MISSING
        }
        missing_fields = sorted(required_field_names - record.keys())
        unexpected_fields = sorted(record.keys() - field_names)
        if missing_fields or unexpected_fields:
            raise IncompatibleResultRecord(missing_fields, unexpected_fields)


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
            stream.write(result.to_json_line() + "\n")

    def load_results(self) -> List[TargetResult]:
        """
        :return: All results recorded so far, oldest first.
        """
        if not self.results_file.exists():
            return []
        lines = self.results_file.read_text(encoding="utf-8").splitlines()
        return [TargetResult.from_json_line(line) for line in lines if line.strip()]

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
