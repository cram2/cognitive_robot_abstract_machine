"""
Configuration of the tool-based action experiment.

An experiment is a grid of trials: every configured task is run once per seed. Each
trial spawns its targets at seeded random poses, so rerunning a trial specification
reproduces the exact same scene.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from typing_extensions import List, Tuple


class ToolBasedTask(enum.Enum):
    """
    The tool-based composite tasks the experiment can run.
    """

    CUTTING = "cutting"
    MIXING = "mixing"
    POURING = "pouring"
    WIPING = "wiping"


@dataclass(frozen=True)
class SpawnRegion:
    """
    An axis-aligned rectangle on a support surface in which targets are spawned.
    """

    minimum_x: float
    """
    Lower X bound of the region in the world frame.
    """

    maximum_x: float
    """
    Upper X bound of the region in the world frame.
    """

    minimum_y: float
    """
    Lower Y bound of the region in the world frame.
    """

    maximum_y: float
    """
    Upper Y bound of the region in the world frame.
    """

    height: float
    """
    Z coordinate in the world frame at which targets are spawned.
    """

    def contains(self, x: float, y: float) -> bool:
        """
        :param x: X coordinate in the world frame.
        :param y: Y coordinate in the world frame.
        :return: True if the point lies inside the region.
        """
        return (
            self.minimum_x <= x <= self.maximum_x
            and self.minimum_y <= y <= self.maximum_y
        )

    def grid_capacity(self, clearance: float) -> int:
        """
        :param clearance: Minimum distance in meters between two targets.
        :return: A conservative number of targets that provably fit into the region,
            based on an axis-aligned grid packing.
        """
        columns = int((self.maximum_x - self.minimum_x) / clearance) + 1
        rows = int((self.maximum_y - self.minimum_y) / clearance) + 1
        return columns * rows


@dataclass(frozen=True)
class TrialSpecification:
    """
    One fully reproducible trial of the experiment grid.
    """

    task: ToolBasedTask
    """
    The tool-based task the trial runs.
    """

    seed: int
    """
    Seed that fixes the sampled scene of this trial.
    """

    robot_name: str
    """
    Name of the robot the trial runs with.
    """

    environment_name: str
    """
    Name of the environment the trial runs in.
    """

    @property
    def identifier(self) -> str:
        """
        :return: A unique, human-readable identifier of this trial.
        """
        return (
            f"{self.task.value}:{self.environment_name}:{self.robot_name}:{self.seed}"
        )


@dataclass(frozen=True)
class ExperimentConfiguration:
    """
    The full configuration of one experiment campaign.
    """

    tasks: Tuple[ToolBasedTask, ...] = tuple(ToolBasedTask)
    """
    The tasks to run.
    """

    seeds: Tuple[int, ...] = (910001, 910002, 910003)
    """
    The seeds to run every task with.
    """

    robot_name: str = "pr2"
    """
    Name of the robot the trials run with, recorded with every result.
    """

    environment_name: str = "apartment"
    """
    Name of the environment the trials run in, recorded with every result.
    """

    minimum_targets_per_trial: int = 2
    """
    Smallest number of targets a trial spawns.
    """

    maximum_targets_per_trial: int = 3
    """
    Largest number of targets a trial spawns.
    """

    target_clearance: float = 0.35
    """
    Minimum distance in meters between two spawned targets.
    """

    surface_names: Tuple[str, ...] = (
        "island_countertop",
        "countertop",
        "table_area_main",
    )
    """
    Names of the support surface bodies targets are spawned on.
    """

    surface_margin: float = 0.15
    """
    Distance in meters kept from every surface edge when spawning.
    """

    spawn_height_offset: float = 0.05
    """
    Height in meters above a surface top at which targets are spawned.
    """

    results_file: Path = field(
        default_factory=lambda: Path(__file__).parent
        / "records"
        / "tool_based_experiment_results.jsonl"
    )
    """
    File the trial results are appended to, one JSON object per line.
    """

    trial_timeout: float = 600.0
    """
    Wall-clock limit in seconds for a single trial process.
    """

    def build_trial_specifications(self) -> List[TrialSpecification]:
        """
        :return: The full trial grid, one specification per task and seed.
        """
        return [
            TrialSpecification(
                task=task,
                seed=seed,
                robot_name=self.robot_name,
                environment_name=self.environment_name,
            )
            for task in self.tasks
            for seed in self.seeds
        ]

    def to_json(self) -> str:
        """
        :return: This configuration serialized as JSON, e.g. to hand it to a trial
            subprocess.
        """
        record = asdict(self)
        record["tasks"] = [task.value for task in self.tasks]
        record["results_file"] = str(self.results_file)
        return json.dumps(record)

    @classmethod
    def from_json(cls, text: str) -> ExperimentConfiguration:
        """
        :param text: JSON produced by :meth:`to_json`.
        :return: The deserialized configuration.
        """
        record = json.loads(text)
        record["tasks"] = tuple(ToolBasedTask(task) for task in record["tasks"])
        record["seeds"] = tuple(record["seeds"])
        record["surface_names"] = tuple(record["surface_names"])
        record["results_file"] = Path(record["results_file"])
        return cls(**record)
