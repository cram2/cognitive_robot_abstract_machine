"""
Runs demos with SegMind watching the objects their plans handle, and reports for each demo
which of its ground-truth events SegMind detected, which it missed, and what else it
detected.

A demo's ground truth is written from the actions of its plan. Each expected event has to
be matched by a detected event of its own; detected events left over are reported, not
counted against the demo.

Usage::

    python segmind/scripts/report_demo_events.py                # every demo
    python segmind/scripts/report_demo_events.py bullet_world   # some of them
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from typing_extensions import List, Optional, Type

from segmind.datastructures.events import (
    DetectionEvent,
    LossOfSupportEvent,
    PickUpEvent,
    PlacingEvent,
    StopTranslationEvent,
    SupportEvent,
    TranslationEvent,
)
from segmind.live_segmenter import EventRecord, SegmindEnvironmentVariable

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
"""
The directory demo scripts are named relative to.
"""

DEMO_TIMEOUT_SECONDS = 600
"""
How long a demo may run before it is reported as not finished.
"""


class DemoName(StrEnum):
    """
    The demos a ground truth is written for, by the name they are asked for on the
    command line.
    """

    BULLET_WORLD = "bullet_world"


# %% ground truths


@dataclass(frozen=True)
class ExpectedEvent:
    """
    An event a demo's plan should give rise to.
    """

    event_type: Type[DetectionEvent]
    """
    The kind of event.
    """

    tracked_object: str
    """
    The name of the body the event is about.
    """

    with_object: Optional[str] = None
    """
    The name of the other body the action determines, or none where it determines none.
    """

    def is_matched_by(self, record: EventRecord) -> bool:
        """
        :return: Whether ``record`` is a detection of this event.
        """
        if record.event_type != self.event_type.__name__:
            return False
        if record.tracked_object != self.tracked_object:
            return False
        return self.with_object is None or record.with_object == self.with_object

    def __str__(self) -> str:
        with_object = f" with {self.with_object}" if self.with_object else ""
        return f"{self.event_type.__name__}({self.tracked_object}{with_object})"


def transported(body: str, taken_from: str, put_on: str) -> List[ExpectedEvent]:
    """
    The events of picking ``body`` up from ``taken_from`` and placing it on ``put_on``.

    A pick-up is concluded from the object losing its support and moving, and a placing
    from it coming to rest on a surface, so nothing here names the robot: what the
    gripper does to the object is not yet detected.
    """
    return [
        ExpectedEvent(LossOfSupportEvent, body, taken_from),
        ExpectedEvent(TranslationEvent, body),
        ExpectedEvent(PickUpEvent, body),
        ExpectedEvent(StopTranslationEvent, body),
        ExpectedEvent(SupportEvent, body, put_on),
        ExpectedEvent(PlacingEvent, body, put_on),
    ]


@dataclass
class DemoGroundTruth:
    """
    A demo, and the events its plan should give rise to.
    """

    name: DemoName
    """
    The name the demo is asked for by.
    """

    script: Path
    """
    The demo's script, relative to the repository root.
    """

    expected_events: List[ExpectedEvent]
    """
    The events the plan's actions should give rise to.
    """

    extra_arguments: List[str] = field(default_factory=list)
    """
    Arguments the script is started with.
    """


GROUND_TRUTHS = [
    DemoGroundTruth(
        name=DemoName.BULLET_WORLD,
        script=Path("coraplex/demos/coraplex_bullet_world_demo/demo.py"),
        expected_events=[
            *transported("milk.stl", "island_countertop", "table_area_main"),
            *transported("bowl.stl", "island_countertop", "table_area_main"),
            *transported("spoon.stl", "cabinet10_drawer_top", "table_area_main"),
        ],
    ),
]
"""
The ground truth of the demo SegMind is wired into: the three objects the plan takes
somewhere else. The cereal box is left standing where it is.
"""


# %% running a demo and scoring it


class RunOutcome(StrEnum):
    """
    How a demo's run ended.
    """

    FINISHED = "finished"
    FAILED = "failed"
    NO_EVENTS_WRITTEN = "finished without writing SegMind's events"


@dataclass
class DemoReport:
    """
    What SegMind detected while a demo ran, set against the demo's ground truth.
    """

    ground_truth: DemoGroundTruth
    """
    The demo and its expected events.
    """

    outcome: RunOutcome
    """
    How the run ended.
    """

    records: List[EventRecord]
    """
    Every event SegMind detected.
    """

    found: List[ExpectedEvent] = field(init=False, default_factory=list)
    """
    The expected events a detected event matched.
    """

    missed: List[ExpectedEvent] = field(init=False, default_factory=list)
    """
    The expected events no detected event matched.
    """

    extra: List[EventRecord] = field(init=False, default_factory=list)
    """
    The detected events no expected event took.
    """

    def __post_init__(self):
        unmatched = list(self.records)
        for expected in self.ground_truth.expected_events:
            match = next(
                (record for record in unmatched if expected.is_matched_by(record)), None
            )
            if match is None:
                self.missed.append(expected)
                continue
            unmatched.remove(match)
            self.found.append(expected)
        self.extra = unmatched

    def render(self) -> str:
        """
        :return: The report as readable lines.
        """
        total = len(self.ground_truth.expected_events)
        lines = [
            f"== {self.ground_truth.name}: {self.outcome}, "
            f"{len(self.found)} of {total} expected events found",
            *(f"   found   {expected}" for expected in self.found),
            *(f"   MISSED  {expected}" for expected in self.missed),
            *(
                f"   extra   {record.event_type}({record.tracked_object}"
                f"{' with ' + record.with_object if record.with_object else ''})"
                for record in self.extra
            ),
        ]
        return "\n".join(lines)


def run(ground_truth: DemoGroundTruth) -> DemoReport:
    """
    Run a demo in a process of its own with SegMind's events written to a temporary
    file, and score them against the demo's ground truth.
    """
    with tempfile.TemporaryDirectory() as directory:
        events_file = Path(directory) / "events.json"
        environment = {
            **os.environ,
            SegmindEnvironmentVariable.EVENTS_FILE: str(events_file),
        }
        completed = subprocess.run(
            [
                sys.executable,
                str(REPOSITORY_ROOT / ground_truth.script),
                *ground_truth.extra_arguments,
            ],
            cwd=REPOSITORY_ROOT / ground_truth.script.parent,
            env=environment,
            timeout=DEMO_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            return DemoReport(ground_truth, RunOutcome.FAILED, [])
        if not events_file.exists():
            return DemoReport(ground_truth, RunOutcome.NO_EVENTS_WRITTEN, [])
        return DemoReport(
            ground_truth, RunOutcome.FINISHED, EventRecord.read_all(events_file)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "demos",
        nargs="*",
        choices=[name.value for name in DemoName],
        help="The demos to run; every demo when none is named.",
    )
    arguments = parser.parse_args()
    asked_for = set(arguments.demos) or {name.value for name in DemoName}
    reports = [
        run(ground_truth)
        for ground_truth in GROUND_TRUTHS
        if ground_truth.name in asked_for
    ]
    for report in reports:
        print(report.render(), flush=True)
    found = sum(len(report.found) for report in reports)
    expected = sum(len(report.ground_truth.expected_events) for report in reports)
    print(f"== overall: {found} of {expected} expected events found")


if __name__ == "__main__":
    main()
