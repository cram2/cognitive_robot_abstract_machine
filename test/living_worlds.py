"""
The worlds a test run leaves in memory, and the tests that created them.

A leaked world only shows up as a total, long after the test that created it finished,
so a run that just counts can name no culprit. Recording every world against the test
that was running when it was created turns that total into the list of tests to look at.
"""

from __future__ import annotations

import gc
import json
import weakref
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from krrood.exceptions import DataclassException
from typing_extensions import Any, ClassVar, List, Tuple

MAXIMUM_LIVING_WORLDS: int = 30
"""
The default budget of worlds still in memory: how many one process may still have
when a test module has finished, and separately, how many every process of a run may
add up to together - :meth:`LivingWorlds.enforce_limit` and
:meth:`WorldTallyLedger.enforce_combined_limit` each default to this same number for
their own, unrelated budget.
"""

BEFORE_THE_FIRST_TEST = "before the first test ran"
"""
Stands in for the test a world is attributed to while no test is running, such as during
collection.
"""

CREATION_ATTRIBUTE = "__new__"
"""
What a type creates its instances with, and what the record replaces to see them.
"""


@dataclass
class WorldsLeftBehind:
    """
    How many of the worlds one test created are still in memory.
    """

    test: str
    """
    Name of the test that created them.
    """

    worlds: int
    """
    How many of its worlds are still in memory.
    """


@dataclass
class WorldCreation:
    """
    A world that was created, and the test that created it.
    """

    test: str
    """
    Name of the test that was running when the world was created.
    """

    world: weakref.ReferenceType
    """
    Reference to the world that does not keep it alive, so that recording a world never
    turns into the leak it reports.
    """

    @property
    def world_is_alive(self) -> bool:
        """
        Whether the world is still in memory.
        """
        return self.world() is not None


@dataclass
class UnwatchableWorldTypeError(DataclassException, TypeError):
    """
    Raised when a type creates its instances its own way, which the record could only
    watch by taking that over.
    """

    world_type: type
    """
    The type that was to be watched.
    """

    def error_message(self) -> str:
        return (
            f"{self.world_type.__name__} creates its instances its own way rather than "
            f"with the plain {CREATION_ATTRIBUTE}, which watching it would take over."
        )

    def suggest_correction(self) -> str:
        return (
            f"Record the worlds of a type that leaves {CREATION_ATTRIBUTE} as it "
            f"inherits it, and that no other record watches already."
        )


@dataclass
class LeakedWorldsError(DataclassException, MemoryError):
    """
    Raised when a test module leaves more worlds in memory than a run allows.
    """

    module: str
    """
    Name of the test module whose end was checked.
    """

    worlds_in_memory: int
    """
    How many worlds were still in memory.
    """

    limit: int
    """
    How many of them the run allows.
    """

    left_behind: Tuple[WorldsLeftBehind, ...]
    """
    The tests whose worlds survived, the test that left the most first.
    """

    def error_message(self) -> str:
        return "\n".join(
            [
                f"{self.worlds_in_memory} worlds are still in memory when "
                f"{self.module} finished, more than the {self.limit} a test module may "
                f"leave behind.",
                "The tests that created them:",
                *(
                    f"  {left_behind.test}: {left_behind.worlds}"
                    for left_behind in self.left_behind
                ),
            ]
        )

    def suggest_correction(self) -> str:
        return (
            "The worlds are counted once the module has finished, so pytest reports this "
            "under whichever of its tests ran last; look at the tests listed above "
            "instead."
        )


@dataclass
class LivingWorlds:
    """
    The record of which test created each world, and of the worlds still in memory.

    A world is recorded as it is created, so it reaches the record however it was made:
    built, copied or read back from a file.
    """

    world_type: type
    """
    The type whose instances are recorded.
    """

    creations: List[WorldCreation] = field(default_factory=list)
    """
    One entry per world created while watching, minus the ones dropped again as their
    worlds were collected.
    """

    current_test: str = BEFORE_THE_FIRST_TEST
    """
    Name of the test the worlds created now are attributed to.
    """

    def watch(self) -> None:
        """
        Record every world the watched type creates from now on.

        :raises UnwatchableWorldTypeError: When the type creates its instances its own
            way, which watching it would take over. A type that is already watched
            creates them the record's way, so it is refused a second record too.

        ..note:: Watching cannot be undone: python leaves a type that was once given a
            :attr:`CREATION_ATTRIBUTE` dispatching through one, so taking it away again
            leaves the type unable to create anything. A watched type therefore keeps
            creating its worlds through the record for as long as the process lives,
            which costs it one reference per world.
        """
        if self.world_type.__new__ is not object.__new__:
            raise UnwatchableWorldTypeError(self.world_type)
        self.world_type.__new__ = self.create_and_record

    def create_and_record(
        self, world_type: type, *arguments: Any, **keyword_arguments: Any
    ) -> Any:
        """
        Create a world the plain way, and record it against the test running now.

        :param world_type: The type asked for a world, which is the watched type or one
            deriving from it.
        :return: The created world, for its own initialization to fill in.

        ..note:: The arguments are the ones the world is initialized with, which plain
            creation neither reads nor accepts.
        """
        world = object.__new__(world_type)
        self.record(world)
        return world

    def record(self, world: Any) -> None:
        """
        Attribute a world to the test running now.

        :param world: The world that was just created.
        """
        self.creations.append(WorldCreation(self.current_test, weakref.ref(world)))

    def forget_collected_worlds(self) -> None:
        """
        Drop the record of every world that has been collected since.
        """
        self.creations = [
            creation for creation in self.creations if creation.world_is_alive
        ]

    def surviving_worlds(self) -> Tuple[WorldsLeftBehind, ...]:
        """
        The tests whose worlds are still in memory, the test that left the most first.
        """
        worlds_per_test = Counter(creation.test for creation in self.creations)
        return tuple(
            WorldsLeftBehind(test, worlds)
            for test, worlds in worlds_per_test.most_common()
        )

    def collect_surviving_worlds(self) -> Tuple[WorldsLeftBehind, ...]:
        """
        Collect garbage, drop the worlds that have gone, and read back what survived.

        :return: The tests whose worlds are still in memory, the test that left the
            most first.
        """
        gc.collect()
        self.forget_collected_worlds()
        return self.surviving_worlds()

    def enforce_limit(self, module: str, limit: int = MAXIMUM_LIVING_WORLDS) -> None:
        """
        Report the worlds a finished test module left in memory, when there are more of
        them than it may leave.

        :param module: Name of the test module that has just finished.
        :param limit: How many worlds it may leave behind.
        :raises LeakedWorldsError: When more worlds than that survived.
        """
        left_behind = self.collect_surviving_worlds()
        worlds_in_memory = sum(entry.worlds for entry in left_behind)
        if worlds_in_memory <= limit:
            return
        raise LeakedWorldsError(
            module=module,
            worlds_in_memory=worlds_in_memory,
            limit=limit,
            left_behind=left_behind,
        )


# %% combining tallies across processes


class WorkerTallyJSONKey(StrEnum):
    """
    Field names of a :class:`WorkerTally` as written to its JSON file.
    """

    WORKER = "worker"
    LEFT_BEHIND = "left_behind"
    TEST = "test"
    WORLDS = "worlds"


@dataclass(frozen=True)
class WorkerTally:
    """
    One process's final tally of worlds still in memory when its share of the suite
    finished, written where every other process of the same run can read it back.
    """

    worker: str
    """
    Name of the process that wrote this tally: an xdist worker id (``"gw0"``, ...), or
    ``"master"`` when the run was not split across xdist workers.
    """

    left_behind: Tuple[WorldsLeftBehind, ...]
    """
    The tests whose worlds this process still held, the test that left the most first.
    """

    @property
    def worlds_in_memory(self) -> int:
        """
        How many worlds this process still held in total.
        """
        return sum(entry.worlds for entry in self.left_behind)

    def to_json(self) -> dict:
        """
        :return: This tally as a JSON-compatible mapping.
        """
        return {
            WorkerTallyJSONKey.WORKER: self.worker,
            WorkerTallyJSONKey.LEFT_BEHIND: [
                {
                    WorkerTallyJSONKey.TEST: entry.test,
                    WorkerTallyJSONKey.WORLDS: entry.worlds,
                }
                for entry in self.left_behind
            ],
        }

    @classmethod
    def from_json(cls, data: dict) -> WorkerTally:
        """
        :param data: A mapping as written by :meth:`to_json`.
        :return: The tally it describes.
        """
        return cls(
            worker=data[WorkerTallyJSONKey.WORKER],
            left_behind=tuple(
                WorldsLeftBehind(
                    test=entry[WorkerTallyJSONKey.TEST],
                    worlds=entry[WorkerTallyJSONKey.WORLDS],
                )
                for entry in data[WorkerTallyJSONKey.LEFT_BEHIND]
            ),
        )


@dataclass
class LeakedWorldsAcrossWorkersError(DataclassException, MemoryError):
    """
    Raised when the worlds still in memory across every process of a run add up to
    more than the run's combined budget, even where no single process went over its
    own share.
    """

    worlds_in_memory: int
    """
    How many worlds were still in memory, summed across every process.
    """

    limit: int
    """
    How many the run's combined budget allows.
    """

    tallies: Tuple[WorkerTally, ...]
    """
    Each process's own tally.
    """

    def error_message(self) -> str:
        ranked = sorted(
            self.tallies, key=lambda tally: tally.worlds_in_memory, reverse=True
        )
        return "\n".join(
            [
                f"{self.worlds_in_memory} worlds are still in memory across every "
                f"process when the run finished, more than the {self.limit} its "
                "combined budget allows.",
                "The processes that held them:",
                *(f"  {tally.worker}: {tally.worlds_in_memory}" for tally in ranked),
            ]
        )

    def suggest_correction(self) -> str:
        return (
            "No single process necessarily went over its own limit; look at which "
            "processes held the most above, then that process's own LeakedWorldsError "
            "reports for the tests responsible."
        )


@dataclass
class WorldTallyLedger:
    """
    Where every process of a run writes its final :class:`WorkerTally`, so whichever
    process finishes last can read every other process's tally back and enforce a
    limit on their combined total.
    """

    DIRECTORY_NAME: ClassVar[str] = ".living_worlds_tally"
    """
    Name of the directory each tally is written into and read back from, relative to
    the run's root directory so every process - however xdist started it - resolves
    the same path.
    """

    directory: Path
    """
    Directory each tally is written into and read back from, shared by every process
    of one run rather than a per-process temporary directory.
    """

    def record(self, tally: WorkerTally) -> None:
        """
        Write a process's tally where the ledger's other readers will find it.

        :param tally: The tally to record.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / f"{tally.worker}.json").write_text(
            json.dumps(tally.to_json())
        )

    def read_all(self) -> Tuple[WorkerTally, ...]:
        """
        :return: Every tally recorded so far.
        """
        if not self.directory.is_dir():
            return ()
        return tuple(
            WorkerTally.from_json(json.loads(tally_path.read_text()))
            for tally_path in sorted(self.directory.glob("*.json"))
        )

    def clear(self) -> None:
        """
        Remove every tally a previous run left behind, so this run starts from none.
        """
        if not self.directory.is_dir():
            return
        for tally_path in self.directory.glob("*.json"):
            tally_path.unlink()

    def enforce_combined_limit(self, limit: int = MAXIMUM_LIVING_WORLDS) -> None:
        """
        Report the worlds every process recorded, combined, when they add up to more
        than the run's combined budget.

        :param limit: How many worlds every process's tally may add up to across the
            whole run, whatever number of processes reported one.
        :raises LeakedWorldsAcrossWorkersError: When the combined total exceeds it.
        """
        tallies = self.read_all()
        worlds_in_memory = sum(tally.worlds_in_memory for tally in tallies)
        if worlds_in_memory <= limit:
            return
        raise LeakedWorldsAcrossWorkersError(
            worlds_in_memory=worlds_in_memory,
            limit=limit,
            tallies=tallies,
        )
