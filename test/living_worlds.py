"""
The worlds a test run leaves in memory, and the tests that created them.

A leaked world only shows up as a total, long after the test that created it finished,
so a run that just counts can name no culprit. Recording every world against the test
that was running when it was created turns that total into the list of tests to look at.
"""

from __future__ import annotations

import gc
import weakref
from collections import Counter
from dataclasses import dataclass, field

from krrood.exceptions import DataclassException
from typing_extensions import Any, List, Tuple

MAXIMUM_LIVING_WORLDS: int = 30
"""
The default number of worlds one process may still have in memory when a test module
has finished.
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
