"""
SegMind itself: what it watches and what it is asked to detect while a run goes on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing_extensions import List, Self, Sequence, Type

from segmind import event_logger
from segmind.detectors.base import AbstractDetector
from segmind.live_segmenter import LiveSegmenter
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

logger = logging.getLogger(__name__)


@dataclass
class Segmind:
    """
    SegMind segmenting a run into events while it goes on.

    Entering it starts the detectors and leaving it stops them and reports what they
    detected, so a run states what it wants watched and what it wants detected, and
    keeps the rest to its plan.
    """

    world: World
    """
    The world the run takes place in.
    """

    bodies: List[Body]
    """
    The bodies the run handles, which the detectors watch.
    """

    detectors: Sequence[Type[AbstractDetector]] = ()
    """
    The kinds of detector asked for; every kind they are read from is brought along.
    Without any, everything SegMind can detect.
    """

    segmenter: LiveSegmenter = field(init=False)
    """
    Ticks the detectors against the world while the run goes on.
    """

    def __post_init__(self) -> None:
        self.segmenter = LiveSegmenter.watching(self.world, self.bodies, self.detectors)

    @classmethod
    def watching_bodies_named(
        cls,
        world: World,
        names: Sequence[str],
        detectors: Sequence[Type[AbstractDetector]] = (),
    ) -> Self:
        """
        SegMind watching the bodies a run names.

        :param world: The world the run takes place in.
        :param names: The names of the bodies to watch.
        :param detectors: The kinds of detector asked for.
        """
        return cls(
            world=world,
            bodies=[world.get_body_by_name(name) for name in names],
            detectors=detectors,
        )

    @property
    def detector_names(self) -> List[str]:
        """
        The kinds of detector ticked, each named once, in the order they are ticked.
        """
        return list(
            dict.fromkeys(
                type(detector).__name__ for detector in self.segmenter.detectors
            )
        )

    def report_detected_events(self) -> None:
        """
        Put every event detected on the console, which is where a run is read.

        SegMind reports what it detected at debug level, which nothing shows by default,
        and a run that already has a handler of its own would otherwise show each event
        through both.
        """
        detected_events = logging.getLogger(event_logger.__name__)
        detected_events.setLevel(logging.DEBUG)
        if not detected_events.handlers:
            detected_events.addHandler(logging.StreamHandler())
        propagated = detected_events.propagate
        detected_events.propagate = False
        self.segmenter.event_logger.print_events()
        detected_events.propagate = propagated

    def __enter__(self) -> Self:
        logger.info("SegMind detectors: %s", ", ".join(self.detector_names))
        self.segmenter.start()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.segmenter.stop()
        self.report_detected_events()
