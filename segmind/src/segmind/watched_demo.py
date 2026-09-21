"""
A demo watched by SegMind: what it watches, what it is asked to detect, and what it
shows of it while it runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing_extensions import (
    List,
    Optional,
    Self,
    Sequence,
    TYPE_CHECKING,
    Type,
)

from segmind import event_logger
from segmind.detectors.base import AbstractDetector
from segmind.live_segmenter import LiveSegmenter
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

if TYPE_CHECKING:
    from segmind.dashboard.server import LiveEventDashboard

logger = logging.getLogger(__name__)


@dataclass
class WatchedDemo:
    """
    SegMind watching a demo while its plan runs.

    Entering it starts watching and leaving it stops, so a demo states what it wants
    watched and what it wants detected, and keeps the rest to its plan.
    """

    world: World
    """
    The world the demo runs in.
    """

    bodies: List[Body]
    """
    The bodies the demo's plan handles, which the detectors watch.
    """

    detectors: Sequence[Type[AbstractDetector]] = ()
    """
    The kinds of detector asked for; every kind they are read from is brought along.
    Without any, everything SegMind can detect.
    """

    show_live_events: bool = False
    """
    Whether the events are served as a page while the demo runs. It is segmind's
    ``dashboard`` extra, so nothing here needs flask while this is off.
    """

    segmenter: LiveSegmenter = field(init=False)
    """
    Ticks the detectors against the world while the demo runs.
    """

    dashboard: Optional[LiveEventDashboard] = field(init=False, default=None)
    """
    Serves the page, between entering and leaving a demo that asks for one.
    """

    def __post_init__(self) -> None:
        self.segmenter = LiveSegmenter.watching(self.world, self.bodies, self.detectors)

    @classmethod
    def watching_bodies_named(
        cls,
        world: World,
        names: Sequence[str],
        detectors: Sequence[Type[AbstractDetector]] = (),
        show_live_events: bool = False,
    ) -> Self:
        """
        A watch over the bodies a demo names.

        :param world: The world the demo runs in.
        :param names: The names of the bodies to watch.
        :param detectors: The kinds of detector asked for.
        :param show_live_events: Whether to serve the events as a page while it runs.
        """
        return cls(
            world=world,
            bodies=[world.get_body_by_name(name) for name in names],
            detectors=detectors,
            show_live_events=show_live_events,
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
        Put every event detected on the console, which is where a demo is read.

        SegMind reports what it detected at debug level, which nothing shows by default.
        """
        detected_events = logging.getLogger(event_logger.__name__)
        detected_events.setLevel(logging.DEBUG)
        if not detected_events.handlers:
            detected_events.addHandler(logging.StreamHandler())
        self.segmenter.event_logger.print_events()

    def __enter__(self) -> Self:
        logger.info("SegMind detectors: %s", ", ".join(self.detector_names))
        if self.show_live_events:
            from segmind.dashboard.server import LiveEventDashboard

            self.dashboard = LiveEventDashboard.watching(self.segmenter)
            self.dashboard.start()
        self.segmenter.start()
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.segmenter.stop()
        if self.dashboard is not None:
            self.dashboard.stop()
            self.dashboard = None
        self.report_detected_events()
