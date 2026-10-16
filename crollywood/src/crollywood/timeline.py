"""
A video as scenes laid end to end.

A scene is the part a video plays in a timeline: any video, whether a slide, footage or
another timeline, can be one. The timeline lays its scenes one after another, dissolving
or cutting from each into the next, and reads the whole out as frames at one rate.
Nothing here knows what a scene shows, so a title, a chart being drawn and a recording
being replayed are all laid out the same way.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto

from krrood.patterns.role import Role
from typing_extensions import Iterator

from crollywood.exceptions import FramesOfDifferentSizes, MomentOutsideScene
from crollywood.frame import Frame, Resolution
from crollywood.video import Video

# %% fading


@dataclass(frozen=True)
class FrameFade:
    """
    A change that starts and ends slowly over a while.
    """

    duration: timedelta
    """
    How long the change takes.
    """

    def weight_at(self, elapsed: timedelta) -> float:
        """
        :param elapsed: How long the change has been going on; before it starts counts
            as its start and after it ends as its end.
        :return: How far along the change is, from zero to one, along a cosine ramp.
        """
        clamped = min(max(elapsed / self.duration, 0.0), 1.0)
        return 0.5 - 0.5 * math.cos(math.pi * clamped)


# %% one scene


class Transition(Enum):
    """
    How the scene before passes into a scene.
    """

    DISSOLVE = auto()
    """
    The scene before fades into it while the two overlap.
    """

    CUT = auto()
    """
    It replaces the scene before at once, as where the two would write different text in
    one place.
    """


@dataclass(eq=False)
class TimelineScene(Role[Video]):
    """
    The part a video plays in a timeline: one stretch of it, joined to the stretch
    before.
    """

    transition: Transition = Transition.DISSOLVE
    """
    How the scene before passes into this one.
    """

    def frame_at(self, moment: timedelta) -> Frame:
        """
        :param moment: A moment of the scene's own time, from its start.
        :return: What the scene shows then.
        :raises MomentOutsideScene: If the moment lies before the scene starts or after
            it ends.
        """
        if not timedelta(0) <= moment < self.role_taker.duration:
            raise MomentOutsideScene(moment=moment, duration=self.role_taker.duration)
        return self.role_taker.frame_at(moment)

    def progress_at(self, moment: timedelta) -> float:
        """
        :param moment: A moment of the scene's own time.
        :return: How far through the scene that moment is, from zero to one.
        """
        return moment / self.role_taker.duration


# %% what is drawn over every scene


@dataclass
class Overlay(ABC):
    """
    Something drawn over every frame of a video, whatever scene plays: it knows the
    moment of the whole video rather than of one scene.
    """

    @abstractmethod
    def drawn_over(self, frame: Frame, moment: timedelta, runs_for: timedelta) -> Frame:
        """
        :param frame: The frame of the scenes at that moment.
        :param moment: The moment, from the video's start.
        :param runs_for: How long the video lasts.
        :return: A copy of the frame with this drawn over it.
        """


# %% the timeline


@dataclass
class TimelineStyle:
    """
    How a timeline plays its scenes: the rate it reads them out at, how long each
    dissolves into the next, and what is drawn over them all.
    """

    frames_per_second: int = 25
    """
    How many frames one second of the video holds.
    """

    dissolve: timedelta = timedelta(seconds=0.5)
    """
    How long a scene takes to dissolve into the next; the two overlap by that much.
    """

    overlays: list[Overlay] = field(default_factory=list)
    """
    What is drawn over every frame, in order, after the scenes.
    """


@dataclass
class VideoTimeline(Video):
    """
    Scenes laid end to end, each dissolving or cutting into the next, read out as
    frames.
    """

    scenes: list[TimelineScene]
    """
    The scenes, in the order they play.
    """

    style: TimelineStyle = field(default_factory=TimelineStyle)
    """
    How the scenes are played.
    """

    @property
    def duration(self) -> timedelta:
        """
        How long the whole video lasts.
        """
        overlaps = sum(
            (self.dissolve_into(index) for index in range(1, len(self.scenes))),
            timedelta(0),
        )
        return sum((scene.duration for scene in self.scenes), timedelta(0)) - overlaps

    def dissolve_into(self, index: int) -> timedelta:
        """
        :param index: Which scene, from zero.
        :return: How long the scene before dissolves into it: no time where it cuts in.
        """
        if index > 0 and self.scenes[index].transition is Transition.DISSOLVE:
            return self.style.dissolve
        return timedelta(0)

    @property
    def frame_count(self) -> int:
        """
        How many frames the video is read out as.
        """
        return round(self.duration.total_seconds() * self.style.frames_per_second)

    def starts(self) -> list[timedelta]:
        """
        :return: When each scene starts, from the video's start.
        """
        starts = []
        start = timedelta(0)
        for index, scene in enumerate(self.scenes):
            starts.append(start)
            start += scene.duration - (
                self.dissolve_into(index + 1)
                if index + 1 < len(self.scenes)
                else timedelta(0)
            )
        return starts

    def frame_at(self, moment: timedelta) -> Frame:
        """
        :param moment: The moment, from the video's start.
        :return: What the video shows then: its scenes, with every overlay drawn over
            them.
        """
        frame = self.scenes_at(moment)
        runs_for = self.duration
        for overlay in self.style.overlays:
            frame = overlay.drawn_over(frame, moment, runs_for)
        return frame

    def scenes_at(self, moment: timedelta) -> Frame:
        """
        :param moment: The moment, from the video's start.
        :return: What the scenes show then: the scene playing blended with the one
            before it while the two dissolve.
        """
        starts = self.starts()
        current = max(
            index for index, start in enumerate(starts) if start <= moment or index == 0
        )
        scene = self.scenes[current]
        last_moment = scene.duration - timedelta(microseconds=1)
        local = min(moment - starts[current], last_moment)
        frame = scene.frame_at(max(local, timedelta(0)))
        dissolve = self.dissolve_into(current)
        if current == 0 or local >= dissolve:
            return frame
        previous = self.scenes[current - 1]
        earlier = previous.frame_at(
            min(
                moment - starts[current - 1],
                previous.duration - timedelta(microseconds=1),
            )
        )
        self._same_size(earlier, frame)
        return earlier.blended(frame, FrameFade(dissolve).weight_at(local))

    def frames(self) -> Iterator[Frame]:
        """
        Every frame of the video, in order.

        :return: The frames.
        :raises FramesOfDifferentSizes: If two scenes draw frames of different sizes.
        """
        size: Resolution | None = None
        for index in range(self.frame_count):
            frame = self.frame_at(
                timedelta(seconds=index / self.style.frames_per_second)
            )
            if size is None:
                size = frame.resolution
            self._same_size(size.blank(), frame)
            yield frame

    @staticmethod
    def _same_size(first: Frame, second: Frame) -> None:
        """
        :param first: A frame.
        :param second: Another.
        :raises FramesOfDifferentSizes: If the two frames differ in size.
        """
        if first.resolution != second.resolution:
            raise FramesOfDifferentSizes(
                first=first.resolution, second=second.resolution
            )
