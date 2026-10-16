"""
What is said over the video, and when.

A line of narration belongs to the scene it starts with and may run on into the scenes
that follow, as long as the next line has not started by then. A slide grows to hold its
line; a close-up keeps the length of what it shows, so a line that does not fit one is
refused rather than talked over.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
from krrood.patterns.role import Role
from typing_extensions import Sequence

from crollywood.exceptions import NarrationOverrun
from crollywood.speech import SoundChannel, TextToSpeech
from crollywood.subtitles import SubRipFile, SubtitleCue, SubtitleRoom
from crollywood.slides import HeldSlide
from crollywood.timeline import TimelineScene, TimelineStyle, VideoTimeline

# %% a line and its speech


@dataclass(frozen=True)
class NarratedLine:
    """
    One thing the narrator says.
    """

    written: str
    """
    The line as the subtitles show it.
    """

    spoken: str = ""
    """
    The line as the voice says it, where that differs from how it is written; numbers
    are said the way people say them.

    Empty when the two agree.
    """

    pace: float = 1.0
    """
    How fast the line is said, as a share of the voice's own speed: below one for a line
    the viewer needs time to follow.
    """

    @property
    def said(self) -> str:
        """
        What the voice is given to say.
        """
        return self.spoken or self.written

    def spoken_by(self, voice: TextToSpeech) -> SoundChannel:
        """
        :param voice: What says the line.
        :return: The line, said at its pace.
        """
        return voice.speaks(self.said, self.pace)

    @staticmethod
    def start_times(
        lines: Sequence[NarratedLine], voice: TextToSpeech, pause: timedelta
    ) -> list[timedelta]:
        """
        :param lines: The lines, in order.
        :param voice: What says the lines.
        :param pause: How long it is between one line and the next.
        :return: How long after the first of the lines starts each of them starts, said
            one after the other: for a scene to time itself to its lines.
        """
        starts: list[timedelta] = []
        at = timedelta(0)
        for line in lines:
            starts.append(at)
            at += line.spoken_by(voice).duration + pause
        return starts


# %% a line placed in time


@dataclass(frozen=True)
class SpokenLine:
    """
    A line, said, and placed where it is heard.
    """

    line: NarratedLine
    """
    The line.
    """

    starts: timedelta
    """
    When it is heard, from the video's start.
    """

    speech: SoundChannel
    """
    The line as sound.
    """

    @property
    def ends(self) -> timedelta:
        """
        When it has been said.
        """
        return self.starts + self.speech.duration

    def cues(self, room: SubtitleRoom | None = None) -> list[SubtitleCue]:
        """
        :param room: How much text one subtitle may show; a player's one row if not
            given.
        :return: The line cut into subtitles, each short enough for the room, each shown
            for its share of the speech by its share of the letters.
        """
        fitting = room if room is not None else SubtitleRoom.of_player()
        pieces = fitting.pieces(self.line.written)
        letters = sum(len(piece) for piece in pieces)
        cues: list[SubtitleCue] = []
        start = self.starts
        for piece in pieces:
            end = start + self.speech.duration * len(piece) / letters
            cues.append(SubtitleCue(start, end, piece))
            start = end
        return cues


# %% every line, one after the other


@dataclass(frozen=True)
class Soundtrack(SoundChannel):
    """
    The whole narration as one stretch of sound, silent where nothing is said.
    """


@dataclass
class Narration:
    """
    Every line of the video, placed in time.
    """

    lines: list[SpokenLine]
    """
    The lines, in order.
    """

    silent_sample_rate: int = 24_000
    """
    Samples per second of a soundtrack nothing is said over, in hertz: the rate the
    Kokoro model speaks at.
    """

    def check(self, runs_for: timedelta) -> None:
        """
        Refuse a line that runs into the next, or past the video's end.

        :param runs_for: How long the video lasts.
        :raises NarrationOverrun: For the first line that does.
        """
        for line, following in zip(self.lines, self.lines[1:]):
            if line.ends > following.starts:
                raise NarrationOverrun(line.line, line.ends - following.starts)
        if self.lines and self.lines[-1].ends > runs_for:
            raise NarrationOverrun(self.lines[-1].line, self.lines[-1].ends - runs_for)

    def soundtrack(self, runs_for: timedelta) -> Soundtrack:
        """
        :param runs_for: How long the video lasts.
        :return: The lines laid where they start, on silence.
        """
        sample_rate = (
            self.lines[0].speech.sample_rate if self.lines else self.silent_sample_rate
        )
        track = np.zeros(
            round(runs_for.total_seconds() * sample_rate), dtype=np.float32
        )
        for line in self.lines:
            first = round(line.starts.total_seconds() * sample_rate)
            last = min(first + len(line.speech.samples), len(track))
            track[first:last] = line.speech.samples[: last - first]
        return Soundtrack(track, sample_rate)

    def report(self, runs_for: timedelta) -> str:
        """
        :param runs_for: How long the video lasts.
        :return: One row per line: when it starts, how long it is said, and the silence
            before the next line or the video's end (negative where it runs over).
        """
        rows = []
        for line, next_start in zip(
            self.lines, [each.starts for each in self.lines[1:]] + [runs_for]
        ):
            free = (next_start - line.ends).total_seconds()
            rows.append(
                f"{line.starts.total_seconds():6.1f} s  "
                f"{line.speech.duration.total_seconds():5.1f} s said  "
                f"{free:+6.1f} s free   {line.line.written[:60]}"
            )
        return "\n".join(rows)

    def cues(self, room: SubtitleRoom | None = None) -> list[SubtitleCue]:
        """
        :param room: How much text one subtitle may show; a player's one row if not
            given.
        :return: Every subtitle, in order.
        """
        return [cue for line in self.lines for cue in line.cues(room)]

    def written_to(self, path: Path, room: SubtitleRoom | None = None) -> Path:
        """
        :param path: Where the SubRip file goes.
        :param room: How much text one subtitle may show; a player's one row if not
            given.
        :return: The path.
        """
        fitting = room if room is not None else SubtitleRoom.of_player()
        path.write_text(SubRipFile(self.cues(fitting), fitting).text, encoding="utf-8")
        return path


@dataclass(eq=False)
class NarratedTimeline(Role[VideoTimeline]):
    """
    A timeline with what is said over it, placed on it.
    """

    narration: Narration
    """
    The lines, said and placed where their scenes start.
    """

    def check(self) -> None:
        """
        Refuse a line that runs into the next, or past the video's end.

        :raises NarrationOverrun: For the first line that does.
        """
        self.narration.check(runs_for=self.role_taker.duration)

    def soundtrack(self) -> Soundtrack:
        """
        :return: The narration as sound as long as the video, silent where nothing is
            said.
        """
        return self.narration.soundtrack(runs_for=self.role_taker.duration)


# %% the scenes with their lines


@dataclass(eq=False)
class NarratedScene(TimelineScene):
    """
    A scene and the lines that start with it, if any do.
    """

    lines: tuple[NarratedLine, ...] = ()
    """
    The lines that start with it, said one after the other, if any do.
    """

    delay: timedelta = timedelta(0)
    """
    How long into the scene the line starts, beyond the lead every line has.
    """

    waits: tuple[timedelta, ...] = ()
    """
    How long each line after the first waits beyond the pause, for what it speaks of to
    come up; nothing for every line to follow the one before at once.
    """

    runs_on: bool = False
    """
    Whether the line may run on into the scenes that follow instead of the scene growing
    to hold it; a scene that is not a slide never grows, whatever this says.
    """

    def wait_before(self, number: int) -> timedelta:
        """
        :param number: Which of the lines, from zero.
        :return: How long it waits beyond the pause after the line before it.
        """
        return self.waits[number - 1] if 0 < number <= len(self.waits) else timedelta(0)


@dataclass
class Storyboard:
    """
    The scenes in order, each with the line that starts with it.
    """

    scenes: list[NarratedScene]
    """
    The scenes, in the order they play.
    """

    tail: timedelta = timedelta(seconds=0.6)
    """
    How long a slide is held after its line has ended.
    """

    pause: timedelta = timedelta(seconds=0.5)
    """
    How long it is between two lines said over the same scene.
    """

    lead: timedelta = timedelta(seconds=0.3)
    """
    How long a line starts after its scene has, so it is not heard over the dissolve in.
    """

    def fitted_to(self, voice: TextToSpeech) -> None:
        """
        Grow every held scene to hold the lines that start with it.

        :param voice: What says the lines.
        """
        for each in self.scenes:
            if (
                not each.lines
                or each.runs_on
                or not isinstance(each.role_taker, HeldSlide)
            ):
                continue
            said = sum(
                (line.spoken_by(voice).duration for line in each.lines), timedelta(0)
            )
            needed = (
                self.lead
                + each.delay
                + said
                + self.pause * (len(each.lines) - 1)
                + sum(each.waits, timedelta(0))
                + self.tail
            )
            each.role_taker.held_for = max(each.role_taker.held_for, needed)

    def narrated_by(self, voice: TextToSpeech, style: TimelineStyle) -> Narration:
        """
        :param voice: What says the lines.
        :param style: How the scenes are played.
        :return: The lines said and placed where their scenes start, as the timeline
            puts them.
        """
        spoken: list[SpokenLine] = []
        for each, start in zip(
            self.scenes, VideoTimeline(list(self.scenes), style).starts()
        ):
            at = start + self.lead + each.delay
            for number, line in enumerate(each.lines):
                spoken.append(
                    SpokenLine(
                        line, at + each.wait_before(number), line.spoken_by(voice)
                    )
                )
                at = spoken[-1].ends + self.pause
        return Narration(spoken)
