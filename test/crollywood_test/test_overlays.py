"""
Tests for :mod:`crollywood.overlays`: the progress bar drawn over the whole video.
"""

from __future__ import annotations


from datetime import timedelta

import numpy as np
import pytest

from crollywood.frame import Frame, Resolution
from crollywood.overlays import ProgressBar
from crollywood.theme import VideoTheme
from crollywood.slides import SingleFrameSlide
from crollywood.timeline import TimelineScene, TimelineStyle, VideoTimeline

FRAME_SIZE = Resolution(width=200, height=100)
RUNS_FOR = timedelta(seconds=4)


@pytest.mark.parametrize("played_seconds", [0.0, 1.0, 2.0, 4.0])
def test_the_progress_bar_fills_the_top_of_the_frame_as_far_as_the_video_has_played(
    played_seconds: float,
) -> None:
    theme = VideoTheme.dark()
    bar = ProgressBar(theme=theme)
    drawn = bar.drawn_over(
        theme.page_of(FRAME_SIZE), timedelta(seconds=played_seconds), RUNS_FOR
    ).pixels
    played = round(FRAME_SIZE.width * played_seconds / RUNS_FOR.total_seconds())
    assert (drawn[: bar.thickness, :played] == tuple(theme.accent.to_rgb_bytes())).all()
    assert (drawn[: bar.thickness, played:] == tuple(theme.panel.to_rgb_bytes())).all()
    assert (drawn[bar.thickness :] == tuple(theme.page.to_rgb_bytes())).all()


def test_the_progress_bar_runs_on_across_the_scenes_of_a_timeline() -> None:
    theme = VideoTheme.dark()
    page = theme.page_of(FRAME_SIZE)
    hold = timedelta(seconds=2)
    moment = timedelta(seconds=3)
    timeline = VideoTimeline(
        [
            TimelineScene(role_taker=SingleFrameSlide(page, held_for=hold)),
            TimelineScene(role_taker=SingleFrameSlide(page, held_for=hold)),
        ],
        style=TimelineStyle(
            frames_per_second=10,
            dissolve=timedelta(0),
            overlays=[ProgressBar(theme=theme)],
        ),
    )
    top_row = timeline.frame_at(moment).pixels[0]
    accent_columns = np.flatnonzero(
        (top_row == tuple(theme.accent.to_rgb_bytes())).all(axis=1)
    )
    assert len(accent_columns) == round(FRAME_SIZE.width * moment / timeline.duration)
