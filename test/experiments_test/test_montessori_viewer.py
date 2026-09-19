"""
Tests for the windows the perception node draws its incoming frames in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Dict, List

from experiments.montessori.perception.viewer import (
    CameraFrameViewer,
    ImageDisplay,
    PerceptionWindow,
    colorize_depth,
    scale_to_fit,
)

# %% watching the frames as they arrive


@dataclass
class RecordingDisplay(ImageDisplay):
    """
    Stands in for a screen, remembering what it was asked to draw.
    """

    drawn: Dict[str, np.ndarray] = field(default_factory=dict)
    """
    The newest image drawn in each window, by window name.
    """

    waits: List[int] = field(default_factory=list)
    """
    How long each call to :meth:`wait` was given, in milliseconds.
    """

    closed: bool = False
    """
    Whether the windows have been taken off screen.
    """

    def draw(self, window_name: str, image: np.ndarray) -> None:
        self.drawn[window_name] = image

    def wait(self, milliseconds: int) -> None:
        self.waits.append(milliseconds)

    def close(self) -> None:
        self.closed = True


def test_unmeasured_depth_pixels_are_drawn_black():
    depth = np.array([[1.0, 0.0]], dtype=np.float32)

    colored = colorize_depth(depth)

    assert colored.shape == (1, 2, 3)
    assert colored[0, 1].tolist() == [0, 0, 0]


def test_the_nearest_and_furthest_depths_are_drawn_in_different_colours():
    depth = np.array([[0.5, 2.5]], dtype=np.float32)

    colored = colorize_depth(depth)

    assert colored[0, 0].tolist() != colored[0, 1].tolist()


def test_a_depth_image_with_nothing_measured_is_drawn_black():
    colored = colorize_depth(np.zeros((2, 3), dtype=np.float32))

    assert colored.shape == (2, 3, 3)
    assert not colored.any()


def test_an_image_smaller_than_the_window_is_drawn_at_its_own_size():
    image = np.zeros((4, 8, 3), dtype=np.uint8)

    assert scale_to_fit(image, 16, 16) is image


def test_a_wide_image_is_shrunk_to_the_window_keeping_its_proportions():
    image = np.zeros((100, 400, 3), dtype=np.uint8)

    scaled = scale_to_fit(image, 200, 200)

    assert scaled.shape == (50, 200, 3)


def test_a_tall_image_is_shrunk_to_the_window_keeping_its_proportions():
    image = np.zeros((400, 100, 3), dtype=np.uint8)

    scaled = scale_to_fit(image, 200, 200)

    assert scaled.shape == (200, 50, 3)


def test_the_viewer_draws_the_newest_frame_it_was_shown():
    display = RecordingDisplay()
    viewer = CameraFrameViewer(display=display)
    viewer.show_color(np.zeros((2, 2, 3), dtype=np.uint8))
    viewer.show_depth(np.ones((2, 2), dtype=np.float32))
    viewer.show_color(np.full((2, 2, 3), 7, dtype=np.uint8))

    viewer.refresh()

    assert display.drawn[PerceptionWindow.COLOR][0, 0].tolist() == [7, 7, 7]
    assert set(display.drawn) == {PerceptionWindow.COLOR, PerceptionWindow.DEPTH}


def test_the_rectified_view_is_drawn_in_its_own_window():
    display = RecordingDisplay()
    viewer = CameraFrameViewer(display=display)
    viewer.show_color(np.zeros((2, 2, 3), dtype=np.uint8))
    viewer.show_rectified(np.full((2, 2, 3), 5, dtype=np.uint8))

    viewer.refresh()

    assert display.drawn[PerceptionWindow.RECTIFIED][0, 0].tolist() == [5, 5, 5]
    assert set(display.drawn) == {PerceptionWindow.COLOR, PerceptionWindow.RECTIFIED}


def test_a_stream_that_has_not_arrived_leaves_only_its_own_window_empty():
    display = RecordingDisplay()
    viewer = CameraFrameViewer(display=display)
    viewer.show_color(np.zeros((2, 2, 3), dtype=np.uint8))

    viewer.refresh()

    assert set(display.drawn) == {PerceptionWindow.COLOR}


def test_the_viewer_draws_nothing_before_a_frame_has_arrived():
    display = RecordingDisplay()

    CameraFrameViewer(display=display).refresh()

    assert display.drawn == {}
    assert display.waits


def test_closing_the_viewer_takes_its_windows_off_screen():
    display = RecordingDisplay()

    CameraFrameViewer(display=display).close()

    assert display.closed
