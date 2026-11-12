"""
Required color-to-depth scaling and annotator failure handling.
"""

from __future__ import annotations

# %% Imports
import pytest
from py_trees.common import Status
from robokudo.annotators.core import Worker
from robokudo.annotators.sift_annotator import SIFTAnnotator
from robokudo.types.cv import ColorToDepthRatio
from robokudo.cas import CAS, CASViews
from robokudo.exceptions import ColorToDepthRatioMissing
from robokudo.types.scene import ObjectHypothesis
from robokudo.utils.error_handling import get_blackboard_exception
from robokudo.utils.cv_helper import get_scaled_color_image_for_depth_image
import numpy as np


# %% Required ratio
@pytest.mark.parametrize("views", [{}, {CASViews.COLOR_TO_DEPTH_RATIO: None}])
def test_missing_color_to_depth_ratio_raises(views: dict[CASViews, None]) -> None:
    """
    Absent and unset scaling raise the same precondition exception.
    """
    with pytest.raises(ColorToDepthRatioMissing):
        _ = CAS(views=views).color_to_depth_ratio


def test_color_to_depth_ratio_returns_stored_value() -> None:
    """
    Scaling access preserves the value assigned by the producer.
    """
    cas = CAS()
    cas.color_to_depth_ratio = ColorToDepthRatio(horizontal=0.5, vertical=0.5)
    assert cas.color_to_depth_ratio == cas.views[CASViews.COLOR_TO_DEPTH_RATIO]


# %% Annotator failure
@pytest.mark.parametrize("views", [{}, {CASViews.COLOR_TO_DEPTH_RATIO: None}])
def test_sift_missing_ratio_records_failure(
    views: dict[CASViews, None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Missing scaling fails SIFT and records the specific exception.
    """
    cas = CAS(views=views)
    cas.annotations.append(ObjectHypothesis())
    annotator = SIFTAnnotator()
    monkeypatch.setattr(annotator, "get_cas", lambda: cas)
    assert annotator.compute() == Status.FAILURE
    assert isinstance(get_blackboard_exception(), ColorToDepthRatioMissing)


@pytest.mark.parametrize("views", [{}, {CASViews.COLOR_TO_DEPTH_RATIO: None}])
def test_image_scaling_missing_ratio_raises(views: dict[CASViews, None]) -> None:
    """
    Image scaling reports missing scaling through the CAS exception.
    """
    with pytest.raises(ColorToDepthRatioMissing):
        get_scaled_color_image_for_depth_image(
            CAS(views=views), np.zeros((2, 2, 3), dtype=np.uint8)
        )


def test_sift_worker_missing_ratio_returns_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The SIFT worker returns failure rather than propagating a missing-ratio exception.
    """
    cas = CAS()
    cas.annotations.append(ObjectHypothesis())
    annotator = SIFTAnnotator()
    monkeypatch.setattr(annotator, "get_cas", lambda: cas)
    worker = Worker(annotator.compute)
    thread = worker.start()
    thread.join()
    assert worker.future.result() == Status.FAILURE
    assert isinstance(get_blackboard_exception(), ColorToDepthRatioMissing)
