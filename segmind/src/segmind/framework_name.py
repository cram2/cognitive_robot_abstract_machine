"""
The name of the framework that ticks the nodes of segmind.
"""

from __future__ import annotations

from enum import StrEnum


class SegmentationFramework(StrEnum):
    """
    The frameworks of segmind, as :attr:`~cramph.node.StatechartNode.ticking_framework`
    names them.
    """

    SEGMIND = "segmind"
    """
    Ticks the detectors that segment an episode, see
    :class:`~segmind.detectors.base.AbstractDetector`.
    """
