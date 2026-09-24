"""
Choosing the detectors a run uses from what it is asked to detect.

Every kind of detector has to be defined before the kinds are searched, which is what
the imports of the detector modules below are for.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

from krrood.utils import recursive_subclasses
from typing_extensions import List, Self, Tuple, Type

from segmind.detectors import (  # noqa: F401
    agent_event_detector_nodes,
    atomic_event_detectors_nodes,
    coarse_event_detector_nodes,
    spatial_relation_detector_nodes,
)
from segmind.detectors.base import AbstractDetector


@dataclass(frozen=True)
class DetectorSelection:
    """
    The kinds of detector a run uses: those asked for, every kind they are read from,
    and for each of those the kind reporting that what it reports has ended.

    Asking for what is to be detected is enough. A pick-up is read from supports and
    translations, so asking for pick-ups brings their detectors along; a detector that
    can use grasps as well, but does not need them, does not bring grasps along.
    """

    detector_types: Tuple[Type[AbstractDetector], ...]
    """
    Every kind chosen, each after the kinds it is read from.
    """

    @classmethod
    def everything(cls) -> Self:
        """
        :return: Every concrete kind of detector SegMind defines, each after the kinds
            it is read from.
        """
        segmind_modules = {
            agent_event_detector_nodes.__name__,
            atomic_event_detectors_nodes.__name__,
            coarse_event_detector_nodes.__name__,
            spatial_relation_detector_nodes.__name__,
        }
        return cls.of(
            *(
                candidate
                for candidate in recursive_subclasses(AbstractDetector)
                if not inspect.isabstract(candidate)
                and candidate.__module__ in segmind_modules
            )
        )

    @classmethod
    def of(cls, *asked_for: Type[AbstractDetector]) -> Self:
        """
        :param asked_for: The kinds of detector a run is asked to use.
        :return: Those kinds and everything they need.
        """
        chosen: List[Type[AbstractDetector]] = []
        for detector_type in asked_for:
            cls._choose(detector_type, chosen)
        return cls(detector_types=tuple(chosen))

    @classmethod
    def _choose(
        cls, detector_type: Type[AbstractDetector], chosen: List[Type[AbstractDetector]]
    ) -> None:
        """
        Add ``detector_type`` to ``chosen`` after what it is read from, followed by its
        counterparts.
        """
        if detector_type in chosen:
            return
        for required in detector_type.get_required_detector_types():
            cls._choose(required, chosen)
        chosen.append(detector_type)
        for counterpart in cls._counterparts_of(detector_type):
            cls._choose(counterpart, chosen)

    @staticmethod
    def _counterparts_of(
        detector_type: Type[AbstractDetector],
    ) -> List[Type[AbstractDetector]]:
        """
        :return: The kinds reporting the end of what ``detector_type`` reports, and the
            kind reporting the beginning of what it reports the end of.
        """
        ending_it = [
            candidate
            for candidate in recursive_subclasses(AbstractDetector)
            if candidate.get_counterpart_detector_type() is detector_type
        ]
        beginning = detector_type.get_counterpart_detector_type()
        beginning_it = [] if beginning is None else [beginning]
        return ending_it + beginning_it
