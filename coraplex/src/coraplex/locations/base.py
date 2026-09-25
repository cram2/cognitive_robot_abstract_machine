from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Iterator, Iterable

from coraplex.locations.sampling import CandidateDraw
from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class Location(Iterable[Pose], ABC):
    """
    A region of poses the robot can be sent to, iterated as the pose candidates drawn
    from it.
    """

    draw: CandidateDraw = field(default_factory=CandidateDraw, kw_only=True)
    """
    The terms this location's candidates are drawn on.
    """

    @abstractmethod
    def candidates(self, draw: CandidateDraw) -> Iterator[Pose]:
        """
        Draw pose candidates from this location.

        Every location says what it does with the terms it is given, so none of them is
        chosen on a caller's behalf.

        :param draw: The terms to draw the candidates on.
        :return: The pose candidates, in the order they should be tried.
        """

    def ground(self) -> Pose:
        """
        :return: The first pose candidate of this location.
        """
        return next(iter(self))

    def __iter__(self) -> Iterator[Pose]:
        """
        :return: The candidates drawn on :attr:`draw`.

        .. warning::
            Must stay a generator, so nothing is drawn before the first ``next``. EQL's
            ``variable`` calls :func:`iter` on its domain while the plan is built.
        """
        yield from self.candidates(self.draw)
