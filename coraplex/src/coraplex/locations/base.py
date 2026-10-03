from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import Iterator, Iterable, Optional

from semantic_digital_twin.spatial_types.spatial_types import Pose


@dataclass
class Location(Iterable[Pose], ABC):
    """
    A region of poses the robot can be sent to, iterated as the pose candidates sampled
    from it.
    """

    number_of_samples: int = field(default=2000, kw_only=True)
    """
    How many candidates to sample.

    Far more than a caller judges properly, since a standing pose inside the furniture
    costs nothing to refuse.
    """

    seed: Optional[int] = field(default=None, kw_only=True)
    """
    Fixes the sampling, so a run can be repeated exactly.

    ``None`` samples afresh every time, which is what sampling from a map buys over
    reading it off in the order the map rates it.
    """

    @abstractmethod
    def candidates(self) -> Iterator[Pose]:
        """
        Sample pose candidates from this location, :attr:`number_of_samples` of them
        from :attr:`seed`.

        :return: The pose candidates, in the order they should be tried.
        """

    def ground(self) -> Pose:
        """
        :return: The first pose candidate of this location.
        """
        return next(iter(self))

    def __iter__(self) -> Iterator[Pose]:
        """
        :return: The candidates of this location.

        .. warning::
            Must stay a generator, so nothing is sampled before the first ``next``.
            EQL's ``variable`` calls :func:`iter` on its domain while the plan is built.
        """
        yield from self.candidates()
