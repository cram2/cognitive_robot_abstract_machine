"""
How a rated set of pose candidates is drawn from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Optional


@dataclass
class CostmapSamplingStrategy(ABC):
    """
    How a costmap picks which of its entries to offer as candidates.

    A costmap rates every entry it holds; a strategy decides what that rating is used
    for.
    """

    def choose(self, ratings: NDArray[np.float64], count: int) -> NDArray[np.intp]:
        """
        Pick which entries to offer.

        Offers fewer than asked for when the map holds fewer entries this strategy can
        reach, since an entry is only ever offered once.

        :param ratings: The flattened costmap, one rating per entry.
        :param count: How many entries to pick at most.
        :return: The indices to offer, in the order they should be offered.
        """
        offerable = min(count, self.count_offerable_entries(ratings))
        if offerable <= 0:
            return np.empty(0, dtype=np.intp)
        return self._pick(ratings, offerable)

    def count_offerable_entries(self, ratings: NDArray[np.float64]) -> int:
        """
        How many of the given entries this strategy can offer.

        :param ratings: The flattened costmap, one rating per entry.
        """
        return ratings.size

    @abstractmethod
    def _pick(self, ratings: NDArray[np.float64], count: int) -> NDArray[np.intp]:
        """
        Pick exactly ``count`` entries, never more than this strategy can offer.

        :param ratings: The flattened costmap, one rating per entry.
        :param count: How many entries to pick.
        :return: The indices to offer, in the order they should be offered.
        """


@dataclass
class HighestRatedFirst(CostmapSamplingStrategy):
    """
    Offers the highest rated entries first, in order, leaving nothing to chance.

    Lets a caller take the first candidate that passes its own checks. A caller that can
    only afford to judge a handful never sees past what the map rates highest, though --
    for a ring, that is its own radius, one angle at a time.
    """

    def _pick(self, ratings: NDArray[np.float64], count: int) -> NDArray[np.intp]:
        highest = np.argpartition(ratings, -count)[-count:]
        return highest[np.argsort(ratings[highest])[::-1]]


@dataclass
class RandomCostmapSamplingStrategy(CostmapSamplingStrategy, ABC):
    """
    Base for the strategies that draw entries at random.
    """

    seed: Optional[int] = field(default=None, kw_only=True)
    """
    Fixes the draw, so a run can be repeated exactly.

    ``None`` draws afresh every time.
    """

    random_generator: np.random.Generator = field(init=False, repr=False)
    """
    Source of randomness, kept off numpy's global state so one draw cannot disturb
    another.
    """

    def __post_init__(self) -> None:
        self.random_generator = np.random.default_rng(self.seed)


@dataclass
class WeightedByRating(RandomCostmapSamplingStrategy):
    """
    Draws entries at random, an entry's rating being its chance of being drawn.

    Treats the map as the distribution its shape describes, so what it rates highest is
    merely likeliest and the rest of the region still comes up.
    """

    def count_offerable_entries(self, ratings: NDArray[np.float64]) -> int:
        """
        How many of the given entries this strategy can offer.

        An entry rated zero stands no chance of being drawn, so only the rated ones can
        be offered -- unless the map rates nothing at all, which is drawn from evenly.

        :param ratings: The flattened costmap, one rating per entry.
        """
        return int(np.count_nonzero(ratings)) or ratings.size

    def _pick(self, ratings: NDArray[np.float64], count: int) -> NDArray[np.intp]:
        if not ratings.any():
            return self.random_generator.choice(ratings.size, count, replace=False)
        return self.random_generator.choice(
            ratings.size, count, replace=False, p=ratings / ratings.sum()
        )


@dataclass
class UniformlyAtRandom(RandomCostmapSamplingStrategy):
    """
    Draws entries at random, all of them equally likely, ignoring their ratings.

    For a caller that wants the region covered rather than the part of it the map rates
    highest.
    """

    def _pick(self, ratings: NDArray[np.float64], count: int) -> NDArray[np.intp]:
        return self.random_generator.choice(ratings.size, count, replace=False)
