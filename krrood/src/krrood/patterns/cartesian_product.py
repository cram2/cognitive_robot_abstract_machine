"""
A cartesian product that is read from its factors only as far as it is iterated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count, islice

from typing_extensions import Any, Iterable, Iterator

# %% the cartesian product by rank sum


@dataclass
class CartesianProductByRankSum:
    """
    The cartesian product of :attr:`factors`, the tuples with the lowest sum of ranks
    first, so the values every factor yields first are combined before any tuple takes a
    later one. Tuples of one rank sum come in the order of the first factor's ranks,
    then the second's, and so on.

    A factor is read only as far as a tuple reaches into it, and each of its values only
    once, so a factor may be a one-off iterator.
    """

    factors: list[Iterable[Any]]
    """
    The iterables a tuple takes one value of each, in their order.
    """

    read: list[list[Any]] = field(init=False)
    """
    The values read from each factor so far, in the order it yielded them.
    """

    _iterators: list[Iterator[Any]] = field(init=False, repr=False)
    """
    The iterator each factor is read through.
    """

    _exhausted: list[bool] = field(init=False, repr=False)
    """
    Whether each factor has no value beyond those in :attr:`read`.
    """

    def __post_init__(self) -> None:
        self._iterators = [iter(factor) for factor in self.factors]
        self.read = [[] for _ in self.factors]
        self._exhausted = [False for _ in self.factors]

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        """
        :return: The tuples, each holding one value per factor, in the order of
            :attr:`factors`.
        """
        for rank_sum in count():
            for ranks in self._ranks_summing_to(rank_sum, len(self.factors)):
                if all(self._reaches(index, rank) for index, rank in enumerate(ranks)):
                    yield tuple(
                        self.read[index][rank] for index, rank in enumerate(ranks)
                    )
            if self._has_no_tuple_beyond(rank_sum):
                return

    def _reaches(self, index: int, rank: int) -> bool:
        """
        Read the factor at `index` until it yielded its value at `rank`, or ran out.

        :param index: The position of the factor in :attr:`factors`.
        :param rank: The position of the value in the order the factor yields them.
        :return: Whether the factor has a value at `rank`.
        """
        values = self.read[index]
        while len(values) <= rank and not self._exhausted[index]:
            read_before = len(values)
            values.extend(islice(self._iterators[index], 1))
            self._exhausted[index] = len(values) == read_before
        return rank < len(values)

    def _has_no_tuple_beyond(self, rank_sum: int) -> bool:
        """
        Every factor was asked for its value at `rank_sum` while the tuples up to it
        were taken, so each one that has no further value is exhausted by then.

        :param rank_sum: The sum of ranks of the tuples taken last.
        :return: Whether no tuple has a higher sum of ranks.
        """
        if any(
            exhausted and not values
            for exhausted, values in zip(self._exhausted, self.read)
        ):
            return True
        highest_rank_sum = sum(len(values) - 1 for values in self.read)
        return all(self._exhausted) and rank_sum >= highest_rank_sum

    @classmethod
    def _ranks_summing_to(cls, total: int, parts: int) -> Iterator[tuple[int, ...]]:
        """
        :param total: What the ranks add up to.
        :param parts: How many ranks there are.
        :return: Every tuple of `parts` ranks that add up to `total`.
        """
        if parts == 0:
            if total == 0:
                yield ()
            return
        for first in range(total + 1):
            for rest in cls._ranks_summing_to(total - first, parts - 1):
                yield first, *rest
