"""
Chains of references between objects, to show that one object does not keep another
alive.
"""

from __future__ import annotations

import gc
import sys
from collections import deque
from dataclasses import dataclass

from typing_extensions import Any


# %% reference search


@dataclass(kw_only=True)
class ReferenceSearch:
    """
    A breadth-first search through the references objects hold, outwards from one
    object.
    """

    start: Any
    """
    The object whose references the search follows.
    """

    maximum_depth: int = 20
    """
    How many references a chain may follow at most.
    """

    def find_reference_chain(self, *, target: Any) -> list[Any]:
        """
        The shortest chain of objects from :attr:`start` to the target, each holding a
        reference to the next.

        :param target: The object to look for.
        :return: The chain, starting with :attr:`start` and ending with the target, or
            an empty list when the target cannot be reached.
        """
        gc.collect()
        first_referrers = self.first_referrers(target=target)
        if id(target) not in first_referrers:
            return []
        chain = [target]
        while chain[-1] is not self.start:
            chain.append(first_referrers[id(chain[-1])])
        return chain[::-1]

    def first_referrers(self, *, target: Any) -> dict[int, Any]:
        """
        The object each object the search reached was first reached from, searching
        until the target is reached.

        :param target: The object whose discovery ends the search.
        :return: The referrer of each reached object, by the reached object's id;
            :attr:`start` is mapped to itself.
        """
        first_referrers = {id(self.start): self.start}
        # Through modules and classes the search visits millions of objects; popping the
        # front of a list would cost time proportional to its length on every visit.
        queue = deque([(self.start, 0)])
        ignored = {id(first_referrers), id(queue), id(sys._getframe())}
        ignored.add(id(ignored))
        while queue:
            current, depth = queue.popleft()
            if current is target:
                return first_referrers
            if depth >= self.maximum_depth:
                continue
            for referent in gc.get_referents(current):
                if id(referent) in ignored or id(referent) in first_referrers:
                    continue
                first_referrers[id(referent)] = current
                queue.append((referent, depth + 1))
        return first_referrers
