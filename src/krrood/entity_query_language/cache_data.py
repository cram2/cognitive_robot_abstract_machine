from __future__ import annotations

"""
Cache utilities.

This module provides caching datastructures and utilities.
"""
from dataclasses import dataclass, field
from typing_extensions import Dict


@dataclass
class SeenSet:
    """
    Coverage index for previously seen partial assignments.

    This replaces the linear scan with a trie-based index using a fixed key order.
    An assignment A is considered covered if there exists a stored constraint C
    such that C.items() is a subset of A.items().
    """

    keys: tuple = field(default_factory=tuple, repr=False)
    all_seen: bool = field(default=False, init=False)
    constraints: list = field(default_factory=list, init=False, repr=False)
    exact: set = field(default_factory=set, init=False, repr=False)

    def add(self, assignment: Dict) -> None:
        """
        Add a constraint (partial assignment) to the coverage index.
        """
        if self.all_seen:
            return
        if not assignment:
            # Empty constraint means everything is covered
            self.all_seen = True
            return
        # Maintain exact-match set only when all keys are present
        if self.keys and all(k in assignment for k in self.keys):
            self.exact.add(tuple(assignment[k] for k in self.keys))
        self.constraints.append(dict(assignment))

    def check(self, assignment: Dict) -> bool:
        """
        Return True if any stored constraint is a subset of the given assignment.
        Mirrors previous semantics: encountering an empty assignment flips all_seen
        but returns False the first time to allow population.
        """
        if self.all_seen:
            return True
        if not assignment:
            # First observation of empty assignment should not be considered covered
            # but should mark the index so later checks short-circuit.
            self.all_seen = True
            return False

        # Fast exact-key path when all keys are present
        if self.exact_contains(assignment):
            return True

        # Fallback to coverage check using constraints
        for constraint in self.constraints:
            if all(
                (k in assignment) and (assignment[k] == v)
                for k, v in constraint.items()
            ):
                return True
        return False

    def exact_contains(self, assignment: Dict) -> bool:
        """
        Return True if the assignment contains all cache keys and the exact key tuple
        exists in the cache. This is an O(1) membership test and does not consult
        the coverage trie.
        """
        if self.keys and all(k in assignment for k in self.keys):
            t = tuple(assignment[k] for k in self.keys)
            if t in self.exact:
                return True
        return False

    def clear(self):
        self.all_seen = False
        self.constraints.clear()
        self.exact.clear()
