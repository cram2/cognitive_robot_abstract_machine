"""
Mimics of a rule tree that is evaluated from inside another query's evaluation, through
a variable domain that runs the rule tree when it is iterated.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Iterator

from krrood.entity_query_language.query.query import Query

# %% classified case


@dataclass(unsafe_hash=True)
class Opening:
    """
    A case the rule tree classifies by its width.
    """

    width: float
    """
    The width of the opening.
    """


@dataclass(unsafe_hash=True)
class OpeningKind:
    """
    The kind the rule tree concludes for an :class:`Opening`.
    """

    opening: Opening
    """
    The opening this kind was concluded for.
    """


@dataclass(unsafe_hash=True)
class NarrowOpeningKind(OpeningKind):
    """
    The kind concluded for an opening narrower than the rule tree's threshold.
    """


@dataclass(unsafe_hash=True)
class WideOpeningKind(OpeningKind):
    """
    The kind concluded for an opening wider than the rule tree's threshold.
    """


# %% domain evaluated on iteration


@dataclass
class KindConcludedOnIteration:
    """
    A variable domain that evaluates a rule tree each time it is iterated, and yields
    the first kind it concludes.
    """

    rule_tree: Query
    """
    The rule tree whose first conclusion this domain yields.
    """

    def __iter__(self) -> Iterator[OpeningKind]:
        concluded_kind = next(iter(self.rule_tree.evaluate()), None)
        if concluded_kind is not None:
            yield concluded_kind
