"""
Mimics of a blocksworld whose blocks rest on each other, on a table or in the slots of a
shelf, for evaluating queries against a planning state.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.entity_query_language.predicate import (
    Predicate,
    SymbolicFunction,
    symbolic_function,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Prepositions
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    FunctionVerbalizationTemplates,
    Noun,
    Verb,
    clause,
)

# %% blocks, the table and the shelf


@dataclass(eq=False)
class Table:
    """
    The surface every tower stands on.

    Compared by identity, so it can be passed to a method call in a query.
    """

    name: str
    """
    The name of the table.
    """


@dataclass
class Block:
    """
    A block resting on another block or on the table.

    Compared by value, so two blocks with the same name and support are equal but remain
    distinct objects.
    """

    name: str
    """
    The name of the block.
    """

    support: Block | Table
    """
    What the block rests on.
    """

    def rests_on(self, support: Block | Table, directly: bool = True) -> bool:
        """
        :param support: The block or table to check.
        :param directly: Whether *support* must be what this block rests on, rather than
            anything below it in its tower.
        :return: Whether this block rests on *support*.
        """
        if self.support is support:
            return True
        if directly or not isinstance(self.support, Block):
            return False
        return self.support.rests_on(support, directly=False)


@dataclass
class Shelf:
    """
    A shelf whose slots each hold a block.
    """

    name: str
    """
    The name of the shelf.
    """

    slots: list[Block]
    """
    The block in each slot, in order.
    """


# %% derived facts


@symbolic_function
def tower_height(block: Block) -> int:
    """
    :param block: The top block of the tower.
    :return: The number of blocks from *block* down to the table.
    """
    if isinstance(block.support, Block):
        return 1 + tower_height(block.support)
    return 1


@symbolic_function
def tallest_tower(tops: list[Block]) -> int:
    """
    :param tops: The top blocks of some towers.
    :return: The height of the tallest of those towers.
    """
    return max(tower_height(top) for top in tops)


@symbolic_function
def is_table(support: Block | Table) -> bool:
    """
    :param support: What a block rests on.
    :return: Whether *support* is a table.
    """
    return isinstance(support, Table)


@dataclass(eq=False)
class IsOn(Predicate):
    """
    A block resting directly on a support.
    """

    block: Block
    """
    The block that rests.
    """

    support: Block | Table
    """
    What it rests on.
    """

    def __call__(self) -> bool:
        return self.block.support is self.support

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(
            Noun(fields["block"]),
            Verb("rest"),
            Prepositions.ON,
            Noun(fields["support"]),
        )


@dataclass(eq=False)
class SupportName(SymbolicFunction):
    """
    The name of what a block rests on.
    """

    block: Block
    """
    The block whose support is named.
    """

    def __call__(self) -> str:
        return self.block.support.name

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return FunctionVerbalizationTemplates.possessive(cls, *fields.values())
