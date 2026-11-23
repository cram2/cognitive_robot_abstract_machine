"""
Mimics of a blocksworld whose blocks are symbols, with action schemas that stack them,
put them on trays, build new blocks and discard old ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import ClassVar

from krrood.entity_query_language.factories import ConditionType, and_, inference
from krrood.planning.action_schema import ActionSchema
from krrood.planning.effect import (
    Assignment,
    Effect,
    ElementInsertion,
    ObjectCreation,
    ObjectRemoval,
)
from krrood.symbol_graph.symbol_graph import Symbol

# %% surfaces


@dataclass(kw_only=True, eq=False)
class Surface(Symbol):
    """
    Something a block can rest on.

    Compared by identity, as a planning domain's objects are.
    """

    name: str
    """
    The name of the surface.
    """

    clear: bool = True
    """
    Whether nothing rests on the surface.
    """


@dataclass(kw_only=True, eq=False)
class StackingTable(Surface):
    """
    The table every tower stands on; it stays clear however many blocks it carries.
    """


@dataclass(kw_only=True, eq=False)
class StackingBlock(Surface):
    """
    A block resting on another block or on the table.
    """

    support: Surface
    """
    What the block rests on.
    """


@dataclass(kw_only=True, eq=False)
class PlannedBlock(StackingBlock):
    """
    A block that exists only in the states that build it.
    """

    _cache_instances_: ClassVar[bool] = False
    """
    Constructing one leaves the live symbol graph unchanged.
    """


@dataclass(kw_only=True, eq=False)
class Tray(Symbol):
    """
    A tray that collects blocks.
    """

    name: str
    """
    The name of the tray.
    """

    blocks: list[StackingBlock] = field(default_factory=list)
    """
    The blocks on the tray, in the order they were put on it.
    """


# %% worlds


@dataclass(kw_only=True)
class StackingWorld:
    """
    Block ``c`` on block ``a``, blocks ``a`` and ``b`` on the table, and an empty tray.
    """

    table: StackingTable
    """
    The table both towers stand on.
    """

    a: StackingBlock
    """
    The bottom block of the first tower.
    """

    b: StackingBlock
    """
    The only block of the second tower.
    """

    c: StackingBlock
    """
    The top block of the first tower.
    """

    tray: Tray
    """
    The empty tray.
    """

    @classmethod
    def build(cls) -> StackingWorld:
        """
        :return: A new world, whose objects the live symbol graph holds.
        """
        table = StackingTable(name="table")
        a = StackingBlock(name="a", support=table, clear=False)
        return cls(
            table=table,
            a=a,
            b=StackingBlock(name="b", support=table),
            c=StackingBlock(name="c", support=a),
            tray=Tray(name="tray"),
        )


# %% action schemas


@dataclass(kw_only=True)
class Stack(ActionSchema):
    """
    Moves a clear block onto another clear block.
    """

    block: StackingBlock
    """
    The block that moves.
    """

    target: StackingBlock
    """
    The block it is put on.
    """

    def precondition(self) -> ConditionType:
        return and_(
            self.block.clear == True,
            self.target.clear == True,
            self.block != self.target,
        )

    def effects(self) -> list[Effect]:
        return [
            Assignment(fact=self.block.support.clear == True),
            Assignment(fact=self.block.support == self.target),
            Assignment(fact=self.target.clear == False),
        ]


@dataclass(kw_only=True)
class MoveToTable(ActionSchema):
    """
    Moves a clear block from another block onto the table.
    """

    block: StackingBlock
    """
    The block that moves.
    """

    table: StackingTable
    """
    The table it is put on.
    """

    def precondition(self) -> ConditionType:
        return and_(self.block.clear == True, self.block.support != self.table)

    def effects(self) -> list[Effect]:
        return [
            Assignment(fact=self.block.support.clear == True),
            Assignment(fact=self.block.support == self.table),
        ]


@dataclass(kw_only=True)
class PutOnTray(ActionSchema):
    """
    Records a block on a tray.
    """

    block: StackingBlock
    """
    The block put on the tray.
    """

    tray: Tray
    """
    The tray that collects it.
    """

    def precondition(self) -> ConditionType:
        return self.block.clear == True

    def effects(self) -> list[Effect]:
        return [ElementInsertion(collection=self.tray.blocks, element=self.block)]


@dataclass(kw_only=True)
class BuildBlock(ActionSchema):
    """
    Builds a new block, named after the block it is built on, on top of a clear block.
    """

    base: StackingBlock
    """
    The block the new block is built on.
    """

    def precondition(self) -> ConditionType:
        return self.base.clear == True

    def effects(self) -> list[Effect]:
        return [
            ObjectCreation(
                objects=inference(PlannedBlock)(name=self.base.name, support=self.base)
            ),
            Assignment(fact=self.base.clear == False),
        ]


@dataclass(kw_only=True)
class Discard(ActionSchema):
    """
    Discards a clear block, which clears what it rested on.
    """

    block: StackingBlock
    """
    The block discarded.
    """

    def precondition(self) -> ConditionType:
        return self.block.clear == True

    def effects(self) -> list[Effect]:
        return [
            ObjectRemoval(objects=self.block),
            Assignment(fact=self.block.support.clear == True),
        ]
