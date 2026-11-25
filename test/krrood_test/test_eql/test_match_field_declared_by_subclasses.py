"""
A pattern written for a class that leaves a collection field to its subclasses, or that
is the value of a field declared with a type variable, still matches the field's
elements one by one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from krrood.entity_query_language.factories import a, an, every_is, none_is
from krrood.symbol_graph.symbol_graph import Symbol


@dataclass(eq=False)
class Finger(Symbol):
    name: str


@dataclass(eq=False)
class LeftFinger(Finger): ...


@dataclass(eq=False)
class RightFinger(Finger): ...


@dataclass(eq=False)
class Hand(Symbol):
    """A hand that leaves its fingers to its subclasses."""

    name: str


@dataclass(eq=False)
class TwoFingerHand(Hand):
    fingers: list[Finger] = field(default_factory=list)


THand = TypeVar("THand")


@dataclass(eq=False)
class Limb(Symbol, Generic[THand]):
    name: str
    hand: THand = None


def build_limbs() -> list[Limb]:
    """
    :return: A limb whose hand has a left and a right finger, and one whose hand has a
        left finger only.
    """
    both = TwoFingerHand(
        name="both", fingers=[LeftFinger(name="left"), RightFinger(name="right")]
    )
    left_only = TwoFingerHand(name="left only", fingers=[LeftFinger(name="left")])
    return [Limb(name="first", hand=both), Limb(name="second", hand=left_only)]


def test_a_pattern_of_the_class_reads_the_field_its_subclass_declares():
    limbs = build_limbs()
    hands = [limb.hand for limb in limbs]
    found = an(Hand)(fingers=a(RightFinger)).from_(domain=hands).tolist()
    assert [hand.name for hand in found] == ["both"]


def test_none_is_over_a_field_its_subclass_declares():
    hands = [limb.hand for limb in build_limbs()]
    found = an(Hand)(fingers=none_is(RightFinger)).from_(domain=hands).tolist()
    assert [hand.name for hand in found] == ["left only"]


def test_a_pattern_nested_in_a_field_declared_with_a_type_variable():
    limbs = build_limbs()
    found = (
        an(Limb)(hand=an(Hand)(fingers=a(RightFinger))).from_(domain=limbs).tolist()
    )
    assert [limb.name for limb in found] == ["first"]


def test_every_is_nested_in_a_field_declared_with_a_type_variable():
    limbs = build_limbs()
    found = (
        an(Limb)(hand=an(Hand)(fingers=every_is(LeftFinger)))
        .from_(domain=limbs)
        .tolist()
    )
    assert [limb.name for limb in found] == ["second"]
