from krrood.utils import inheritance_path_length
from ..dataset.example_classes import *


class A: ...


class B(A): ...


class C(A): ...


class D(B, C): ...


class E(D, B): ...


class WackyEnum(E, Enum): ...


def test_distance_between_classes():
    assert inheritance_path_length(Position5D, Position) == 2
    assert inheritance_path_length(Position, Position5D) is None
    assert inheritance_path_length(Atom, Symbol) == 1
    assert inheritance_path_length(MultipleInheritance, PrimaryBase) == 1
    assert inheritance_path_length(D, A) == 2
    assert inheritance_path_length(E, A) == 2
    assert inheritance_path_length(WackyEnum, Enum) == 1
