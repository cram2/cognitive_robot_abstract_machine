from typing import List, get_origin
from dataclasses import dataclass
from typing_extensions import TypeVar
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.factories import variable_from
from krrood.patterns.role import Role
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from typing_extensions import Generic


@dataclass
class TestHasRootBody:
    name: str


TTestHasRootBody = TypeVar("TTestHasRootBody", bound=TestHasRootBody)


@dataclass
class TestHasStorageSpace(Generic[TTestHasRootBody], SubClassSafeGeneric):
    objects: List[TTestHasRootBody]


TTestLiquid = TypeVar("TTestLiquid", bound=TestHasRootBody)


@dataclass
class TestBottle(TestHasStorageSpace[TTestLiquid]):
    pass


@dataclass
class TestWine(TestHasRootBody):
    pass


@dataclass
class TestWineBottle(Role[TestBottle[TestWine]]):
    bottle: TestBottle[TestWine]

    @classmethod
    def role_taker_attribute(cls) -> Attribute:
        return variable_from(cls).bottle


def test_specialized_generic_resolution():
    # This test reproduces the issue where get_type_hints failed on specialized generics in Python 3.12
    # and also ensures shadowing doesn't happen in the resolution namespace.

    classes = [TestWineBottle, TestBottle, TestWine]
    diagram = ClassDiagram(classes=classes)

    # Verify that we can resolve fields of WineBottle
    wrapped_wine_bottle = diagram.get_wrapped_class(TestWineBottle)
    bottle_field = next(f for f in wrapped_wine_bottle.fields if f.name == "bottle")

    # This should not raise TypeError
    resolved = bottle_field.resolved_type
    assert get_origin(resolved) is TestBottle

    # Now verify that specialized generic node also works
    specialization_type = TestBottle[TestWine]
    wrapped_specialization = diagram.get_wrapped_class(specialization_type)

    # Verify its fields
    for field in wrapped_specialization.fields:
        # This should not raise TypeError
        assert field.resolved_type is not None
