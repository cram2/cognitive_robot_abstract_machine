import json
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Self

import pytest


from krrood.adapters.json_serializer import (
    SubclassJSONSerializer,
    to_json,
    from_json,
)
from krrood.adapters.exceptions import (
    MissingTypeError,
    InvalidTypeFormatError,
    UnknownModuleError,
    ClassNotFoundError,
    JSON_TYPE_NAME,
)
from krrood.utils import get_full_class_name


@dataclass
class Animal(SubclassJSONSerializer):
    """
    Base animal used in tests.
    """

    name: str
    age: int

    def to_json(self):
        data = super().to_json()
        data.update(
            {
                "name": self.name,
                "age": self.age,
            }
        )
        return data

    @classmethod
    def _from_json(cls, data, **kwargs):
        return cls(
            name=(data["name"]),
            age=(data["age"]),
        )


@dataclass
class Dog(Animal):
    """
    Dog subtype for tests.
    """

    breed: str = "mixed"

    def to_json(self):
        data = super().to_json()
        data.update(
            {
                "breed": self.breed,
            }
        )
        return data

    @classmethod
    def _from_json(cls, data, **kwargs):
        return cls(
            name=(data["name"]),
            age=(data["age"]),
            breed=(data["breed"]),
        )


@dataclass
class Bulldog(Dog):
    """
    Deep subtype to ensure deep discovery works.
    """

    stubborn: bool = True

    def to_json(self):
        data = super().to_json()
        data.update(
            {
                "stubborn": (self.stubborn),
            }
        )
        return data

    @classmethod
    def _from_json(cls, data, **kwargs):
        return cls(
            name=(data["name"]),
            age=(data["age"]),
            breed=(data["breed"]),
            stubborn=(data["stubborn"]),
        )


@dataclass
class Cat(Animal):
    """
    Cat subtype for tests.
    """

    lives: int = 9

    def to_json(self):
        data = super().to_json()
        data.update(
            {
                "lives": (self.lives),
            }
        )
        return data

    @classmethod
    def _from_json(cls, data, **kwargs):
        return cls(
            name=(data["name"]),
            age=(data["age"]),
            lives=(data["lives"]),
        )


@dataclass
class ClassThatNeedsKWARGS(SubclassJSONSerializer):
    a: int
    b: float = 0

    def to_json(self) -> Dict[str, Any]:
        return {**super().to_json(), "a": (self.a)}

    @classmethod
    def _from_json(cls, data: Dict[str, Any], **kwargs) -> Self:
        return cls(a=(data["a"]), b=(kwargs["b"]))


class CustomEnum(str, Enum):
    A = "a"
    B = "b"


def test_roundtrip_dog_and_cat():
    dog = Dog(name="Rex", age=5, breed="Shepherd")
    cat = Cat(name="Misty", age=3, lives=7)

    dog_json = dog.to_json()
    cat_json = cat.to_json()

    assert dog_json[JSON_TYPE_NAME] == get_full_class_name(Dog)
    assert cat_json[JSON_TYPE_NAME] == get_full_class_name(Cat)

    dog2 = SubclassJSONSerializer.from_json(dog_json)
    cat2 = SubclassJSONSerializer.from_json(cat_json)

    assert isinstance(dog2, Dog)
    assert isinstance(cat2, Cat)
    assert dog2 == dog
    assert cat2 == cat


def test_deep_subclass_discovery():
    b = Bulldog(name="Butch", age=4, breed="Bulldog", stubborn=True)
    b_json = b.to_json()

    assert b_json[JSON_TYPE_NAME] == get_full_class_name(Bulldog)

    b2 = SubclassJSONSerializer.from_json(b_json)
    assert isinstance(b2, Bulldog)
    assert b2 == b


def test_unknown_module_raises_unknown_module_error():
    with pytest.raises(UnknownModuleError):
        SubclassJSONSerializer.from_json({JSON_TYPE_NAME: "non.existent.Class"})


def test_missing_type_raises_missing_type_error():
    with pytest.raises(MissingTypeError):
        SubclassJSONSerializer.from_json({})


def test_invalid_type_format_raises_invalid_type_format_error():
    with pytest.raises(InvalidTypeFormatError):
        SubclassJSONSerializer.from_json({JSON_TYPE_NAME: "NotAQualifiedName"})


essential_existing_module = "krrood.utils"


def test_class_not_found_raises_class_not_found_error():
    with pytest.raises(ClassNotFoundError):
        SubclassJSONSerializer.from_json(
            {JSON_TYPE_NAME: f"{essential_existing_module}.DoesNotExist"}
        )


def test_uuid_encoding():
    u = uuid.uuid4()
    encoded = to_json(u)
    result = from_json(encoded)
    assert u == result

    us = [uuid.uuid4(), uuid.uuid4()]
    encoded = to_json(us)
    result = from_json(encoded)
    assert us == result


def test_with_kwargs():
    obj = ClassThatNeedsKWARGS(a=1, b=2.0)
    data = obj.to_json()
    result = from_json(data, b=2.0)
    assert obj == result


def test_list_of_enums():
    obj = [CustomEnum.A, CustomEnum.B]
    data = to_json(obj)
    result = from_json(data)
    assert result == obj


def test_exception():
    e = ImportError("test")
    data = to_json(e)
    result = from_json(data)

    assert isinstance(result, ImportError)
    assert result.args == e.args
