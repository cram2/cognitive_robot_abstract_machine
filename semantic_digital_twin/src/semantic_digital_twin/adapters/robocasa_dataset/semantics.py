from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import ClassVar, Dict, Optional, Type

from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Banana,
    Bottle,
    Bowl,
    Bread,
    Cabinet,
    Carrot,
    CoffeeMachine,
    Cooktop,
    CounterTop,
    Cup,
    Dishwasher,
    Drawer,
    Fridge,
    Hood,
    Kettle,
    Lettuce,
    Microwave,
    Mug,
    Orange,
    Oven,
    Pan,
    Plate,
    Pot,
    Potato,
    Sink,
    Toaster,
    Tomato,
)
from semantic_digital_twin.utils import camel_case_split


class RoboCasaKitchenApplianceCategory(StrEnum):
    """
    RoboCasa fixture categories that this adapter knows how to map to a
    SemanticAnnotation subclass.
    """

    CABINET = "cabinet"
    DRAWER = "drawer"
    FRIDGE = "fridge"
    MICROWAVE = "microwave"
    OVEN = "oven"
    DISHWASHER = "dishwasher"
    HOOD = "hood"
    COUNTER = "counter"
    SINK = "sink"
    STOVE = "stove"
    TOASTER = "toaster"
    COFFEE_MACHINE = "coffee_machine"


class RoboCasaObjectCategory(StrEnum):
    """
    RoboCasa object categories that this adapter knows how to map to a
    SemanticAnnotation subclass.
    """

    APPLE = "apple"
    BANANA = "banana"
    ORANGE = "orange"
    TOMATO = "tomato"
    LETTUCE = "lettuce"
    CARROT = "carrot"
    POTATO = "potato"
    BOTTLE = "bottle"
    CUP = "cup"
    MUG = "mug"
    BOWL = "bowl"
    PLATE = "plate"
    PAN = "pan"
    POT = "pot"
    KETTLE = "kettle"
    BREAD = "bread"


def _tokenize_category(category: str) -> set[str]:
    """
    Split a RoboCasa category into lower-case word tokens, handling snake_case object
    categories (for example ``"coffee_machine"``), upper camel case fixture class names
    (for example ``"HingeCabinet"``), and plain single-case words (for example
    ``"MICROWAVE"`` or ``"sink"``).

    :param category: The category string or class name to tokenize.
    :return: The set of lower-case word tokens.
    """
    tokens: set[str] = set()
    for part in category.split("_"):
        if not part:
            continue
        # camel_case_split only makes sense for genuinely mixed-case words; an already all-upper or
        # all-lower part (for example "MICROWAVE" or "sink") is a single token as-is.
        if part.isupper() or part.islower():
            tokens.add(part.lower())
        else:
            tokens.update(token.lower() for token in camel_case_split(part) if token)
    return tokens


@dataclass
class RoboCasaCategoryResolver:
    """
    Maps a RoboCasa category (a fixture Python class name or an object category string) to the
    matching SemanticAnnotation subclass already defined in the digital twin, by token-subset
    matching: a registered category matches if all of its word tokens appear in the given
    category's tokens. This lets a compound RoboCasa fixture class name such as ``"HingeCabinet"``
    still match a shorter registered category such as ``"cabinet"``. When more than one registered
    category matches, the one with the most tokens (the most specific match) wins.

    Categories with no matching registered category resolve to ``None``; callers should fall back to
    a generic annotation (for example :class:`~semantic_digital_twin.semantic_annotations.natural_language.NaturalLanguageWithTypeDescription`)
    for those.
    """

    category_to_annotation_class: ClassVar[Dict[str, Type[HasRootBody]]]
    """
    Lookup table from registered RoboCasa category to the matching SemanticAnnotation
    subclass.
    """

    def resolve(self, category: str) -> Optional[Type[HasRootBody]]:
        """
        Resolve a RoboCasa category to the matching SemanticAnnotation subclass.

        :param category: The RoboCasa category, for example ``"cabinet"``,
            ``"HingeCabinet"``, or ``"coffee_machine"``.
        :return: The most specific matching SemanticAnnotation subclass, or None if no
            registered category matches.
        """
        category_tokens = _tokenize_category(category)

        best_match: Optional[Type[HasRootBody]] = None
        best_match_token_count = 0
        for (
            registered_category,
            annotation_class,
        ) in self.category_to_annotation_class.items():
            registered_tokens = _tokenize_category(registered_category)
            if not registered_tokens <= category_tokens:
                continue
            if len(registered_tokens) > best_match_token_count:
                best_match = annotation_class
                best_match_token_count = len(registered_tokens)
        return best_match


@dataclass
class RoboCasaKitchenApplianceResolver(RoboCasaCategoryResolver):
    """
    Resolves RoboCasa fixture categories (the module names under
    ``robocasa.models.fixtures``, or the fixture's Python class name) to the matching
    SemanticAnnotation subclass.
    """

    category_to_annotation_class: ClassVar[
        Dict[RoboCasaKitchenApplianceCategory, Type[HasRootBody]]
    ] = {
        RoboCasaKitchenApplianceCategory.CABINET: Cabinet,
        RoboCasaKitchenApplianceCategory.DRAWER: Drawer,
        RoboCasaKitchenApplianceCategory.FRIDGE: Fridge,
        RoboCasaKitchenApplianceCategory.MICROWAVE: Microwave,
        RoboCasaKitchenApplianceCategory.OVEN: Oven,
        RoboCasaKitchenApplianceCategory.DISHWASHER: Dishwasher,
        RoboCasaKitchenApplianceCategory.HOOD: Hood,
        RoboCasaKitchenApplianceCategory.COUNTER: CounterTop,
        RoboCasaKitchenApplianceCategory.SINK: Sink,
        RoboCasaKitchenApplianceCategory.STOVE: Cooktop,
        RoboCasaKitchenApplianceCategory.TOASTER: Toaster,
        RoboCasaKitchenApplianceCategory.COFFEE_MACHINE: CoffeeMachine,
    }


@dataclass
class RoboCasaObjectResolver(RoboCasaCategoryResolver):
    """
    Resolves RoboCasa object categories (the keys of
    ``robocasa.models.objects.kitchen_objects.OBJ_CATEGORIES``) to the matching
    SemanticAnnotation subclass.
    """

    category_to_annotation_class: ClassVar[
        Dict[RoboCasaObjectCategory, Type[HasRootBody]]
    ] = {
        RoboCasaObjectCategory.APPLE: Apple,
        RoboCasaObjectCategory.BANANA: Banana,
        RoboCasaObjectCategory.ORANGE: Orange,
        RoboCasaObjectCategory.TOMATO: Tomato,
        RoboCasaObjectCategory.LETTUCE: Lettuce,
        RoboCasaObjectCategory.CARROT: Carrot,
        RoboCasaObjectCategory.POTATO: Potato,
        RoboCasaObjectCategory.BOTTLE: Bottle,
        RoboCasaObjectCategory.CUP: Cup,
        RoboCasaObjectCategory.MUG: Mug,
        RoboCasaObjectCategory.BOWL: Bowl,
        RoboCasaObjectCategory.PLATE: Plate,
        RoboCasaObjectCategory.PAN: Pan,
        RoboCasaObjectCategory.POT: Pot,
        RoboCasaObjectCategory.KETTLE: Kettle,
        RoboCasaObjectCategory.BREAD: Bread,
    }
