"""
What an experiment has to say about its domain for the comparison to run on it: which
class is the relational example, which class holds its aggregation counts, what the
outcome the questions ask about is, and the words the report uses for it all.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, fields
from enum import StrEnum

from typing_extensions import Any, Dict, Tuple, Type, get_args, get_type_hints


class ExampleView(StrEnum):
    """
    How much of an example a likelihood is taken over.
    """

    SCALARS = "scalars"
    """
    The example's own scalars.
    """

    SCALARS_AND_COUNTS = "scalars and counts"
    """
    The scalars and the aggregation counts over the parts.
    """

    WHOLE = "whole"
    """
    The scalars, the counts, and every part.
    """


class AbsentPart(StrEnum):
    """
    The symbol an unrolled column holds where an example has no part at that position.
    """

    ABSENT = "absent"


@dataclass(frozen=True)
class PartPadding:
    """
    What an unrolled column holds at a position the example has no part for, per kind of
    attribute; the values must lie outside every real one.
    """

    symbol: AbsentPart = AbsentPart.ABSENT
    """
    For an enum attribute: a symbol of its own next to the enum's members.
    """

    integer: int = -1
    """
    For an integer attribute.
    """

    real: float = -1.0
    """
    For a continuous attribute.
    """

    def value_for(self, attribute_type: Type) -> Any:
        """
        :param attribute_type: The attribute's type.
        :return: The padding value of that kind.
        """
        if issubclass(attribute_type, enum.Enum):
            return self.symbol
        if issubclass(attribute_type, float):
            return self.real
        return self.integer


@dataclass(frozen=True)
class RelationalDomain:
    """
    One experiment's relational example and the words for it.
    """

    example_class: Type
    """
    The dataclass that is the relational example, with its exchangeable parts as list
    fields.
    """

    aggregation_class: Type
    """
    The aggregation statistics over the example's parts; its registry names the
    exchangeable-part fields.
    """

    effect_field: str
    """
    The example's own field the questions ask about, whose rate the report summarises.
    """

    noun: str
    """
    What one example is called, such as ``scene``.
    """

    plural: str
    """
    What several are called, such as ``scenes``.
    """

    effect_phrase: str = "shows the effect"
    """
    What an example that shows the effect does, as a verb phrase after the noun, such as
    ``leaves every object graspable``.
    """

    part_nouns: Dict[str, str] = field(default_factory=dict)
    """
    Per exchangeable-part field, what one part is called, such as ``object``.
    """

    padding: PartPadding = PartPadding()
    """
    What an unrolled column holds at a position without a part.
    """

    @property
    def part_fields(self) -> Tuple[str, ...]:
        """
        The exchangeable-part fields of the example, which are the fields the
        aggregation class aggregates over.
        """
        return tuple(self.aggregation_class.aggregation_registry)

    @property
    def scalar_fields(self) -> Tuple[str, ...]:
        """
        The example's own scalar fields.
        """
        return tuple(
            example_field.name
            for example_field in fields(self.example_class)
            if example_field.name not in self.part_fields
        )

    def part_class(self, part_field: str) -> Type:
        """
        :param part_field: An exchangeable-part field.
        :return: The class of that field's parts.
        """
        [part_class] = get_args(get_type_hints(self.example_class)[part_field])
        return part_class

    def part_attribute_types(self, part_field: str) -> Dict[str, Type]:
        """
        :param part_field: An exchangeable-part field.
        :return: The attributes of that field's parts and their types.
        """
        return get_type_hints(self.part_class(part_field))

    def part_noun(self, part_field: str) -> str:
        """
        :param part_field: An exchangeable-part field.
        :return: What one of its parts is called.
        """
        return self.part_nouns.get(part_field, part_field.rstrip("s"))

    def view_label(self, view: ExampleView) -> str:
        """
        :param view: How much of an example a likelihood is taken over.
        :return: The view in words, such as ``whole scene``.
        """
        if view is ExampleView.WHOLE:
            return f"whole {self.noun}"
        return view.value

    def parts_of(self, example: Any, part_field: str) -> list:
        """
        :param example: An example.
        :param part_field: An exchangeable-part field.
        :return: The example's parts of that kind.
        """
        return vars(example)[part_field]
