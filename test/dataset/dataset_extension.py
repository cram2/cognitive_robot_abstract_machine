"""
This file contains classes that behave like an extension to classes in example_classes.py
This file is needed to test orm interface extension.
"""

from dataclasses import dataclass

from typing_extensions import List

from .example_classes import Position, Entity


@dataclass
class CustomPosition(Position):
    custom_value: str


@dataclass
class AggregatorOfExternalInstances:
    a: List[Position]
    entity: Entity
