"""
This file contains classes that behave like an extension to classes in example_classes.py
This file is needed to krrood_test orm interface extension.
"""

from dataclasses import dataclass

from typing_extensions import List

from .example_classes import KRROODPosition, Entity


@dataclass
class CustomPosition(KRROODPosition):
    custom_value: str


@dataclass
class AggregatorOfExternalInstances:
    a: List[KRROODPosition]
    entity: Entity
