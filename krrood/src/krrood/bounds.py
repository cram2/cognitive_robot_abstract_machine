from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Generic, TypeVar

from krrood.patterns.interface import Interface
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

# %% the corners of an axis-aligned region

T = TypeVar("T")


class BoundsInterface(Generic[T], SubClassSafeGeneric, Interface):
    """
    The lower and upper corner of an axis-aligned region.
    """

    lower: T
    """
    The corner with the smallest coordinate on every axis.
    """

    upper: T
    """
    The corner with the largest coordinate on every axis.
    """


@dataclass
class Bounds(BoundsInterface[T]):
    """
    The lower and upper corner of an axis-aligned region, stored as given.
    """

    lower: T
    """
    The corner with the smallest coordinate on every axis.
    """

    upper: T
    """
    The corner with the largest coordinate on every axis.
    """
