"""Test action dataclasses — minimal PyCRAM-free stand-ins for introspection and slot-filling tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing_extensions import Optional

from krrood.symbol_graph.symbol_graph import Symbol


class GraspType(Enum):
    """Grasp style enumeration."""

    FRONT = "FRONT"
    """Front-facing grasp."""

    TOP = "TOP"
    """Top-down grasp."""

    SIDE = "SIDE"
    """Side grasp."""


@dataclass
class MockGraspDescription:
    """Minimal stand-in for PyCRAM GraspDescription."""

    grasp_type: GraspType
    """Which grasp style to use."""

    manipulator: Optional[Symbol] = None
    """The manipulator performing the grasp."""


@dataclass
class MockPickUpAction:
    """Minimal stand-in for PyCRAM PickUpAction."""

    object_designator: Symbol
    """The object to pick up."""

    grasp_description: Optional[MockGraspDescription] = None
    """How to grasp the object."""

    timeout: Optional[float] = None
    """Maximum seconds to attempt the action."""


@dataclass
class MockNavigateAction:
    """Minimal stand-in for PyCRAM NavigateAction."""

    target_location: Symbol
    """Destination location."""
