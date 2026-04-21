"""Action dataclass stand-ins — PyCRAM-free shapes for introspection, match, and resolution tests.

Every action here is an ordinary dataclass so the introspector and KRROOD
``Match`` machinery can operate on it exactly as they would on a real
``pycram.robot_plans.actions.*`` class, without pulling the simulator in.

Covered FieldKind paths:
  ENTITY    ``Symbol`` / ``Symbol`` subclass — :class:`MockPickUpAction.object_designator`, :class:`MockRequiredManipulatorAction.manipulator`.
  POSE      resolved via ``body_xyz`` — :class:`MockPoseAction.target_pose` (name-matched by MRO).
  ENUM      :class:`GraspType` — :class:`MockGraspDescription.grasp_type`.
  COMPLEX   nested dataclass — :class:`MockPickUpAction.grasp_description`, :class:`MockRequiredNestedAction.grasp`.
  PRIMITIVE scalar — :class:`MockPickUpAction.timeout`, :class:`MockNestedWithTimeoutAction.priority`.
  TYPE_REF  ``Type[Symbol]`` — :class:`MockTypeRefAction.annotation_type`.
  Container ``List[Symbol]`` — :class:`MockContainerAction.object_designators`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing_extensions import List, Optional, Type

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


@dataclass
class MockTypeRefAction:
    """Action whose only slot is a ``Type[Symbol]`` reference."""

    annotation_type: Type[Symbol]
    """Semantic annotation class the action should target."""


@dataclass
class MockContainerAction:
    """Action with a list-of-symbol slot — exercises the container fallback."""

    object_designators: List[Symbol] = field(default_factory=list)
    """Objects to act on in bulk."""


@dataclass
class MockRequiredNestedAction:
    """Action with a required nested COMPLEX slot — no optional fields."""

    object_designator: Symbol
    """The object to pick up."""

    grasp: MockGraspDescription
    """Required grasp description (no default)."""


@dataclass
class MockRequiredManipulatorAction:
    """Action with a required ``Symbol`` subclass slot — exercises ENTITY grounding.

    Useful as a target for tests that need an action whose only required slot is
    a domain-specific Symbol subclass rather than the base ``Symbol``.
    """

    manipulator: Symbol
    """Manipulator the action should control."""


@dataclass
class MockNestedWithTimeoutAction:
    """Action combining a required nested slot with a primitive and a primitive-default."""

    grasp: MockGraspDescription
    """Required grasp description."""

    priority: int
    """Required primitive slot."""

    timeout: Optional[float] = 1.5
    """Optional primitive slot with default."""


@dataclass
class Pose:
    """Minimal stand-in for ``Pose``.

    Classification keys off the class name via MRO, so the class itself must be
    named exactly ``Pose`` — see :attr:`PycramIntrospector.POSE_TYPE_NAMES`.
    """


# Backwards-compat alias preserved for older test imports.
MockPose = Pose


@dataclass
class MockPoseAction:
    """Action with a POSE-classified slot."""

    target_pose: Pose
    """Where to move."""
