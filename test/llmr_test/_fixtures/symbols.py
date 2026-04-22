"""Symbol subclass stand-ins used across world_reader, grounder, and slot_resolution tests.

These are intentionally lightweight — no simulator, no geometry engine — so tests
only rely on the duck-typed accessors in :mod:`llmr.bridge.world_reader`
(``body_display_name``, ``body_xyz``, ``body_bounding_box``) and the
``.bodies`` / ``.parent_connection`` conventions.

Classes:
  :class:`WorldBody`        — a groundable scene object with optional parent.
  :class:`Manipulator`      — a named Symbol whose type-name matches the robot-annotation MRO guard.
  :class:`ParallelGripperLike` — subclass of ``Manipulator`` to cover MRO-based classification.
  :class:`MilkAnnotation`   — semantic annotation with ``.bodies`` and a ``_synonyms`` class var.
  :class:`FakeRobotAnnotation` — imitates a robot annotation (``.root``, ``._robot``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing_extensions import ClassVar, Optional, Set

from krrood.symbol_graph.symbol_graph import Symbol


class WorldBody(Symbol):
    """Generic groundable scene object."""

    def __init__(self, name: str, parent: "Optional[WorldBody]" = None) -> None:
        self.name = name
        self.parent_connection = (
            SimpleNamespace(parent=parent) if parent is not None else None
        )


class Manipulator(Symbol):
    """Symbol subclass whose class name matches the robot-annotation MRO guard."""

    def __init__(self, name: str = "manipulator") -> None:
        self.name = name


class ParallelGripperLike(Manipulator):
    """Robot-component subclass of :class:`Manipulator` — exercises MRO-based detection.

    Concrete gripper types in real PyCRAM do not inherit ``KinematicChain`` and
    lack a ``.bodies`` property; the world reader's MRO check must still pick them
    up under their ``Manipulator`` ancestor.
    """

    def __init__(self, name: str = "parallel_gripper") -> None:
        super().__init__(name=name)


class MilkAnnotation(Symbol):
    """Semantic annotation pointing at one or more bodies; resolvable via synonym."""

    _synonyms: ClassVar[Set[str]] = {"milk"}

    def __init__(self, *bodies: WorldBody) -> None:
        self.bodies = list(bodies)


class FakeRobotAnnotation(Symbol):
    """Robot-shaped annotation — has ``.root`` and ``._robot`` like real annotations."""

    def __init__(self, *bodies: WorldBody) -> None:
        self.bodies = list(bodies)
        self.root = bodies[0] if bodies else None
        self._robot = self
