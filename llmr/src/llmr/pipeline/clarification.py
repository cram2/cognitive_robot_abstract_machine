"""Clarification types: raised when entity grounding finds zero candidates.

When the entity grounder cannot resolve an object or surface name to any
``Body`` in the world, the action handler raises ``ClarificationNeededError``
instead of a generic ``RuntimeError``.  The ``ExecutionLoop`` catches this
specific exception and surfaces it as a structured ``ClarificationRequest``
in the ``ExecutionResult``, allowing callers to ask the user for a corrected
name rather than treating it as a hard execution failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import List, Optional


@dataclass
class ClarificationRequest:
    """Structured description of why clarification is needed.

    :param entity_name: The name the LLM extracted that could not be grounded.
    :param entity_role: Human-readable role of the entity (e.g. ``"object"``
        or ``"target surface"``).
    :param available_names: Names of all bodies currently visible in the world
        so the caller can present valid alternatives to the user.
    :param message: Human-readable explanation of the failure.
    """

    entity_name: str
    entity_role: str
    available_names: List[str] = field(default_factory=list)
    message: str = ""


class ClarificationNeededError(Exception):
    """Raised by an ActionHandler when zero bodies are found for an entity.

    Carry the :class:`ClarificationRequest` as ``self.request`` so the
    ``ExecutionLoop`` can forward it to the caller without re-parsing the
    exception message.
    """

    def __init__(self, request: ClarificationRequest) -> None:
        self.request: ClarificationRequest = request
        super().__init__(request.message or f"Cannot ground entity '{request.entity_name}'")


@dataclass
class ArmCapacityRequest:
    """Structured description of an arm capacity constraint violation.

    :param occupied_arms: Names of the arms that are currently occupied.
    :param held_object_names: Names of the objects currently held.
    :param message: Human-readable explanation.
    """

    occupied_arms: List[str] = field(default_factory=list)
    held_object_names: List[str] = field(default_factory=list)
    message: str = ""


class ArmCapacityError(Exception):
    """Raised during planning when all robot arms are already occupied.

    Signals that a PickUpAction cannot be planned because the robot has no
    free arm.  The caller should place one of the held objects before
    attempting another pickup.
    """

    def __init__(self, request: ArmCapacityRequest) -> None:
        self.request: ArmCapacityRequest = request
        super().__init__(request.message or f"All arms occupied: {request.held_object_names}")
