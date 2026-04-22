"""Slot-value resolution — turns LLM slot outputs into concrete Python values.

Dispatches by :class:`FieldKind` (pre-classified on each :class:`MatchSlot`):
  ENTITY / POSE / TYPE_REF → ground via :class:`EntityGrounder`.
  ENUM                     → coerce string to enum member.
  PRIMITIVE                → cast string to bool / int / float / str.
  COMPLEX                  → not resolved here (nested Match handles it).
"""

from __future__ import annotations

import logging
import typing
from typing import TYPE_CHECKING
from typing_extensions import Any, Dict, Optional

from llmr.bridge.introspect import FieldKind
from llmr.bridge.world_reader import resolve_symbol_class

if TYPE_CHECKING:
    from llmr.bridge.match_reader import MatchSlot
    from llmr.resolution.grounder import EntityGrounder
    from llmr.schemas import SlotValue

logger = logging.getLogger(__name__)


def resolve_binding_value(
    slot: "MatchSlot",
    slot_by_name: Dict[str, "SlotValue"],
    grounder: "EntityGrounder",
    resolved_params: Dict[str, Any],
    unresolved: Any,
) -> Any:
    """Resolve one free :class:`MatchSlot` from LLM output and SymbolGraph grounding."""
    field_type = slot.field_type
    kind = slot.field_kind

    if kind in (FieldKind.ENTITY, FieldKind.POSE, FieldKind.TYPE_REF):
        slot_value = slot_by_name.get(slot.prompt_name)
        if slot_value is None:
            return unresolved
        return resolve_entity_slot(
            slot_value,
            grounder,
            kind,
            slot.prompt_name,
            expected_type=field_type,
            resolved_params=resolved_params,
            unresolved=unresolved,
        )

    if kind == FieldKind.ENUM:
        slot_value = slot_by_name.get(slot.prompt_name)
        if slot_value is not None and slot_value.value:
            return coerce_enum(slot_value.value, field_type)
        return unresolved

    if kind == FieldKind.COMPLEX:
        return unresolved

    slot_value = slot_by_name.get(slot.prompt_name)
    if slot_value is not None and slot_value.value is not None:
        return coerce_primitive(slot_value.value, field_type)

    return unresolved


def resolve_entity_slot(
    sv: "SlotValue",
    grounder: "EntityGrounder",
    kind: FieldKind,
    field_name: str,
    expected_type: Optional[type] = None,
    resolved_params: Optional[Dict[str, Any]] = None,
    unresolved: Any = None,
) -> Any:
    """Ground an ENTITY / POSE / TYPE_REF slot to a Symbol instance via :class:`EntityGrounder`."""
    from llmr.schemas import EntityDescriptionSchema

    ed = sv.entity_description
    if ed is not None:
        grounding_ed = ed
    elif sv.value:
        # LLM used the plain value field instead of entity_description.
        # Recover semantic_type from the known field type so Tier 1 grounding fires.
        inferred_type = (
            expected_type.__name__ if isinstance(expected_type, type) else None
        )
        grounding_ed = EntityDescriptionSchema(
            name=sv.value, semantic_type=inferred_type
        )
    else:
        logger.warning(
            "resolve_entity_slot: field '%s' has neither entity_description nor value.",
            field_name,
        )
        return unresolved

    grounding = grounder.ground(grounding_ed, expected_type=expected_type)
    if grounding.warning:
        logger.warning("Grounding warning for '%s': %s", field_name, grounding.warning)
    if not grounding.bodies:
        logger.warning(
            "resolve_entity_slot: no bodies found for field '%s' (name=%r, type=%r).",
            field_name,
            grounding_ed.name,
            grounding_ed.semantic_type,
        )
        return unresolved

    body = grounding.bodies[0]

    if kind == FieldKind.ENTITY and isinstance(expected_type, type):
        if not isinstance(body, expected_type):
            logger.warning(
                "resolve_entity_slot: grounded value for field '%s' has type %s, expected %s.",
                field_name,
                type(body).__name__,
                expected_type.__name__,
            )
            return unresolved

    if kind == FieldKind.POSE:
        try:
            return body.global_pose
        except AttributeError:
            logger.warning("Grounded body for '%s' has no global_pose.", field_name)
            return unresolved

    if kind == FieldKind.TYPE_REF:
        if ed is not None and ed.semantic_type:
            cls = resolve_symbol_class(
                ed.semantic_type,
                symbol_graph=getattr(grounder, "symbol_graph", None),
            )
            if cls is not None:
                return cls
        return body

    return body


def coerce_enum(value: str, enum_type: type) -> Any:
    """Convert a string to the matching enum member (case-insensitive fallback)."""
    try:
        return enum_type[value]
    except KeyError:
        pass

    value_upper = value.upper()
    for member in enum_type:
        if member.name.upper() == value_upper:
            return member

    first = next(iter(enum_type))
    logger.warning(
        "coerce_enum: '%s' is not a valid member of %s %s; falling back to %s.",
        value,
        enum_type.__name__,
        list(enum_type.__members__),
        first.name,
    )
    return first


def coerce_primitive(value: str, field_type: Any) -> Any:
    """Cast an LLM string output to bool, int, float, or str as required."""
    origin = typing.get_origin(field_type)
    if origin is typing.Union:
        args = [arg for arg in typing.get_args(field_type) if arg is not type(None)]
        unwrapped = args[0] if len(args) == 1 else field_type
    else:
        unwrapped = field_type

    if unwrapped is bool:
        return value.lower() in ("true", "1", "yes")
    if unwrapped is int:
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    if unwrapped is float:
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    return value
