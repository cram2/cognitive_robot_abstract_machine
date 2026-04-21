"""Gateway for KRROOD Match expressions — snapshot into plain MatchData / MatchSlot.

Downstream llmr modules work with :class:`MatchData` and :class:`MatchSlot` and never
touch the underlying Match expression directly.

  read_match               — snapshot a Match expression; pre-classifies every slot.
  write_slot_value         — write a resolved value back to a slot's variable.
  finalize_match           — propagate literal values and construct the action instance.
  required_match           — build a Match with required public fields left free.
  unresolved_required_fields — list required fields still unset after resolution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing_extensions import Any, Dict, Iterable, List, Optional

from krrood.entity_query_language.query.match import Match

from llmr.bridge.introspect import FieldKind, PycramIntrospector

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class MatchSlot:
    """One leaf variable in a Match expression, pre-classified for resolution."""

    attribute_name: str
    """Leaf field name, e.g. ``'arm'`` or ``'grasp_type'``."""

    prompt_name: str
    """Root-stripped full path used in prompts, e.g. ``'grasp_description.grasp_type'``."""

    field_type: Any
    """Resolved Python type for the slot."""

    field_kind: FieldKind
    """Pre-computed :class:`FieldKind` — lets consumers skip the introspector."""

    value: Any
    """Current variable value; ``Ellipsis`` when the slot is free."""

    is_free: bool
    """True when ``value`` is ``Ellipsis``."""

    _variable: Any = field(repr=False)
    """Opaque KRROOD variable reference used for write-back; do not touch outside bridge."""


@dataclass
class MatchData:
    """Plain-data snapshot of a KRROOD Match expression."""

    action_type: type
    """The action dataclass being matched (e.g. ``PickUpAction``)."""

    action_name: str
    """Convenience: ``action_type.__name__``."""

    slots: List[MatchSlot]
    """Every leaf variable discovered in the expression."""

    _expression: Match[Any] = field(repr=False)
    """Opaque KRROOD Match reference used for finalisation; do not touch outside bridge."""

    @property
    def free_slots(self) -> List[MatchSlot]:
        """Slots whose value is still ``Ellipsis``."""
        return [slot for slot in self.slots if slot.is_free]

    @property
    def free_slot_names(self) -> List[str]:
        """Prompt-facing names of all free slots."""
        return [slot.prompt_name for slot in self.free_slots]

    @property
    def fixed_bindings(self) -> Dict[str, Any]:
        """Prompt-name → value map for slots that are already bound."""
        return {slot.prompt_name: slot.value for slot in self.slots if not slot.is_free}


# ── Public API ────────────────────────────────────────────────────────────────


def read_match(
    expression: Match[Any],
    introspector: Optional[PycramIntrospector] = None,
    unresolved: Any = ...,
) -> MatchData:
    """Snapshot *expression* into a :class:`MatchData` with every slot pre-classified.

    Reads each leaf variable, evaluates selectable variables once, and stores the
    resolved :class:`FieldKind` on each slot so downstream code never needs the
    introspector again.

    :param expression: KRROOD Match expression to read.
    :param introspector: Used for ``classify_type``; a default instance is created if ``None``.
    :param unresolved: Sentinel returned when a variable has no value; defaults to ``Ellipsis``.
    """
    intro = introspector or PycramIntrospector()
    action_cls = expression.type

    slots: List[MatchSlot] = []
    for attr_match in expression.matches_with_variables:
        variable = attr_match.assigned_variable
        field_type = variable._type_
        value = _read_variable_value(variable, unresolved)
        try:
            field_kind = intro.classify_type(field_type)
        except Exception:
            field_kind = FieldKind.PRIMITIVE

        slots.append(
            MatchSlot(
                attribute_name=attr_match.attribute_name,
                prompt_name=_strip_root_prefix(
                    attr_match.name_from_variable_access_path,
                    action_cls,
                ),
                field_type=field_type,
                field_kind=field_kind,
                value=value,
                is_free=isinstance(value, type(Ellipsis)),
                _variable=variable,
            )
        )

    return MatchData(
        action_type=action_cls,
        action_name=getattr(action_cls, "__name__", str(action_cls)),
        slots=slots,
        _expression=expression,
    )


def write_slot_value(slot: MatchSlot, value: Any) -> bool:
    """Write *value* into the KRROOD variable behind *slot*. Returns ``False`` on failure."""
    try:
        slot._variable._value_ = value
        slot.value = value
        slot.is_free = False
        return True
    except Exception as exc:
        logger.warning("cannot set slot '%s': %s", slot.attribute_name, exc)
        return False


def finalize_match(match_data: MatchData) -> Any:
    """Propagate resolved literal values into Match kwargs and construct the action instance."""
    expression = match_data._expression
    expression._update_kwargs_from_literal_values()
    return expression.construct_instance()


def required_match(
    action_cls: type,
    introspector: Optional[PycramIntrospector] = None,
) -> Match[Any]:
    """Return ``Match(action_cls)`` with every required public field left free (Ellipsis).

    Nested :class:`FieldKind.COMPLEX` fields become nested Match expressions whose
    required sub-fields are themselves free.
    """
    match = Match(action_cls)
    intro = introspector or PycramIntrospector()

    try:
        fields = intro.introspect(action_cls).fields
    except Exception:
        return match

    kwargs = _required_match_kwargs(fields)
    if kwargs:
        match(**kwargs)
    return match


def unresolved_required_fields(
    match_data: MatchData,
    introspector: Optional[PycramIntrospector] = None,
) -> List[str]:
    """Return required action fields still unset after slot filling.

    KRROOD may represent a required top-level field as a nested ``Match`` whose
    unresolved leaves live below it (e.g. ``action.slot.member``); such a slot
    is considered present even without a direct variable bound to the top-level name.
    """
    intro = introspector or PycramIntrospector()

    try:
        required = {
            fld.name
            for fld in intro.introspect(match_data.action_type).fields
            if not fld.is_optional
        }
    except Exception:
        return []

    expression = match_data._expression
    unresolved: List[str] = []
    seen: set[str] = set()

    for attr_match in expression.matches_with_variables:
        field_name = attr_match.attribute_name
        top_level_name = _top_level_field_name(attr_match)
        seen.add(field_name)
        if top_level_name:
            seen.add(top_level_name)
        value = attr_match.assigned_variable._value_
        if field_name in required and isinstance(value, type(Ellipsis)):
            unresolved.append(field_name)

    expression._update_kwargs_from_literal_values()
    for field_name in sorted(required):
        if field_name not in expression.kwargs:
            if field_name not in seen:
                unresolved.append(field_name)
            continue
        if not _match_value_is_resolved(expression.kwargs[field_name]):
            unresolved.append(field_name)

    return list(dict.fromkeys(unresolved))


# ── Internal helpers ──────────────────────────────────────────────────────────


def _strip_root_prefix(name: str, action_cls: type) -> str:
    """Strip the ``<ActionClass>.`` root prefix from a variable access path name."""
    prefix = f"{action_cls.__name__}."
    return name[len(prefix) :] if name.startswith(prefix) else name


def _read_variable_value(variable: Any, unresolved: Any) -> Any:
    """Return a variable's concrete value, evaluating it once if it's selectable."""
    try:
        value = vars(variable).get("_value_", unresolved)
    except TypeError:
        value = getattr(variable, "_value_", unresolved)
    if value is not unresolved:
        return value

    evaluate = getattr(variable, "evaluate", None)
    if not callable(evaluate):
        return unresolved
    try:
        value = next(iter(evaluate()))
    except Exception:
        return unresolved
    variable._value_ = value
    return value


def _top_level_field_name(attr_match: Any) -> Optional[str]:
    """Return the first field name in a KRROOD variable access path, if any."""
    try:
        access_path = attr_match.assigned_variable._access_path_
    except AttributeError:
        return None

    try:
        steps = iter(access_path)
    except TypeError:
        steps = iter((access_path,))

    for step in steps:
        attribute_name = getattr(step, "_attribute_name_", None)
        if attribute_name:
            return attribute_name
    return None


def _match_value_is_resolved(value: Any) -> bool:
    """Return whether a Match kwarg value can be constructed without ellipses."""
    if isinstance(value, type(Ellipsis)):
        return False
    if isinstance(value, Match):
        return all(_match_value_is_resolved(item) for item in value.kwargs.values())
    if isinstance(value, dict):
        return all(_match_value_is_resolved(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_match_value_is_resolved(item) for item in value)
    return True


def _required_match_kwargs(fields: Iterable[Any]) -> Dict[str, Any]:
    """Build Match kwargs for required public fields (skipping optional / underscore-prefixed)."""
    kwargs: Dict[str, Any] = {}
    for fld in fields:
        if fld.is_optional or fld.name.startswith("_"):
            continue
        kwargs[fld.name] = _free_match_value(fld)
    return kwargs


def _free_match_value(fld: Any) -> Any:
    """Return ``...`` for a leaf required field or a nested Match for a COMPLEX field."""
    if fld.kind != FieldKind.COMPLEX:
        return ...
    nested_match = Match(fld.raw_type)
    nested_kwargs = _required_match_kwargs(fld.sub_fields)
    if nested_kwargs:
        nested_match(**nested_kwargs)
    return nested_match
