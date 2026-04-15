"""LLMBackend — GenerativeBackend implementation that uses an LLM to fill underspecified Match slots.

World context is derived from SymbolGraph.
"""
from __future__ import annotations

import logging
import typing
from dataclasses import dataclass, field
from typing_extensions import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match
from krrood.symbol_graph.symbol_graph import Symbol

from krrood.entity_query_language.utils import T

from llmr.exceptions import LLMSlotFillingFailed, LLMUnresolvedRequiredFields
from llmr._utils import field_short_name as _leaf_field_name, slot_prompt_name as _slot_prompt_name

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr.pycram_bridge.introspector import FieldKind, FieldSpec, PycramIntrospector
    from llmr.schemas.slots import SlotValue
    from llmr.world.grounder import EntityGrounder


# ── Typed sentinel ─────────────────────────────────────────────────────────────

class _Unresolved:
    """Singleton sentinel returned when a slot cannot be resolved.

    Using a dedicated class (rather than ``object()``) lets type checkers
    distinguish unresolved returns from legitimate ``Any`` values, and gives
    a descriptive ``repr`` in log output.
    """

    _instance: "_Unresolved | None" = None

    def __new__(cls) -> "_Unresolved":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<UNRESOLVED>"


_UNRESOLVED = _Unresolved()


# ── LLMBackend ─────────────────────────────────────────────────────────────────

@dataclass
class LLMBackend(GenerativeBackend):
    """A GenerativeBackend that uses an LLM to fill underspecified Match slots."""

    llm: "BaseChatModel"
    """LangChain BaseChatModel — the reasoning engine for slot filling and action classification."""

    groundable_type: Type[Symbol] = field(default=Symbol)
    """
    Symbol subclass scoping entity grounding and world serialisation.
    Defaults to ``Symbol`` (all instances); pass ``Body`` for physical-body-only scope.
    """

    instruction: Optional[str] = field(kw_only=True, default=None)
    """
    NL instruction included in the slot-filler prompt for semantic grounding
    (e.g. ``"the milk from the table"``).  Omit when the action type and fixed
    slots already carry the intent.
    """

    world_context_provider: Optional[Callable[[], str]] = field(kw_only=True, default=None)
    """
    Callable returning a world-context string.  Replaces the default SymbolGraph
    serialisation when provided.  Useful for injecting a custom or pre-cached
    world description.
    """

    strict_required: bool = field(kw_only=True, default=False)
    """
    When ``True``, raise :class:`~llmr.exceptions.LLMUnresolvedRequiredFields`
    if required action fields remain unresolved instead of constructing a partially
    resolved action.
    """

    # ── Core interface ─────────────────────────────────────────────────────────

    def _evaluate(self, expression: Match[T]) -> Iterable[T]:
        """Resolve all free slots in *expression* and yield a fully-constructed action instance."""

        # ── 1. Parse free / fixed slots from the Match variable graph ──────────
        free_slots: List[Tuple[str, Any]] = []
        fixed_slots: Dict[str, Any] = {}
        field_types: Dict[str, Any] = {}
        # name_from_variable_access_path may return 'ClassName.field_name'.
        # We normalise to bare field names for all lookups but keep the full
        # path so we can call _get_mapped_variable_by_name() with it later.
        _full_name_map: Dict[str, str] = {}   # short_name → full access path

        for attr_match in expression.matches_with_variables:
            fname_raw = attr_match.name_from_variable_access_path
            fname = attr_match.attribute_name
            value = _assigned_variable_value(attr_match.assigned_variable)
            ftype = attr_match.assigned_variable._type_
            field_types[fname] = ftype
            _full_name_map[fname] = fname_raw

            if isinstance(value, type(Ellipsis)):
                free_slots.append((fname, ftype))
            else:
                fixed_slots[fname] = value

        if not free_slots:
            expression._update_kwargs_from_literal_values()
            yield expression.construct_instance()
            return

        # ── 2. World context ───────────────────────────────────────────────────
        world_context = self._get_world_context()

        # ── 3. Run the slot filler (LLM call with dynamic prompt) ─────────────
        # krrood already resolved each field's type via get_field_type_endpoint()
        # and stored it in attr_match.assigned_variable._type_ — we use those
        # types directly below instead of re-running full action-class introspection.
        llm_free_slot_names = [
            _slot_prompt_name(_full_name_map.get(name, name), expression.type)
            for name, _ in free_slots
        ]
        output = None
        if llm_free_slot_names:
            from llmr.reasoning.slot_filler import run_slot_filler
            output = run_slot_filler(
                instruction=self.instruction,
                action_cls=expression.type,
                free_slot_names=llm_free_slot_names,
                fixed_slots=fixed_slots,
                world_context=world_context,
                llm=self.llm,
            )
            if output is None:
                raise LLMSlotFillingFailed(action_name=expression.type.__name__)

        # ── 4. Resolve each free slot ──────────────────────────────────────────
        from llmr.pycram_bridge.introspector import FieldKind, PycramIntrospector
        from llmr.world.grounder import EntityGrounder

        _intro = PycramIntrospector()
        grounder = EntityGrounder(self.groundable_type)

        slot_by_name: Dict[str, "SlotValue"] = {}
        if output:
            for sv in output.slots:
                slot_by_name[sv.field_name] = sv
                slot_by_name.setdefault(_leaf_field_name(sv.field_name), sv)  # _leaf_field_name from _utils
        # Tracks successfully resolved top-level values so COMPLEX reconstruction
        # can use them (e.g. arm → pick matching Manipulator from SymbolGraph).
        resolved_params: Dict[str, Any] = {}

        for field_name, field_type in free_slots:
            # Classify using krrood's already-resolved type — no re-introspection needed.
            kind = _intro.classify_type(field_type)

            resolved = _UNRESOLVED

            if kind in (FieldKind.ENTITY, FieldKind.POSE, FieldKind.TYPE_REF):
                sv = slot_by_name.get(field_name)
                if sv is not None:
                    resolved = _resolve_entity_slot(
                        sv, grounder, kind, field_name, expected_type=field_type
                    )

            elif kind == FieldKind.ENUM:
                # field_type IS the enum class — krrood resolved it already.
                sv = slot_by_name.get(field_name)
                if sv is not None and sv.value:
                    resolved = _coerce_enum(sv.value, field_type)

            elif kind == FieldKind.COMPLEX:
                # Lazily introspect sub-fields only for complex types that need
                # recursive construction — avoids upfront full action introspection.
                try:
                    sub_schema = _intro.introspect(field_type)
                    sub_fields = sub_schema.fields
                except Exception:
                    sub_fields = []
                if sub_fields:
                    from llmr.pycram_bridge.introspector import FieldSpec
                    fspec = FieldSpec(
                        name=field_name,
                        raw_type=field_type,
                        kind=kind,
                        sub_fields=sub_fields,
                    )
                    resolved = _reconstruct_complex(
                        field_name=field_name,
                        fspec=fspec,
                        slot_by_name=slot_by_name,
                        grounder=grounder,
                        resolved_params=resolved_params,
                    )

            elif kind == FieldKind.PRIMITIVE or kind is None:
                sv = slot_by_name.get(field_name)
                if sv is not None and sv.value is not None:
                    resolved = coerce_primitive(sv.value, field_type)

            if resolved is _UNRESOLVED:
                logger.debug(
                    "LLMBackend: field '%s' unresolved — leaving as default.",
                    field_name,
                )
                continue

            resolved_params[field_name] = resolved
            try:
                # Use the original full access path (e.g. 'PickUpAction.arm') since
                # _get_mapped_variable_by_name matches on name_from_variable_access_path.
                mapped_var = expression._get_mapped_variable_by_name(
                    _full_name_map.get(field_name, field_name)
                )
                mapped_var._value_ = resolved
            except Exception as exc:
                logger.warning(
                    "LLMBackend: cannot set field '%s': %s", field_name, exc
                )

        if self.strict_required:
            unresolved_required = _unresolved_required_fields(expression, _intro)
            if unresolved_required:
                raise LLMUnresolvedRequiredFields(
                    action_name=getattr(expression.type, "__name__", str(expression.type)),
                    unresolved_fields=unresolved_required,
                )

        expression._update_kwargs_from_literal_values()
        yield expression.construct_instance()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _get_world_context(self) -> str:
        if self.world_context_provider is not None:
            try:
                return self.world_context_provider()
            except Exception as exc:
                logger.warning(
                    "LLMBackend: world_context_provider raised %s — falling back to SymbolGraph.",
                    exc,
                )
        from llmr.world.serializer import serialize_world_from_symbol_graph
        return serialize_world_from_symbol_graph(self.groundable_type)


# ── Per-slot resolvers (module-level, reusable) ────────────────────────────────

def _resolve_entity_slot(
    sv: "SlotValue",
    grounder: "EntityGrounder",
    kind: "FieldKind",
    field_name: str,
    expected_type: Optional[type] = None,
) -> Any:
    """Ground an ENTITY/POSE/TYPE_REF slot to a Symbol instance via EntityGrounder."""
    from llmr.pycram_bridge.introspector import FieldKind
    from llmr.world.grounder import resolve_symbol_class
    from llmr.schemas.entities import EntityDescriptionSchema

    # Build EntityDescriptionSchema for the grounder — use the LLM's entity
    # description if available, otherwise fall back to the value string.
    ed = sv.entity_description
    if ed is not None:
        grounding_ed = ed  # already an EntityDescriptionSchema from the LLM output
    elif sv.value:
        grounding_ed = EntityDescriptionSchema(name=sv.value)
    else:
        logger.warning(
            "_resolve_entity_slot: field '%s' has neither entity_description nor value.",
            field_name,
        )
        return _UNRESOLVED

    grounding = grounder.ground(grounding_ed)
    if grounding.warning:
        logger.warning("Grounding warning for '%s': %s", field_name, grounding.warning)
    if not grounding.bodies:
        logger.warning(
            "_resolve_entity_slot: no bodies found for field '%s' (name=%r, type=%r).",
            field_name, grounding_ed.name, grounding_ed.semantic_type,
        )
        return _UNRESOLVED

    body = grounding.bodies[0]
    if kind == FieldKind.ENTITY and isinstance(expected_type, type):
        if not isinstance(body, expected_type):
            logger.warning(
                "_resolve_entity_slot: grounded value for field '%s' has type %s, expected %s.",
                field_name, type(body).__name__, expected_type.__name__,
            )
            return _UNRESOLVED

    if kind == FieldKind.POSE:
        try:
            return body.global_pose
        except AttributeError:
            logger.warning("Grounded body for '%s' has no global_pose.", field_name)
            return _UNRESOLVED

    if kind == FieldKind.TYPE_REF:
        # TYPE_REF fields expect the *class* (e.g. Type[SemanticAnnotation]),
        # resolved from SymbolGraph class diagram.
        if ed is not None and ed.semantic_type:
            cls = resolve_symbol_class(
                ed.semantic_type,
                symbol_graph=getattr(grounder, "symbol_graph", None),
            )
            if cls is not None:
                return cls
        return body  # fallback: return the instance

    return body


def _reconstruct_complex(
    field_name: str,
    fspec: "FieldSpec",
    slot_by_name: Dict[str, Any],
    grounder: "EntityGrounder",
    resolved_params: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build a complex dataclass (e.g. GraspDescription) from dotted SlotValue entries.

    For each sub-field:
      - ENTITY/POSE kind → ground via EntityGrounder (Manipulator, Camera, Body, etc.)
      - ENUM kind        → coerce string from the dotted SlotValue
      - PRIMITIVE kind   → use string value from the dotted SlotValue
      - Missing ENTITY sub-fields that are required → auto-ground from SymbolGraph,
        using the resolved arm value (from resolved_params) to pick the right one
        when multiple instances exist (e.g. left vs right Manipulator).
      - Missing optional sub-fields → let the dataclass default handle them
    """
    from llmr.pycram_bridge.introspector import FieldKind

    kwargs: Dict[str, Any] = {}
    for sub in fspec.sub_fields:
        sub_key = f"{field_name}.{sub.name}"

        if sub_key not in slot_by_name:
            # For required ENTITY sub-fields omitted by the LLM, auto-ground
            # from SymbolGraph so the dataclass constructor doesn't fail.
            if sub.kind in (FieldKind.ENTITY, FieldKind.POSE) and not sub.is_optional:
                val = _auto_ground_sub_entity(sub.raw_type, resolved_params or {})
                if val is not _UNRESOLVED:
                    kwargs[sub.name] = val
            continue

        sv = slot_by_name[sub_key]
        if sub.kind in (FieldKind.ENTITY, FieldKind.POSE, FieldKind.TYPE_REF):
            val = _UNRESOLVED
            if (
                sub.kind == FieldKind.ENTITY
                and not sub.is_optional
                and any(
                    cls.__name__ == "Manipulator"
                    for cls in getattr(sub.raw_type, "__mro__", ())
                )
            ):
                val = _auto_ground_sub_entity(sub.raw_type, resolved_params or {})
            if val is _UNRESOLVED:
                val = _resolve_entity_slot(
                    sv, grounder, sub.kind, sub_key, expected_type=sub.raw_type
                )
            if (
                val is _UNRESOLVED
                and sub.kind in (FieldKind.ENTITY, FieldKind.POSE)
                and not sub.is_optional
            ):
                val = _auto_ground_sub_entity(sub.raw_type, resolved_params or {})
            if val is not _UNRESOLVED:
                kwargs[sub.name] = val
        elif sub.kind == FieldKind.ENUM:
            if sv.value is not None:
                kwargs[sub.name] = _coerce_enum(sv.value, sub.raw_type)
        elif sub.kind == FieldKind.PRIMITIVE:
            if sv.value is not None:
                kwargs[sub.name] = coerce_primitive(sv.value, sub.raw_type)
        elif sv.value is not None:
            kwargs[sub.name] = sv.value

    try:
        return fspec.raw_type(**kwargs)
    except Exception as exc:
        logger.warning(
            "_reconstruct_complex: cannot build %s for '%s': %s",
            fspec.raw_type.__name__, field_name, exc,
        )
        return _UNRESOLVED


def _auto_ground_sub_entity(raw_type: Any, resolved_params: Dict[str, Any]) -> Any:
    """Auto-ground a required ENTITY sub-field the LLM omitted.

    Queries SymbolGraph for all instances of *raw_type*.  When multiple exist
    (e.g. left and right Manipulator), uses the already-resolved arm value from
    *resolved_params* to pick the matching one.  Falls back to the first instance.
    """
    try:
        from krrood.symbol_graph.symbol_graph import SymbolGraph
        instances = list(SymbolGraph().get_instances_of_type(raw_type))
    except Exception:
        return _UNRESOLVED

    if not instances:
        return _UNRESOLVED
    if len(instances) == 1:
        return instances[0]

    # Try to narrow by arm selection already resolved at the top level.
    for val in resolved_params.values():
        arm_name = _name_to_string(getattr(val, "name", None))
        if arm_name is None:
            continue
        arm_upper = arm_name.upper()
        for inst in instances:
            inst_name = _name_to_string(getattr(inst, "name", "")) or ""
            inst_name = inst_name.upper()
            if arm_upper in inst_name or inst_name in arm_upper:
                return inst

    return instances[0]


def _name_to_string(name: Any) -> Optional[str]:
    """Normalize PrefixedName-like values and plain strings to a comparable string."""
    if name is None:
        return None
    if hasattr(name, "name"):
        name = name.name
    return str(name)


def _assigned_variable_value(assigned_variable: Any) -> Any:
    """Return the concrete value of a KRROOD variable, evaluating it when needed."""
    try:
        value = vars(assigned_variable).get("_value_", _UNRESOLVED)
    except TypeError:
        value = getattr(assigned_variable, "_value_", _UNRESOLVED)
    if value is not _UNRESOLVED:
        return value

    # Try evaluating as a KRROOD selectable via the public .evaluate() API.
    evaluate = getattr(assigned_variable, "evaluate", None)
    if not callable(evaluate):
        return _UNRESOLVED
    try:
        value = next(iter(evaluate()))
    except Exception:
        return _UNRESOLVED
    assigned_variable._value_ = value
    return value


def _top_level_field_name(attr_match: Any) -> Optional[str]:
    """Return the first field name in a KRROOD variable access path, if present."""
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


def _match_kwarg_is_resolved(value: Any) -> bool:
    """Return whether a Match kwarg value can be constructed without ellipses."""
    if isinstance(value, type(Ellipsis)):
        return False
    if isinstance(value, Match):
        return all(_match_kwarg_is_resolved(item) for item in value.kwargs.values())
    if isinstance(value, dict):
        return all(_match_kwarg_is_resolved(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_match_kwarg_is_resolved(item) for item in value)
    return True


def _coerce_enum(value: str, enum_type: type) -> Any:
    """Convert a string to the matching enum member (exact, then case-insensitive)."""
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
        "_coerce_enum: '%s' is not a valid member of %s %s — falling back to %s.",
        value, enum_type.__name__, list(enum_type.__members__), first.name,
    )
    return first


def coerce_primitive(value: str, field_type: Any) -> Any:
    """Cast LLM string output to bool / int / float as required by *field_type*; str passthrough."""
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
    return value  # str or unknown → return as-is


def _unresolved_required_fields(expression: Match[Any], introspector: "PycramIntrospector") -> List[str]:
    """Return required action fields that are still unset in a Match expression.

    KRROOD may represent a required top-level field as a nested ``Match`` whose
    unresolved leaves live below it, for example ``action.slot.member``.  In
    that case the top-level slot is present even though no direct variable named
    ``slot`` appears in ``matches_with_variables``.
    """
    try:
        required = {
            field.name
            for field in introspector.introspect(expression.type).fields
            if not field.is_optional
        }
    except Exception:
        return []

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
    for field_name in sorted(required - seen):
        if _match_kwarg_is_resolved(expression.kwargs.get(field_name)):
            continue
        unresolved.append(field_name)

    return unresolved
