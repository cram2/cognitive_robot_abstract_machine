"""Slot filler — dynamic LLM prompts driven by action introspection.

Two public functions:

  classify_action()  — NL instruction → action class (for nl_plan factory)
  run_slot_filler()  — action class + free slots → ActionReasoningOutput
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import replace
from typing import TYPE_CHECKING
from typing_extensions import Any, Dict, List, Optional, Tuple

from llmr.bridge.introspect import FieldKind, PycramIntrospector
from llmr.exceptions import LLMActionRegistryEmpty
from llmr.schemas import ActionClassification, ActionReasoningOutput

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr.bridge.introspect import ActionSchema, FieldSpec

logger = logging.getLogger(__name__)


# ── System prompts ─────────────────────────────────────────────────────────────

_SLOT_FILLER_SYSTEM = """\
You are a robot action parameter resolver with strong spatial and physical reasoning.

You receive:
  1. A natural-language instruction from a human operator.
  2. The target robot action type, its description, and all free parameter slots.
  3. Already-fixed slot values (do not change these).
  4. The current world state: scene objects, positions, and semantic annotations.

Your task: for every FREE slot, reason carefully and return a SlotValue.

────────────────────────────────────────────────────
ENTITY SLOTS  (objects / surfaces in the world)
────────────────────────────────────────────────────
Return a SlotValue with:
  - field_name  = the role name exactly as listed
  - entity_description populated:
      name           = exact name from Available Semantic Types (for the matching
                       instance) or body_name from Scene Objects
      semantic_type  = EXACT type name from Available Semantic Types that matches
                       the slot's expected type; null only if no match found
      spatial_context = spatial qualifier from instruction ("on the table") or null
      attributes     = discriminating key/value attributes (color, size) or null
  - reasoning   = 1-2 sentences explaining which world object was identified

The entity_description is used for symbolic grounding in SymbolGraph.

────────────────────────────────────────────────────
PARAMETER SLOTS  (enum, primitive, complex sub-fields)
────────────────────────────────────────────────────
Return a SlotValue with:
  - field_name = the parameter name exactly as listed
  - value      = chosen value as a string (exact enum member name or primitive)
  - reasoning  = 1-2 sentences justifying the choice

For ENUM slots use EXACTLY one of the listed allowed values. Never paraphrase,
translate, or describe enum values in natural language.
Complex dataclass fields are resolved through nested KRROOD Match leaves. When
a dotted field is listed (e.g. 'grasp_description.manipulator'), return a
SlotValue using that exact dotted field_name.

Always provide per-slot reasoning. Return structured JSON.
"""

_CLASSIFIER_SYSTEM = """\
You are a robot action classifier.

Given a natural-language instruction, identify which robot action class it
corresponds to from the list of available action classes below.

Available action classes and schema summaries:
{action_classes}

Return the EXACT Python class name (e.g. "PickUpAction" not "pick up action").
Return structured JSON.
"""


# ── Public: action classification ─────────────────────────────────────────────


def classify_action(
    instruction: str,
    llm: "BaseChatModel",
    action_registry: Optional[Dict[str, type]] = None,
) -> Optional[type]:
    """Map an NL instruction to an action class via structured LLM output.

    :param instruction:     The NL instruction.
    :param llm:             LangChain-compatible chat model.
    :param action_registry: Pre-built ``{class_name: class}``; auto-discovered by the bridge if ``None``.
    :returns: The matched action class, or ``None`` if classification fails.
    """
    if action_registry is None:
        from llmr.pycram_bridge import discover_action_classes

        action_registry = discover_action_classes()
    if not action_registry:
        raise LLMActionRegistryEmpty()

    system = _CLASSIFIER_SYSTEM.format(
        action_classes=_build_action_catalog(action_registry)
    )
    structured_llm = llm.with_structured_output(ActionClassification)
    try:
        result: ActionClassification = structured_llm.invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": instruction},
            ]
        )
        return action_registry.get(result.action_type)
    except Exception:
        logger.exception("classify_action: LLM call failed")
        return None


# ── Public: slot filling ───────────────────────────────────────────────────────


def run_slot_filler(
    instruction: Optional[str],
    action_cls: type,
    free_slot_names: List[str],
    fixed_slots: Dict[str, Any],
    world_context: str,
    llm: "BaseChatModel",
) -> Optional[ActionReasoningOutput]:
    """Resolve free Match slots using LLM reasoning driven by action introspection.

    The prompt is built dynamically from the bridge's introspection output:
      - ENTITY/POSE fields get an entity-description section with role, type, and docstring.
      - ENUM fields list all valid member names from the actual Enum class.
      - Nested complex dataclass leaves are handled as dotted field names.

    :param instruction:     Natural-language instruction.
    :param action_cls:      The action dataclass being resolved.
    :param free_slot_names: Field names (from the Match expression) to fill.
    :param fixed_slots:     Already-resolved ``{field_name: value}`` map.
    :param world_context:   Serialised world state string.
    :param llm:             LangChain-compatible chat model.
    :returns: :class:`ActionReasoningOutput` on success; ``None`` on LLM failure.
    """
    slot_names = [_strip_root(n, action_cls) for n in free_slot_names]
    prompt_fixed_slots = {_strip_root(k, action_cls): v for k, v in fixed_slots.items()}

    user_message, expected_slot_names = _build_prompt(
        instruction=instruction,
        action_cls=action_cls,
        free_slot_names=slot_names,
        fixed_slots=prompt_fixed_slots,
        world_context=world_context,
    )

    structured_llm = llm.with_structured_output(ActionReasoningOutput)
    try:
        return _invoke_with_repair(
            structured_llm=structured_llm,
            action_cls=action_cls,
            user_message=user_message,
            expected_slot_names=expected_slot_names,
        )
    except Exception:
        logger.exception("run_slot_filler: LLM call failed for %s", action_cls.__name__)
        return None


# ── Prompt construction ────────────────────────────────────────────────────────


def _build_prompt(
    instruction: Optional[str],
    action_cls: type,
    free_slot_names: List[str],
    fixed_slots: Dict[str, Any],
    world_context: str,
) -> Tuple[str, List[str]]:
    """Build the slot-filler user message and the list of expected SlotValue field_names.

    COMPLEX slots are dropped from the expected list — those leaves surface as
    nested dotted field_names (``'grasp_description.grasp_type'``).
    """
    try:
        action_schema: Optional["ActionSchema"] = PycramIntrospector().introspect(
            action_cls
        )
    except Exception as exc:
        logger.debug(
            "_build_prompt: introspection failed for %s: %s",
            action_cls.__name__,
            exc,
        )
        action_schema = None

    # Resolve each requested slot name to its FieldSpec (including dotted sub-fields).
    name_set = set(free_slot_names)
    field_specs: Dict[str, "FieldSpec"] = {}
    if action_schema is not None:
        for fld in action_schema.fields:
            if fld.name in name_set:
                field_specs[fld.name] = fld
            for sub in fld.sub_fields:
                dotted = f"{fld.name}.{sub.name}"
                if dotted in name_set:
                    field_specs[dotted] = replace(sub, name=dotted)

    # Partition slots by kind and collect the LLM-visible expected names.
    entity_specs: List["FieldSpec"] = []
    param_specs: List["FieldSpec"] = []
    unknown_names: List[str] = []
    expected: List[str] = []
    for name in free_slot_names:
        fspec = field_specs.get(name)
        if fspec is None:
            unknown_names.append(name)
            expected.append(name)
        elif fspec.kind in (FieldKind.ENTITY, FieldKind.POSE, FieldKind.TYPE_REF):
            entity_specs.append(fspec)
            expected.append(name)
        elif fspec.kind == FieldKind.COMPLEX:
            # nested leaves surface as dotted names, not this parent.
            continue
        else:
            param_specs.append(fspec)
            expected.append(name)
    expected = list(dict.fromkeys(expected))

    # Header
    lines: List[str] = []
    if instruction:
        lines.append(f"Instruction: {instruction!r}")
    lines.append(f"Action type: {action_cls.__name__}")

    if expected:
        lines += [
            "",
            "Required free slot field_names:",
            *[f"  - {name}" for name in expected],
            (
                f"Return exactly {len(expected)} SlotValue entries, "
                "one for each field_name above. Do not omit enum or primitive "
                "slots when the instruction leaves them implicit; infer a "
                "reasonable action parameter from the action semantics and "
                "world context."
            ),
        ]

    action_doc = (
        action_schema.docstring
        if action_schema is not None
        else (inspect.getdoc(action_cls) or "").strip()
    )
    if action_doc:
        lines += ["", f"Action description: {action_doc}"]

    if entity_specs:
        lines += [
            "",
            "── Entity slots (world objects) ──────────────────────────────────────",
            "For each entity slot: return a SlotValue with entity_description populated.",
        ]
        for f in entity_specs:
            pose_note = (
                " [return .global_pose of the grounded body]"
                if f.kind == FieldKind.POSE
                else ""
            )
            doc_str = f": {f.docstring}" if f.docstring else ""
            lines.append(
                f"  - {f.name} ({_type_display(f.raw_type)}{pose_note}){doc_str}"
            )

    if param_specs:
        lines += [
            "",
            "── Parameter slots (discrete / primitive values) ──────────────────────",
            "For each: return a SlotValue with value = chosen string.",
        ]
        for f in param_specs:
            if f.enum_members:
                members_str = " | ".join(f.enum_members)
                doc_str = f" — {f.docstring}" if f.docstring else ""
                lines.append(f"  - {f.name} (allowed values: {members_str}){doc_str}")
            else:
                doc_str = f": {f.docstring}" if f.docstring else ""
                lines.append(f"  - {f.name} ({_type_display(f.raw_type)}){doc_str}")

    if unknown_names:
        lines += [
            "",
            "── Additional free slots (no type info — fill by best judgement) ────────",
            *[f"  - {name}" for name in unknown_names],
        ]

    if fixed_slots:
        lines += [
            "",
            "── Already-fixed slots (honour these, do not change) ─────────────────────",
        ]
        lines.extend(f"  - {fname} = {val!r}" for fname, val in fixed_slots.items())

    lines += ["", world_context]
    return "\n".join(lines), expected


def _invoke_with_repair(
    structured_llm: Any,
    action_cls: type,
    user_message: str,
    expected_slot_names: List[str],
) -> ActionReasoningOutput:
    """Invoke the slot-filler LLM; if slots are missing, make one repair call and merge."""
    output: ActionReasoningOutput = structured_llm.invoke(
        [
            {"role": "system", "content": _SLOT_FILLER_SYSTEM},
            {"role": "user", "content": user_message},
        ]
    )
    _normalise_output_slot_names(output, action_cls)

    returned = {slot.field_name for slot in output.slots}
    missing = [name for name in expected_slot_names if name not in returned]
    if not missing:
        return output

    logger.warning(
        "run_slot_filler: LLM omitted required slot(s) for %s: %s",
        action_cls.__name__,
        ", ".join(missing),
    )
    repair_message = "\n".join(
        [
            user_message,
            "",
            "Correction: the previous structured response omitted required free slots.",
            "Missing field_names:",
            *[f"  - {name}" for name in missing],
            "",
            (
                "Return a complete ActionReasoningOutput. Include one SlotValue "
                "for each required field_name:"
            ),
            *[f"  - {name}" for name in expected_slot_names],
            "",
            "Previous structured response:",
            output.model_dump_json(),
        ]
    )
    repaired: ActionReasoningOutput = structured_llm.invoke(
        [
            {"role": "system", "content": _SLOT_FILLER_SYSTEM},
            {"role": "user", "content": repair_message},
        ]
    )
    _normalise_output_slot_names(repaired, action_cls)

    merged = {slot.field_name: slot for slot in output.slots}
    for slot in repaired.slots:
        merged[slot.field_name] = slot
    return repaired.model_copy(update={"slots": list(merged.values())})


def _normalise_output_slot_names(
    output: ActionReasoningOutput,
    action_cls: type,
) -> None:
    """Strip root class prefixes from returned field names in-place."""
    for slot in output.slots:
        slot.field_name = _strip_root(slot.field_name, action_cls)


def _build_action_catalog(action_registry: Dict[str, type]) -> str:
    """Return classifier prompt lines with action names, docs, and field summaries."""
    try:
        introspector: Optional[PycramIntrospector] = PycramIntrospector()
    except Exception:
        introspector = None

    lines: List[str] = []
    for name in sorted(action_registry):
        action_cls = action_registry[name]
        schema = None
        if introspector is not None:
            try:
                schema = introspector.introspect(action_cls)
            except Exception:
                schema = None

        doc_source = (
            schema.docstring
            if schema is not None
            else (inspect.getdoc(action_cls) or "")
        )
        doc = " ".join(doc_source.split())
        if len(doc) > 180:
            doc = f"{doc[:177]}..."

        header = f"  - {name}"
        if doc:
            header += f": {doc}"
        lines.append(header)

        if schema is None or not schema.fields:
            continue
        field_summaries = ", ".join(
            f"{f.name}:{_type_display(f.raw_type)}"
            f"({'optional' if f.is_optional else 'required'})"
            for f in schema.fields
        )
        lines.append(f"    fields: {field_summaries}")

    return "\n".join(lines)


# ── Small inline helpers ──────────────────────────────────────────────────────


def _strip_root(name: str, action_cls: type) -> str:
    """Strip a leading ``<ActionClass>.`` prefix; preserve nested dotted paths."""
    prefix = f"{action_cls.__name__}."
    return name[len(prefix) :] if name.startswith(prefix) else name


def _type_display(raw_type: Any) -> str:
    """Clean display name for a raw type annotation."""
    return raw_type.__name__ if isinstance(raw_type, type) else str(raw_type)
