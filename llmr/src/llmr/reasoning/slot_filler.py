"""
Slot filler — dynamic LLM prompts driven by pycram action introspection.

Two public functions:

  classify_action()  — NL instruction → action class (for nl_plan factory)
  run_slot_filler()  — action class + free slots → ActionReasoningOutput
"""
from __future__ import annotations

import inspect
import logging
import typing
from dataclasses import replace
from typing_extensions import Any, Dict, List, Optional

if typing.TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from llmr.pycram_bridge.introspector import ActionSchema, FieldSpec

logger = logging.getLogger(__name__)

from llmr.exceptions import LLMActionRegistryEmpty
from llmr._utils import slot_prompt_name as _slot_prompt_name
from llmr.schemas.slots import (
    ActionClassification,
    ActionReasoningOutput,
)


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
      name           = noun phrase from the instruction (head noun only, no articles)
      semantic_type  = ontological type from world annotations (null if unknown)
      spatial_context = spatial qualifier from instruction ("on the table") or null
      attributes     = discriminating key/value attributes (color, size) or null
  - reasoning   = 1-2 sentences explaining which world object was identified

The entity_description is used for symbolic grounding in SymbolGraph.
The name must be the local part of the body name visible in the world context.

────────────────────────────────────────────────────
PARAMETER SLOTS  (enum, primitive, complex sub-fields)
────────────────────────────────────────────────────
Return a SlotValue with:
  - field_name = the parameter name exactly as listed
  - value      = chosen value as a string (exact enum member name or primitive)
  - reasoning  = 1-2 sentences justifying the choice

For ENUM slots use EXACTLY one of the listed allowed values. Never paraphrase,
translate, or describe enum values in natural language.
For COMPLEX fields use dotted names (e.g. 'grasp_description.grasp_type').
For ENTITY slots inside complex fields (e.g. 'grasp_description.manipulator'),
return a SlotValue with entity_description populated — treat them like any other
entity slot. For Manipulator fields, choose a concrete manipulator/gripper name
from the world context; never use the robot/platform name.

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
    :param action_registry: Pre-built {class_name: class} dict; auto-discovered by the bridge if None.
    :returns: The matched action class, or None if classification fails.
    """
    if action_registry is None:
        from llmr.pycram_bridge import discover_action_classes
        action_registry = discover_action_classes()
    if not action_registry:
        raise LLMActionRegistryEmpty()

    class_list = _format_action_catalog(action_registry)
    system = _CLASSIFIER_SYSTEM.format(action_classes=class_list)

    structured_llm = llm.with_structured_output(ActionClassification)
    try:
        result: ActionClassification = structured_llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ])
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

    The prompt is built dynamically from PycramIntrospector output:
      - ENTITY/POSE fields get an entity-description section with role, type, and docstring
        (includes Manipulator, Camera — all Symbol subclasses ground from SymbolGraph)
      - ENUM fields list all valid member names from the actual Enum class
      - COMPLEX fields are expanded into dotted sub-field entries

    :param instruction:     Natural-language instruction.
    :param action_cls:      The action dataclass being resolved.
    :param free_slot_names: Field names (from the Match expression) to fill.
    :param fixed_slots:     Already-resolved {field_name: value} map.
    :param world_context:   Serialized world state string.
    :param llm:             LangChain-compatible chat model.
    :returns: ActionReasoningOutput on success; None on LLM failure.
    """
    # Normalise names: strip the root 'ClassName.' prefix while preserving
    # nested paths such as 'grasp_description.approach_direction'.
    slot_names = [_slot_prompt_name(n, action_cls) for n in free_slot_names]
    prompt_fixed_slots = {
        _slot_prompt_name(k, action_cls): v for k, v in fixed_slots.items()
    }

    # ── Introspect action class → field specs ──────────────────────────────────
    action_schema = _introspect_action_schema(action_cls)
    field_specs = _select_free_slot_specs(action_schema, slot_names)

    # ── Build dynamic user message ─────────────────────────────────────────────
    user_message = _build_dynamic_message(
        instruction=instruction,
        action_cls=action_cls,
        action_schema=action_schema,
        free_slot_names=slot_names,
        fixed_slots=prompt_fixed_slots,
        world_context=world_context,
        field_specs=field_specs,
    )

    structured_llm = llm.with_structured_output(ActionReasoningOutput)
    try:
        output: ActionReasoningOutput = structured_llm.invoke([
            {"role": "system", "content": _SLOT_FILLER_SYSTEM},
            {"role": "user", "content": user_message},
        ])
        return output
    except Exception:
        logger.exception("run_slot_filler: LLM call failed for %s", action_cls.__name__)
        return None


# ── Prompt construction ────────────────────────────────────────────────────────

def _introspect_action_schema(action_cls: type) -> Optional["ActionSchema"]:
    """Return the KRROOD-backed action schema, or None when introspection fails."""
    try:
        from llmr.pycram_bridge.introspector import PycramIntrospector

        return PycramIntrospector().introspect(action_cls)
    except Exception as exc:
        logger.debug(
            "_introspect_action_schema: introspection failed for %s: %s",
            action_cls.__name__, exc,
        )
        return None


def _select_free_slot_specs(
    action_schema: Optional["ActionSchema"],
    free_slot_names: List[str],
) -> Dict[str, "FieldSpec"]:
    """Return a {field_name: FieldSpec} map for the given free slot names."""
    if action_schema is None:
        return {}

    slot_names = set(free_slot_names)
    selected: Dict[str, "FieldSpec"] = {}
    for field in action_schema.fields:
        if field.name in slot_names:
            selected[field.name] = field
        for sub_field in field.sub_fields:
            dotted_name = f"{field.name}.{sub_field.name}"
            if dotted_name in slot_names:
                selected[dotted_name] = replace(sub_field, name=dotted_name)
    return selected


def _build_dynamic_message(
    instruction: Optional[str],
    action_cls: type,
    action_schema: Optional["ActionSchema"],
    free_slot_names: List[str],
    fixed_slots: Dict[str, Any],
    world_context: str,
    field_specs: Dict[str, "FieldSpec"],
) -> str:
    """Build the rich user message for the slot-filler LLM call.

    Sections:
      - Action type + docstring (from action class)
      - Entity slots: role, type name, docstring
      - Parameter slots: name, type, allowed enum values, docstring
      - Complex slots: expanded into dotted sub-field entries
      - Fixed slots: already-resolved values (do not change)
      - World context
    """
    from llmr.pycram_bridge.introspector import FieldKind

    lines: List[str] = [f"Action type: {action_cls.__name__}"]
    if instruction:
        lines.insert(0, f"Instruction: {instruction!r}")

    # Action class docstring from the same schema that provides field metadata.
    action_doc = (
        action_schema.docstring
        if action_schema is not None
        else (inspect.getdoc(action_cls) or "").strip()
    )
    if action_doc:
        lines += ["", f"Action description: {action_doc}"]

    # ── Classify free slots by kind ────────────────────────────────────────────
    entity_specs: List["FieldSpec"] = []
    param_specs: List["FieldSpec"] = []    # ENUM + PRIMITIVE
    complex_specs: List["FieldSpec"] = []  # COMPLEX
    unknown_names: List[str] = []          # no introspection data

    for name in free_slot_names:
        fspec = field_specs.get(name)
        if fspec is None:
            unknown_names.append(name)
            continue
        if fspec.kind in (FieldKind.ENTITY, FieldKind.POSE, FieldKind.TYPE_REF):
            entity_specs.append(fspec)
        elif fspec.kind == FieldKind.ENUM:
            param_specs.append(fspec)
        elif fspec.kind == FieldKind.PRIMITIVE:
            param_specs.append(fspec)
        elif fspec.kind == FieldKind.COMPLEX:
            complex_specs.append(fspec)

    # ── Entity slots section ───────────────────────────────────────────────────
    if entity_specs:
        lines += ["", "── Entity slots (world objects) ──────────────────────────────────────"]
        lines.append(
            "For each entity slot: return a SlotValue with entity_description populated."
        )
        for fspec in entity_specs:
            type_name = (
                fspec.raw_type.__name__
                if isinstance(fspec.raw_type, type) else str(fspec.raw_type)
            )
            pose_note = " [return .global_pose of the grounded body]" if fspec.kind == FieldKind.POSE else ""
            doc_str = f": {fspec.docstring}" if fspec.docstring else ""
            lines.append(f"  - {fspec.name} ({type_name}{pose_note}){doc_str}")

    # ── Parameter slots section ────────────────────────────────────────────────
    if param_specs:
        lines += ["", "── Parameter slots (discrete / primitive values) ──────────────────────"]
        lines.append("For each: return a SlotValue with value = chosen string.")
        for fspec in param_specs:
            type_name = (
                fspec.raw_type.__name__
                if isinstance(fspec.raw_type, type) else str(fspec.raw_type)
            )
            if fspec.enum_members:
                members_str = " | ".join(fspec.enum_members)
                doc_str = f" — {fspec.docstring}" if fspec.docstring else ""
                lines.append(
                    f"  - {fspec.name} (allowed values: {members_str}){doc_str}"
                )
            else:
                doc_str = f": {fspec.docstring}" if fspec.docstring else ""
                lines.append(f"  - {fspec.name} ({type_name}){doc_str}")

    # ── Complex slots section ──────────────────────────────────────────────────
    if complex_specs:
        lines += ["", "── Complex slots (expand into dotted sub-field SlotValues) ────────────"]
        lines.append(
            "For each complex field, return individual SlotValues with dotted names."
        )
        for fspec in complex_specs:
            type_name = (
                fspec.raw_type.__name__
                if isinstance(fspec.raw_type, type) else str(fspec.raw_type)
            )
            doc_str = f": {fspec.docstring}" if fspec.docstring else ""
            lines.append(f"  {fspec.name} ({type_name}){doc_str}")
            if fspec.sub_fields:
                for sub in fspec.sub_fields:
                    sub_type = (
                        sub.raw_type.__name__
                        if isinstance(sub.raw_type, type) else str(sub.raw_type)
                    )
                    if sub.enum_members:
                        members_str = " | ".join(sub.enum_members)
                        sub_doc = f" — {sub.docstring}" if sub.docstring else ""
                        lines.append(
                            f"    {fspec.name}.{sub.name} "
                            f"(allowed values: {members_str}){sub_doc}"
                        )
                    else:
                        sub_doc = f": {sub.docstring}" if sub.docstring else ""
                        if "manipulator" in sub_type.lower():
                            sub_doc += (
                                " Choose a concrete gripper/manipulator name, "
                                "not the robot name."
                            )
                        lines.append(
                            f"    {fspec.name}.{sub.name} ({sub_type}){sub_doc}"
                        )

    # ── Unknown / fallback slots ───────────────────────────────────────────────
    if unknown_names:
        lines += ["", "── Additional free slots (no type info — fill by best judgement) ────────"]
        for name in unknown_names:
            lines.append(f"  - {name}")

    # ── Fixed slots ────────────────────────────────────────────────────────────
    if fixed_slots:
        lines += ["", "── Already-fixed slots (honour these, do not change) ─────────────────────"]
        for fname, val in fixed_slots.items():
            lines.append(f"  - {fname} = {val!r}")

    # ── World context ──────────────────────────────────────────────────────────
    lines += ["", world_context]

    return "\n".join(lines)


def _format_action_catalog(action_registry: Dict[str, type]) -> str:
    """Return classifier prompt lines with action names, docs, and field summaries."""
    try:
        from llmr.pycram_bridge import PycramIntrospector

        introspector = PycramIntrospector()
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

        doc = " ".join(
            (
                schema.docstring
                if schema is not None
                else (inspect.getdoc(action_cls) or "")
            ).split()
        )
        if len(doc) > 180:
            doc = f"{doc[:177]}..."

        header = f"  - {name}"
        if doc:
            header += f": {doc}"
        lines.append(header)

        if schema is None:
            continue
        if schema.fields:
            fields = ", ".join(_format_catalog_field(field) for field in schema.fields)
            lines.append(f"    fields: {fields}")

    return "\n".join(lines)


def _format_catalog_field(fspec: "FieldSpec") -> str:
    type_name = (
        fspec.raw_type.__name__
        if isinstance(fspec.raw_type, type)
        else str(fspec.raw_type)
    )
    required = "optional" if fspec.is_optional else "required"
    return f"{fspec.name}:{type_name}({required})"
