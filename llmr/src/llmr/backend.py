"""LLMBackend — GenerativeBackend implementation that uses an LLM to fill underspecified Match slots.

World context is derived from SymbolGraph.  All krrood access goes through ``llmr.bridge``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Any, Callable, Dict, Iterable, Optional, Type

from krrood.entity_query_language.backends import GenerativeBackend
from krrood.entity_query_language.query.match import Match
from krrood.entity_query_language.utils import T
from krrood.symbol_graph.symbol_graph import Symbol

from llmr.bridge.match_reader import (
    finalize_match,
    read_match,
    unresolved_required_fields,
    write_slot_value,
)
from llmr.bridge.introspect import PycramIntrospector
from llmr.exceptions import LLMSlotFillingFailed, LLMUnresolvedRequiredFields
from llmr.resolution.slot_resolution import resolve_binding_value

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# ── Typed sentinel ─────────────────────────────────────────────────────────────


class _Unresolved:
    """Typed sentinel returned when a slot cannot be resolved."""

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

    world_context_provider: Optional[Callable[[], str]] = field(
        kw_only=True, default=None
    )
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

        # ── 1. Snapshot the Match expression into plain data ──────────────────
        introspector = PycramIntrospector()
        match_data = read_match(expression, introspector, unresolved=_UNRESOLVED)

        if not match_data.free_slots:
            yield finalize_match(match_data)
            return

        # ── 2. World context ───────────────────────────────────────────────────
        world_context = self._get_world_context()

        # ── 3. Slot filler (LLM call with dynamic prompt) ─────────────────────
        from llmr.reasoning.slot_filler import run_slot_filler

        output = run_slot_filler(
            instruction=self.instruction,
            action_cls=match_data.action_type,
            free_slot_names=match_data.free_slot_names,
            fixed_slots=match_data.fixed_bindings,
            world_context=world_context,
            llm=self.llm,
        )
        if output is None:
            raise LLMSlotFillingFailed(action_name=match_data.action_name)

        # ── 4. Resolve each free slot and write it back into the Match ─────────
        from llmr.resolution.grounder import EntityGrounder

        grounder = EntityGrounder(self.groundable_type)
        slot_by_name = {slot.field_name: slot for slot in output.slots}
        # Successfully resolved top-level values are threaded into nested entity
        # auto-grounding (e.g. arm → matching Manipulator).
        resolved_params: Dict[str, Any] = {}

        for slot in match_data.free_slots:
            resolved = resolve_binding_value(
                slot=slot,
                slot_by_name=slot_by_name,
                grounder=grounder,
                resolved_params=resolved_params,
                unresolved=_UNRESOLVED,
            )

            if resolved is _UNRESOLVED:
                logger.debug(
                    "LLMBackend: field '%s' unresolved — leaving as default.",
                    slot.attribute_name,
                )
                continue

            resolved_params[slot.attribute_name] = resolved
            write_slot_value(slot, resolved)

        if self.strict_required:
            unresolved = unresolved_required_fields(match_data, introspector)
            if unresolved:
                raise LLMUnresolvedRequiredFields(
                    action_name=match_data.action_name,
                    unresolved_fields=unresolved,
                )

        yield finalize_match(match_data)

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
        from llmr.bridge.world_reader import serialize_world_from_symbol_graph

        return serialize_world_from_symbol_graph(self.groundable_type)
