"""Entity grounder — resolves NL entity descriptions to Symbol instances.

Two-tier strategy:
  Tier 1  Annotation-based: resolve ``semantic_type`` to a Symbol subclass, collect
          its annotated bodies (via ``.bodies``) or fall back to the annotation itself.
  Tier 2  Name-based: substring-match ``description.name`` across all instances of
          ``groundable_type``.

All SymbolGraph access is delegated to :mod:`llmr.bridge.world_reader`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing_extensions import Any, List, Optional

from llmr.bridge.world_reader import (
    body_display_name,
    get_instances,
    resolve_symbol_class,
)
from llmr.schemas import EntityDescriptionSchema

logger = logging.getLogger(__name__)


# ── Result type ────────────────────────────────────────────────────────────────


@dataclass
class GroundingResult:
    """Result of an entity grounding attempt."""

    bodies: List[Any] = field(default_factory=list)
    """Candidate Symbol instances that match the description, ranked by confidence."""

    warning: Optional[str] = None
    """Non-fatal diagnostic message (e.g. multiple matches, fallback used)."""


# ── EntityGrounder ─────────────────────────────────────────────────────────────


@dataclass
class EntityGrounder:
    """Grounds an :class:`EntityDescriptionSchema` to Symbol instances in the world.

    All SymbolGraph queries go through the bridge — this class stays krrood-free.
    """

    groundable_type: Any = None
    """Symbol subclass representing groundable world entities (e.g. ``Body``).
    ``None`` falls back to ``Symbol`` via :func:`get_instances` (all instances)."""

    symbol_graph: Any = None
    """Optional SymbolGraph to query; ``None`` uses the singleton."""

    # ── Main entry point ───────────────────────────────────────────────────────

    def ground(
        self,
        description: EntityDescriptionSchema,
        expected_type: Optional[type] = None,
    ) -> GroundingResult:
        """Resolve *description* to Symbol instances (Tier 1, then Tier 2).

        :param expected_type: Python type the caller needs (e.g. ``Manipulator``).
            Passed to Tier 1 so annotation instances are returned directly when
            they are themselves of the required type, rather than expanding to
            their physical bodies.
        """
        if description.semantic_type:
            result = self._annotation_ground(description, expected_type=expected_type)
            if result.bodies:
                logger.debug(
                    "Tier 1 grounding: semantic_type=%r → %d instance(s): %s",
                    description.semantic_type,
                    len(result.bodies),
                    [body_display_name(b) for b in result.bodies],
                )
                return result
            logger.debug(
                "Annotation grounding for type '%s' returned no results, "
                "falling back to name search.",
                description.semantic_type,
            )

        result = self._name_ground(description)
        if result.bodies:
            return result

        warning = (
            f"No instances found for '{description.name}' "
            f"(semantic_type={description.semantic_type!r}). "
            "Check that the object exists in the world."
        )
        logger.warning(warning)
        return GroundingResult(bodies=[], warning=warning)

    # ── Tier 1: annotation-based ───────────────────────────────────────────────

    def _annotation_ground(
        self,
        description: EntityDescriptionSchema,
        expected_type: Optional[type] = None,
    ) -> GroundingResult:
        """Resolve ``semantic_type`` to a class, then collect its annotated bodies.

        When *expected_type* is provided and annotation instances are already of
        that type, return the annotations directly instead of expanding to their
        physical bodies.  This handles robot-semantic-annotation fields such as
        ``Manipulator`` where the action slot wants the annotation object, not a
        kinematic link.
        """
        cls = resolve_symbol_class(
            description.semantic_type,
            symbol_graph=self.symbol_graph,
        )
        if cls is None:
            if expected_type is not None:
                # Class diagram may be empty or unpopulated; use the known field type directly.
                cls = expected_type
            else:
                logger.debug(
                    "Cannot resolve '%s' to a Symbol subclass.",
                    description.semantic_type,
                )
                return GroundingResult()

        annotations = get_instances(cls, self.symbol_graph)
        if not annotations:
            return GroundingResult()

        # If the annotation instances themselves are of expected_type, return them
        # directly — no body expansion needed (e.g. Manipulator field → Manipulator instance).
        if expected_type is not None and isinstance(annotations[0], expected_type):
            candidates = list(annotations)
            if description.name and candidates:
                name_lower = description.name.lower()
                name_filtered = [
                    a for a in candidates if name_lower in body_display_name(a).lower()
                ]
                if name_filtered:
                    candidates = name_filtered
            candidates = self._refine(candidates, description)
            return GroundingResult(
                bodies=candidates,
                warning=self._multi_match_warning(candidates, description.name),
            )

        candidates: List[Any] = []
        for ann in annotations:
            try:
                for body in ann.bodies:
                    if body not in candidates:
                        candidates.append(body)
            except AttributeError:
                # Annotation has no .bodies — treat the annotation itself as groundable.
                if ann not in candidates:
                    candidates.append(ann)

        if description.name and candidates:
            name_lower = description.name.lower()
            name_filtered = [
                b for b in candidates if name_lower in body_display_name(b).lower()
            ]
            if name_filtered:
                candidates = name_filtered

        candidates = self._refine(candidates, description)
        return GroundingResult(
            bodies=candidates,
            warning=self._multi_match_warning(candidates, description.name),
        )

    # ── Tier 2: name-based ─────────────────────────────────────────────────────

    def _name_ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        """Substring-scan ``description.name`` over all groundable_type instances."""
        if not description.name:
            return GroundingResult()

        name_lower = description.name.lower()
        all_instances = get_instances(self.groundable_type, self.symbol_graph)

        candidates = [
            b for b in all_instances if name_lower in body_display_name(b).lower()
        ]

        if not candidates:
            return GroundingResult()

        candidates = self._refine(candidates, description)
        return GroundingResult(
            bodies=candidates,
            warning=self._multi_match_warning(candidates, description.name),
        )

    # ── Refinement ─────────────────────────────────────────────────────────────

    def _refine(
        self, candidates: List[Any], description: EntityDescriptionSchema
    ) -> List[Any]:
        """Narrow *candidates* with an attribute-value filter; skipped when only one remains."""
        if description.attributes and len(candidates) > 1:
            refined = self._filter_by_attributes(candidates, description.attributes)
            if refined:
                candidates = refined
        return candidates

    @staticmethod
    def _filter_by_attributes(candidates: List[Any], attributes: dict) -> List[Any]:
        """Retain candidates whose display name or annotation types contain any attribute value."""
        filtered: List[Any] = []
        for body in candidates:
            body_str = body_display_name(body).lower()
            ann_type_names = " ".join(
                type(a).__name__.lower()
                for a in getattr(body, "_semantic_annotations", [])
            )
            combined = body_str + " " + ann_type_names
            for value in attributes.values():
                if value.lower() in combined:
                    filtered.append(body)
                    break
        return filtered if filtered else candidates

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _multi_match_warning(
        candidates: List[Any], name: Optional[str]
    ) -> Optional[str]:
        """Return a warning string when grounding is ambiguous (> 1 candidate), else ``None``."""
        if len(candidates) > 1:
            names = [body_display_name(b) for b in candidates]
            return (
                f"Grounding for '{name}' returned {len(candidates)} candidates: "
                f"{names}. All passed to the action handler."
            )
        return None
