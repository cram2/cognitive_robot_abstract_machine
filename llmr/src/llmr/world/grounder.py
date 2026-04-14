"""Entity grounder — resolves NL entity descriptions to Symbol instances.

Resolution strategy
-------------------
Tier 1  Annotation-based: find instances whose *class name* matches the
        ``semantic_type`` string extracted by the LLM, then optionally
        cross-reference with the groundable type's ``.bodies`` attribute.

Tier 2  Name-based: duck-type access to ``.name`` / ``.name.name`` on every
        groundable instance, substring-matching the extracted name string.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing_extensions import Any, List, Optional, Tuple, Type

from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph

from llmr.schemas.entities import EntityDescriptionSchema
from llmr.world.serializer import body_bounding_box, body_display_name, body_xyz

logger = logging.getLogger(__name__)


# ── Result type ────────────────────────────────────────────────────────────────


@dataclass
class GroundingResult:
    """Result of an entity grounding attempt."""

    bodies: List[Any] = field(default_factory=list)
    """Candidate Symbol instances that match the description, ranked by confidence."""

    warning: Optional[str] = None
    """Non-fatal diagnostic message (e.g. multiple matches, fallback used)."""


# ── Annotation class resolution via SymbolGraph ───────────────────────────────


def _camel_to_tokens(name: str) -> str:
    """Split a CamelCase class name into a lowercase token string for fuzzy matching."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()


def resolve_symbol_class(semantic_type: str) -> Optional[Type[Symbol]]:
    """Resolve a semantic type string to a Symbol subclass via the SymbolGraph class diagram.

    Walks ``SymbolGraph().class_diagram`` — all Symbol subclasses are registered
    there at instantiation time, so no world-package import is needed.

    :param semantic_type: String from the LLM slot schema.
    :return: Matching Symbol subclass, or ``None`` if nothing found.
    """
    query = semantic_type.strip().lower()
    query_tokens = query.replace("_", " ").replace("-", " ")

    try:
        class_diagram = SymbolGraph().class_diagram
    except Exception:
        logger.debug("SymbolGraph not yet initialised — cannot resolve '%s'.", semantic_type)
        return None

    for wrapped_cls in class_diagram.wrapped_classes:
        cls = wrapped_cls.clazz
        # 1. exact class name
        if cls.__name__.lower() == query:
            return cls
        # 2. camel-case expanded
        if _camel_to_tokens(cls.__name__) == query_tokens:
            return cls
        # 3. _synonyms classvar (Set[str]) if present
        synonyms = getattr(cls, "_synonyms", set())
        if any(s.lower() == query_tokens for s in synonyms):
            return cls

    return None


# ── EntityGrounder ─────────────────────────────────────────────────────────────


@dataclass
class EntityGrounder:
    """Grounds an :class:`EntityDescriptionSchema` to Symbol instances in the world.

    Uses :class:`~krrood.symbol_graph.symbol_graph.SymbolGraph` as the sole
    data source — no world object or world-package import required.
    """

    groundable_type: Type[Symbol] = Symbol
    """The Symbol subclass representing groundable world entities (e.g. Body).
    Defaults to Symbol (all instances in SymbolGraph). Pass a more specific
    subclass to narrow the search pool."""

    # ── Main entry point ───────────────────────────────────────────────────────

    def ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        """Resolve *description* to Symbol instances.

        Tries annotation grounding (Tier 1) first, then name-based (Tier 2).
        Applies spatial/attribute refinement when multiple candidates remain.

        :param description: LLM-extracted entity description.
        :return: :class:`GroundingResult` with matching instances and diagnostic info.
        """
        if description.semantic_type:
            result = self._annotation_ground(description)
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

    def _annotation_ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        """Tier 1: resolve semantic_type to a Symbol subclass, collect its annotated bodies."""
        cls = resolve_symbol_class(description.semantic_type)
        if cls is None:
            logger.debug(
                "Cannot resolve '%s' to a Symbol subclass.", description.semantic_type
            )
            return GroundingResult()

        try:
            annotations = list(SymbolGraph().get_instances_of_type(cls))
        except Exception as exc:
            logger.warning("SymbolGraph.get_instances_of_type raised: %s", exc)
            return GroundingResult()

        if not annotations:
            return GroundingResult()

        # Collect groundable instances via the annotation's .bodies attribute (duck typing)
        candidates: List[Any] = []
        for ann in annotations:
            try:
                for body in ann.bodies:
                    if body not in candidates:
                        candidates.append(body)
            except AttributeError:
                # Annotation has no .bodies — treat the annotation itself as groundable
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
        """Tier 2: substring scan of description.name over all groundable_type instances."""
        if not description.name:
            return GroundingResult()

        name_lower = description.name.lower()
        try:
            all_instances = list(SymbolGraph().get_instances_of_type(self.groundable_type))
        except Exception as exc:
            logger.warning("SymbolGraph.get_instances_of_type raised: %s", exc)
            return GroundingResult()

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
        """Narrow *candidates* using spatial_context then attribute filters; skips if only one candidate."""
        if description.spatial_context and len(candidates) > 1:
            refined = self._filter_by_spatial_context(candidates, description.spatial_context)
            if refined:
                candidates = refined

        if description.attributes and len(candidates) > 1:
            refined = self._filter_by_attributes(candidates, description.attributes)
            if refined:
                candidates = refined

        return candidates

    def _filter_by_spatial_context(
        self, candidates: List[Any], spatial_context: str
    ) -> List[Any]:
        """Filter to candidates near surfaces matching *spatial_context*; falls back to name-proximity subtree."""
        context_lower = spatial_context.lower()

        # Try to find a surface annotation type by name from SymbolGraph class diagram
        surface_cls = resolve_symbol_class("HasSupportingSurface")
        if surface_cls is not None:
            try:
                surface_annotations = list(SymbolGraph().get_instances_of_type(surface_cls))
                matched_surfaces = [
                    ann for ann in surface_annotations
                    if _camel_to_tokens(type(ann).__name__) in context_lower
                    or context_lower in _camel_to_tokens(type(ann).__name__)
                ]
                if matched_surfaces:
                    proximity_filtered = [
                        c for c in candidates if self._near_any_surface(c, matched_surfaces)
                    ]
                    if proximity_filtered:
                        return proximity_filtered
            except Exception as exc:
                logger.debug("Surface-based spatial filter failed: %s", exc)

        # Fallback: anchor body name substring match via SymbolGraph
        try:
            all_instances = list(SymbolGraph().get_instances_of_type(self.groundable_type))
        except Exception:
            return candidates

        anchor_bodies = [
            b for b in all_instances
            if body_display_name(b).lower() in context_lower
        ]
        if not anchor_bodies:
            return candidates

        def _in_subtree(body: Any, anchor: Any) -> bool:
            current = body
            while current is not None:
                if current is anchor:
                    return True
                parent_conn = getattr(current, "parent_connection", None)
                current = getattr(parent_conn, "parent", None) if parent_conn else None
            return False

        tree_filtered = [
            c for c in candidates if any(_in_subtree(c, anchor) for anchor in anchor_bodies)
        ]
        return tree_filtered if tree_filtered else candidates

    @staticmethod
    def _near_any_surface(body: Any, surfaces: list) -> bool:
        """True if *body*'s z-position is at or above the top of any surface in *surfaces*."""
        try:
            xyz = body_xyz(body)
            if xyz is None:
                return True
            bz = xyz[2]
            for ann in surfaces:
                try:
                    ann_body = ann.bodies[0]
                    ann_xyz = body_xyz(ann_body)
                    dims = body_bounding_box(ann_body)
                    if ann_xyz is not None and dims is not None:
                        surface_top_z = ann_xyz[2] + dims[2] / 2
                        if bz >= surface_top_z - 0.05:
                            return True
                except Exception:
                    continue
        except Exception:
            pass
        return False

    def _filter_by_attributes(
        self, candidates: List[Any], attributes: dict
    ) -> List[Any]:
        """Retain candidates whose display name or annotation types contain any attribute value."""
        filtered = []
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
    def _multi_match_warning(candidates: List[Any], name: Optional[str]) -> Optional[str]:
        """Return a warning string when grounding is ambiguous (> 1 candidate), else None."""
        if len(candidates) > 1:
            names = [body_display_name(b) for b in candidates]
            return (
                f"Grounding for '{name}' returned {len(candidates)} candidates: "
                f"{names}. All passed to the action handler."
            )
        return None


# ── Module-level convenience ───────────────────────────────────────────────────


def ground_entity(
    description: EntityDescriptionSchema,
    groundable_type: Type[Symbol] = Symbol,
) -> GroundingResult:
    """Convenience wrapper around :class:`EntityGrounder`.

    :param description: LLM-extracted entity description.
    :param groundable_type: The Symbol subclass to search in SymbolGraph.
        Defaults to ``Symbol`` (all instances).
    :return: :class:`GroundingResult`.
    """
    return EntityGrounder(groundable_type).ground(description)
