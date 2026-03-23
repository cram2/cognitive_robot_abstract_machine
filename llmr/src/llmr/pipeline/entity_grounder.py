
from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing_extensions import List, Optional, Type

from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    HasSupportingSurface,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import (
    Body,
    SemanticAnnotation,
)

from llmr.workflows._utils import _pose_to_xyz
from llmr.workflows.schemas.common import EntityDescriptionSchema

logger = logging.getLogger(__name__)


# ── Public result type ─────────────────────────────────────────────────────────


@dataclass
class GroundingResult:
    """Result of an entity grounding attempt."""

    bodies: List[Body] = field(default_factory=list)
    """Candidate Body objects that match the description, ranked by confidence."""

    # kept for backwards compatibility
    @property
    def used_eql(self) -> bool:
        return False

    warning: Optional[str] = None
    """Non-fatal diagnostic message (e.g. multiple matches, fallback used)."""


# ── Annotation class registry ──────────────────────────────────────────────────


def _all_annotation_subclasses() -> List[Type[SemanticAnnotation]]:
    """Return all concrete (non-abstract) SemanticAnnotation subclasses."""
    result: List[Type[SemanticAnnotation]] = []

    def _recurse(cls: type) -> None:
        for sub in cls.__subclasses__():
            if not inspect.isabstract(sub):
                result.append(sub)
            _recurse(sub)

    _recurse(SemanticAnnotation)
    return result


def _camel_to_tokens(name: str) -> str:
    """'DrinkingContainer' → 'drinking container'."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()


def resolve_annotation_class(semantic_type: str) -> Optional[Type[SemanticAnnotation]]:
    """Resolve a semantic type string to a SemanticAnnotation subclass.

    :param semantic_type: String from the LLM slot schema.
    :return: Matching class, or ``None`` if nothing found.
    """
    query = semantic_type.strip().lower()
    query_tokens = query.replace("_", " ").replace("-", " ")

    for cls in _all_annotation_subclasses():
        # 1. exact class name
        if cls.__name__.lower() == query:
            return cls
        # 2. camel-case expanded
        if _camel_to_tokens(cls.__name__) == query_tokens:
            return cls
        # 3. _synonyms classvar (Set[str] on the class)
        synonyms = getattr(cls, "_synonyms", set())
        if any(s.lower() == query_tokens for s in synonyms):
            return cls

    return None


# ── EntityGrounder ─────────────────────────────────────────────────────────────


class EntityGrounder:
    """Grounds an ``EntityDescriptionSchema`` to ``Body`` objects in the world."""

    def __init__(self, world: World) -> None:
        self._world = world

    # ── Main entry point ───────────────────────────────────────────────────────

    def ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        """Resolve an entity description to world bodies.

        Tries annotation grounding (Tier 1) first, then name-based (Tier 2).
        Applies spatial / attribute refinement when multiple candidates remain.

        :param description: LLM-extracted entity description.
        :return: ``GroundingResult`` with matching bodies and diagnostic info.
        """
        # ── semantic annotation type ──────────────────────────────────
        if description.semantic_type:
            result = self._annotation_ground(description)
            if result.bodies:
                logger.debug(
                    "Tier 1 grounding: semantic_type=%r → %d body/bodies: %s",
                    description.semantic_type,
                    len(result.bodies),
                    [str(getattr(getattr(b, "name", None), "name", b)) for b in result.bodies],
                )
                return result
            logger.debug(
                "Annotation grounding for type '%s' returned no results, "
                "falling back to name search.",
                description.semantic_type,
            )

        # ── body name substring ───────────────────────────────────────
        result = self._name_ground(description)
        if result.bodies:
            return result

        # ── Nothing found ─────────────────────────────────────────────────────
        warning = (
            f"No bodies found for '{description.name}' "
            f"(semantic_type={description.semantic_type!r}). "
            "Check that the object exists in the world."
        )
        logger.warning(warning)
        return GroundingResult(bodies=[], warning=warning)

    # ── annotation-based grounding ────────────────────────────────────

    def _annotation_ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        """Ground via SDT semantic annotation type."""
        cls = resolve_annotation_class(description.semantic_type)
        if cls is None:
            logger.debug(
                "Cannot resolve '%s' to a SemanticAnnotation subclass.",
                description.semantic_type,
            )
            return GroundingResult()

        try:
            annotations = self._world.get_semantic_annotations_by_type(cls)
        except Exception as exc:
            logger.warning("get_semantic_annotations_by_type raised: %s", exc)
            return GroundingResult()

        if not annotations:
            return GroundingResult()

        candidates: List[Body] = []
        for ann in annotations:
            for body in ann.bodies:
                if body not in candidates:
                    candidates.append(body)

        if description.name and candidates:
            name_lower = description.name.lower()
            name_filtered = [b for b in candidates if name_lower in self._body_name(b).lower()]
            if name_filtered:
                candidates = name_filtered

        candidates = self._refine(candidates, description)

        return GroundingResult(
            bodies=candidates,
            warning=self._multi_match_warning(candidates, description.name),
        )

    # ── name-based grounding ──────────────────────────────────────────

    def _name_ground(self, description: EntityDescriptionSchema) -> GroundingResult:
        """Ground by substring-matching the entity name over all world bodies."""
        if not description.name:
            return GroundingResult()

        name_lower = description.name.lower()
        candidates = [b for b in self._world.bodies if name_lower in self._body_name(b).lower()]

        if not candidates:
            return GroundingResult()

        candidates = self._refine(candidates, description)
        return GroundingResult(
            bodies=candidates,
            warning=self._multi_match_warning(candidates, description.name),
        )

    # ── refinement ─────────────────────────────────────────────────

    def _refine(self, candidates: List[Body], description: EntityDescriptionSchema) -> List[Body]:
        """Apply spatial context and attribute filters to narrow candidates."""
        if description.spatial_context and len(candidates) > 1:
            refined = self._filter_by_spatial_context(candidates, description.spatial_context)
            if refined:
                candidates = refined

        if description.attributes and len(candidates) > 1:
            refined = self._filter_by_attributes(candidates, description.attributes)
            if refined:
                candidates = refined

        return candidates

    # ── Spatial context filter ─────────────────────────────────────────────────

    def _filter_by_spatial_context(
        self, candidates: List[Body], spatial_context: str
    ) -> List[Body]:
        """Narrow candidates using a spatial context hint."""
        context_lower = spatial_context.lower()

        try:
            surface_annotations = self._world.get_semantic_annotations_by_type(
                HasSupportingSurface
            )
            matched_surfaces = [
                ann
                for ann in surface_annotations
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

        anchor_bodies = [
            b for b in self._world.bodies if self._body_name(b).lower() in context_lower
        ]
        if not anchor_bodies:
            return candidates

        def _in_subtree(body: Body, anchor: Body) -> bool:
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

    def _near_any_surface(self, body: Body, surfaces: list) -> bool:
        """Return True if *body* is positioned above any of the *surfaces*."""
        try:
            body_xyz = _pose_to_xyz(body.global_pose)
            if body_xyz is None:
                return True
            bz = body_xyz[2]

            for ann in surfaces:
                try:
                    bb = ann.as_bounding_box_collection_in_frame(self._world.root).bounding_box()
                    dims = bb.dimensions
                    surface_top_z = float(_pose_to_xyz(ann.bodies[0].global_pose)[2]) + dims[2] / 2
                    if bz >= surface_top_z - 0.05:
                        return True
                except Exception:
                    continue
        except Exception:
            pass
        return False

    # ── Attribute filter ───────────────────────────────────────────────────────

    def _filter_by_attributes(self, candidates: List[Body], attributes: dict) -> List[Body]:
        """Filter by key/value attributes from the entity description."""
        filtered = []
        for body in candidates:
            body_str = self._body_name(body).lower()
            ann_type_names = " ".join(
                type(a).__name__.lower() for a in getattr(body, "_semantic_annotations", [])
            )
            combined = body_str + " " + ann_type_names
            for value in attributes.values():
                if value.lower() in combined:
                    filtered.append(body)
                    break
        return filtered if filtered else candidates

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _body_name(body: Body) -> str:
        """Return a normalised string name for a Body."""
        name_obj = getattr(body, "name", None)
        if name_obj is None:
            return ""
        if hasattr(name_obj, "name"):
            return str(name_obj.name)
        return str(name_obj)

    @staticmethod
    def _multi_match_warning(candidates: List[Body], name: str) -> Optional[str]:
        if len(candidates) > 1:
            names = [str(getattr(getattr(b, "name", None), "name", b)) for b in candidates]
            return (
                f"Grounding for '{name}' returned {len(candidates)} candidates: "
                f"{names}. All passed to PartialDesignator."
            )
        return None


# ── Module-level convenience function ─────────────────────────────────────────


def ground_entity(
    description: EntityDescriptionSchema,
    world: World,
) -> GroundingResult:
    """Convenience wrapper around ``EntityGrounder.ground``.

    :param description: LLM-extracted entity description.
    :param world: The SDT world instance.
    :return: ``GroundingResult``.
    """
    return EntityGrounder(world).ground(description)
