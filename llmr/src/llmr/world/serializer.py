"""
World serializer — converts SymbolGraph state into an LLM-readable string.

Package-independent: uses SymbolGraph (krrood) as the sole data source.

The caller passes groundable_type (for example a Body class from its world
package) so this module never imports a world directly.
"""
from __future__ import annotations

import logging
from typing_extensions import Any, Dict, List, Optional, Tuple, Type

from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph

logger = logging.getLogger(__name__)

# Kinematic-link name suffixes — these are robot structural parts, not scene objects.
# Suffix-based filter is representation-agnostic (unlike the old "/" heuristic which
# relied on a particular prefixed-name string format).
_STRUCTURAL_SUFFIXES = (
    "_link", "_frame", "_joint", "_screw", "_plate",
    "_optical_frame", "_motor", "_pad", "_finger",
)


def _is_structural_link(name: str) -> bool:
    """True if *name* ends with a robot kinematic-link suffix — these are filtered from scene listings."""
    return any(name.endswith(s) for s in _STRUCTURAL_SUFFIXES)


def serialize_world_from_symbol_graph(
    groundable_type: Type[Symbol] = Symbol,
    extra_context: str = "",
) -> str:
    """Build an LLM world-context string from SymbolGraph contents.

    :param groundable_type: Symbol subclass representing scene objects.
        Defaults to ``Symbol`` (all instances).  Pass a more specific caller
        type for a tighter scope covering only the intended world entities.
    :param extra_context: Optional extra text appended at the end.
    :returns: Multi-line string describing the current world state.
    """
    lines = ["## World State Summary\n"]

    # ── Scene objects ──────────────────────────────────────────────────────────
    try:
        all_instances = list(SymbolGraph().get_instances_of_type(groundable_type))
        all_names = [body_display_name(b) for b in all_instances]
        scene_names = [n for n in all_names if not _is_structural_link(n)]
        if scene_names:
            lines.append(f"Scene objects and surfaces: {', '.join(scene_names)}")
        elif all_names:
            lines.append(f"Bodies present: {', '.join(all_names[:30])}")
            if len(all_names) > 30:
                lines.append(f"  … and {len(all_names) - 30} more.")
        else:
            lines.append("No scene objects found in SymbolGraph.")
    except Exception:
        lines.append("Bodies: unavailable")

    # ── Semantic annotations ───────────────────────────────────────────────────
    lines.append("\n## Semantic annotations")
    try:
        ann_summary: Dict[str, List[str]] = {}
        graph = SymbolGraph()
        for wrapped in graph.wrapped_instances:
            inst = wrapped.instance
            if inst is None:
                continue
            bodies_attr = getattr(inst, "bodies", None)
            if bodies_attr is None:
                continue
            ann_type = type(inst).__name__
            try:
                for body in bodies_attr:
                    b_name = body_display_name(body)
                    if _is_structural_link(b_name):
                        continue
                    ann_summary.setdefault(b_name, []).append(ann_type)
            except Exception:
                pass

        if ann_summary:
            unique_types = sorted({t for types in ann_summary.values() for t in types})
            lines.append(f"Available types: {', '.join(unique_types)}")
            lines.append("Per body:")
            for body_name, types in ann_summary.items():
                lines.append(f"  {body_name}: {', '.join(types)}")
        else:
            lines.append("  None found in this world.")
    except Exception:
        lines.append("  (unavailable)")

    if extra_context:
        lines.append(extra_context)

    return "\n".join(lines)


# ── Duck-type body helpers (public, reusable) ──────────────────────────────────
# These use plain getattr / duck typing
# If name/pose attribute conventions change, fix only here.


def body_display_name(body: Any) -> str:
    """Return a clean display name for a body instance (hides PrefixedName chain)."""
    name_obj = getattr(body, "name", None)
    if name_obj is None:
        return ""
    if hasattr(name_obj, "name"):
        return str(name_obj.name)
    return str(name_obj)


def body_xyz(body: Any) -> Optional[Tuple[float, float, float]]:
    """Return (x, y, z) position of a body, or None if unavailable."""
    try:
        pt = body.global_pose.to_position()
        return float(pt.x), float(pt.y), float(pt.z)
    except Exception:
        return None


def body_bounding_box(
    body: Any,
    reference_frame: Optional[Any] = None,
) -> Optional[Tuple[float, float, float]]:
    """Return (depth, width, height) bounding box dims, or None if unavailable."""
    try:
        ref = reference_frame if reference_frame is not None else body
        dims = (
            body.collision
            .as_bounding_box_collection_in_frame(ref)
            .bounding_box()
            .dimensions
        )
        return float(dims[0]), float(dims[1]), float(dims[2])
    except Exception:
        return None
