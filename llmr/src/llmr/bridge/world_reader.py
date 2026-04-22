"""World-state gateway: read SymbolGraph contents and resolve Symbol classes by name.

Single access point for every SymbolGraph query used by llmr:
  serialize_world_from_symbol_graph — LLM-readable world-context string.
  resolve_symbol_class              — semantic-type string → Symbol subclass.
  get_instances                     — safe wrapper over SymbolGraph.get_instances_of_type.
  body_display_name / body_xyz / body_bounding_box — duck-typed body helpers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing_extensions import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from krrood.symbol_graph.symbol_graph import Symbol, SymbolGraph

logger = logging.getLogger(__name__)

_ROBOT_ANNOTATION_TYPE_NAMES = frozenset(
    {
        "AbstractRobot",
        "SemanticRobotAnnotation",
        "KinematicChain",
        "Manipulator",
        "ParallelGripper",
        "HumanoidGripper",
        "Sensor",
        "Camera",
        "Base",
        "Torso",
        "Neck",
        "Arm",
        "Finger",
    }
)

# Subset shown in "Available Semantic Types" for LLM grounding.
# Includes abstract types (e.g. Manipulator) so instances appear under the type
# name the action schema uses, not just the concrete subclass name.
_ROBOT_CONTEXT_TYPE_NAMES = frozenset(
    {
        "Manipulator",
        "ParallelGripper",
        "HumanoidGripper",
        "Sensor",
        "Camera",
        "Base",
        "Torso",
        "Neck",
        "Arm",
        "Finger",
    }
)

# Fallback only: semantic robot annotations are preferred when present.
_STRUCTURAL_SUFFIXES = (
    "_link",
    "_frame",
    "_joint",
    "_screw",
    "_plate",
    "_optical_frame",
    "_motor",
    "_pad",
    "_finger",
)


def _is_structural_name(name: str) -> bool:
    """Fallback name heuristic for robot kinematic links when semantic metadata is unavailable."""
    return any(name.endswith(s) for s in _STRUCTURAL_SUFFIXES)


# ── World serialisation ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class WorldSerializationOptions:
    """Controls how much SymbolGraph detail is rendered for the LLM prompt."""

    max_objects: int = 40
    max_relations: int = 60
    include_geometry: bool = True
    include_parent_context: bool = True
    include_relations: bool = True
    include_structural: bool = False
    exclude_robot_structures: bool = True
    fallback_name_filter: bool = True
    structural_body_filter: Optional[Callable[[Any], bool]] = None
    precision: int = 3


@dataclass
class _BodyRecord:
    instance: Any
    body_name: str
    class_name: str
    semantic_types: List[str] = field(default_factory=list)
    parent_name: Optional[str] = None
    xyz: Optional[Tuple[float, float, float]] = None
    size: Optional[Tuple[float, float, float]] = None
    notes: List[str] = field(default_factory=list)


def serialize_world_from_symbol_graph(
    groundable_type: Type[Symbol] = Symbol,
    extra_context: str = "",
    symbol_graph: Optional[SymbolGraph] = None,
    options: Optional[WorldSerializationOptions] = None,
) -> str:
    """Build an LLM world-context string from SymbolGraph contents.

    :param groundable_type: Symbol subclass representing scene objects.
        Defaults to ``Symbol`` (all instances).  Pass a more specific caller
        type for a tighter scope covering only the intended world entities.
    :param extra_context: Optional extra text appended at the end.
    :param symbol_graph: Existing KRROOD SymbolGraph to query; defaults to the singleton.
    :param options: Optional rendering controls for prompt size and detail.
    :returns: Multi-line string describing the current world state.
    """
    opts = options or WorldSerializationOptions()

    try:
        graph = symbol_graph or SymbolGraph()
    except Exception:
        graph = None

    records: List[_BodyRecord] = []
    annotation_summary: Dict[str, List[str]] = {}
    relation_lines: List[str] = []

    try:
        if graph is None:
            raise RuntimeError("SymbolGraph unavailable")
        structural_body_ids = _collect_structural_body_ids(graph, opts)
        annotations_by_id, annotation_summary = _collect_annotations(
            graph,
            opts,
            structural_body_ids,
        )
        records = _collect_body_records(
            graph=graph,
            groundable_type=groundable_type,
            annotations_by_id=annotations_by_id,
            structural_body_ids=structural_body_ids,
            options=opts,
        )
        if opts.include_relations:
            visible_ids = {id(record.instance) for record in records}
            relation_lines = _collect_relation_lines(
                graph,
                visible_ids,
                structural_body_ids,
                opts,
            )
    except Exception:
        records = []
        annotation_summary = {}
        relation_lines = []

    return _render_world_context(
        records=records,
        annotation_summary=annotation_summary,
        relation_lines=relation_lines,
        extra_context=extra_context,
        options=opts,
    )


def _collect_body_records(
    graph: SymbolGraph,
    groundable_type: Type[Symbol],
    annotations_by_id: Dict[int, List[str]],
    structural_body_ids: Set[int],
    options: WorldSerializationOptions,
) -> List[_BodyRecord]:
    records: List[_BodyRecord] = []
    seen: Set[int] = set()

    for body in graph.get_instances_of_type(groundable_type):
        body_id = id(body)
        if body_id in seen:
            continue
        seen.add(body_id)

        body_name = body_display_name(body)
        if not body_name:
            body_name = f"{type(body).__name__}@{body_id:x}"
        if not options.include_structural and _is_structural_body(
            body,
            body_name,
            structural_body_ids,
            options,
        ):
            continue

        parent_name = (
            _nearest_parent_name(body) if options.include_parent_context else None
        )
        xyz = body_xyz(body) if options.include_geometry else None
        size = body_bounding_box(body) if options.include_geometry else None
        records.append(
            _BodyRecord(
                instance=body,
                body_name=body_name,
                class_name=type(body).__name__,
                semantic_types=sorted(set(annotations_by_id.get(body_id, []))),
                parent_name=parent_name,
                xyz=xyz,
                size=size,
            )
        )

    records.sort(key=lambda record: (record.body_name.lower(), record.class_name))
    return records


def _collect_annotations(
    graph: SymbolGraph,
    options: WorldSerializationOptions,
    structural_body_ids: Set[int],
) -> Tuple[Dict[int, List[str]], Dict[str, List[str]]]:
    annotations_by_id: Dict[int, List[str]] = {}
    annotation_summary: Dict[str, List[str]] = {}

    for wrapped in graph.wrapped_instances:
        inst = wrapped.instance
        if inst is None:
            continue

        ann_type = type(inst).__name__
        inst_mro_names = {cls.__name__ for cls in type(inst).__mro__}

        # Robot semantic annotations: checked BEFORE the .bodies guard because
        # Manipulator subclasses (e.g. ParallelGripper, SuctionGripper) don't
        # inherit KinematicChain and have no .bodies property.
        # MRO check catches any subclass of a known abstract type without requiring
        # explicit enumeration of every concrete robot component class.
        # Register the instance under every parent type in _ROBOT_CONTEXT_TYPE_NAMES
        # so the LLM sees it under the abstract type name the action schema uses
        # (e.g. Manipulator) not only under the concrete class name.
        if inst_mro_names & _ROBOT_ANNOTATION_TYPE_NAMES:
            inst_name = body_display_name(inst) or f"{ann_type}@{id(inst):x}"
            for cls in type(inst).__mro__:
                if cls.__name__ in _ROBOT_CONTEXT_TYPE_NAMES:
                    names_list = annotation_summary.setdefault(cls.__name__, [])
                    if inst_name not in names_list:
                        names_list.append(inst_name)
            continue

        bodies_attr = getattr(inst, "bodies", None)
        if bodies_attr is None:
            continue

        try:
            bodies = list(bodies_attr)
        except Exception:
            continue

        for body in bodies:
            b_name = body_display_name(body)
            if not b_name:
                b_name = f"{type(body).__name__}@{id(body):x}"
            if not options.include_structural and _is_structural_body(
                body,
                b_name,
                structural_body_ids,
                options,
            ):
                continue
            annotations_by_id.setdefault(id(body), []).append(ann_type)
            annotation_summary.setdefault(ann_type, []).append(b_name)

    for body_names in annotation_summary.values():
        body_names[:] = sorted(set(body_names), key=str.lower)
    return annotations_by_id, dict(sorted(annotation_summary.items()))


def _collect_relation_lines(
    graph: SymbolGraph,
    visible_ids: Set[int],
    structural_body_ids: Set[int],
    options: WorldSerializationOptions,
) -> List[str]:
    lines: List[str] = []
    seen: Set[str] = set()

    try:
        relations = list(graph.relations())
    except Exception:
        return []

    for relation in relations:
        source = getattr(getattr(relation, "source", None), "instance", None)
        target = getattr(getattr(relation, "target", None), "instance", None)
        if source is None or target is None:
            continue
        if id(source) not in visible_ids and id(target) not in visible_ids:
            continue

        source_name = (
            body_display_name(source) or f"{type(source).__name__}@{id(source):x}"
        )
        target_name = (
            body_display_name(target) or f"{type(target).__name__}@{id(target):x}"
        )
        if not options.include_structural and (
            _is_structural_body(source, source_name, structural_body_ids, options)
            or _is_structural_body(target, target_name, structural_body_ids, options)
        ):
            continue

        relation_name = str(relation)
        line = f"- {source_name} --{relation_name}--> {target_name}"
        if line not in seen:
            seen.add(line)
            lines.append(line)
        if len(lines) >= options.max_relations:
            break

    return sorted(lines, key=str.lower)


def _collect_structural_body_ids(
    graph: SymbolGraph,
    options: WorldSerializationOptions,
) -> Set[int]:
    if not options.exclude_robot_structures:
        return set()

    structural_body_ids: Set[int] = set()
    try:
        wrapped_instances = list(graph.wrapped_instances)
    except Exception:
        return structural_body_ids

    for wrapped in wrapped_instances:
        inst = wrapped.instance
        if inst is None or not _is_robot_annotation(inst):
            continue
        for body in _iter_robot_annotation_bodies(inst):
            structural_body_ids.add(id(body))

    return structural_body_ids


def _is_robot_annotation(instance: Any) -> bool:
    try:
        mro_names = {cls.__name__ for cls in type(instance).__mro__}
    except AttributeError:
        return False
    if mro_names & _ROBOT_ANNOTATION_TYPE_NAMES:
        return True
    return hasattr(instance, "_robot") and (
        hasattr(instance, "root")
        or hasattr(instance, "tool_frame")
        or hasattr(instance, "tip")
    )


def _iter_robot_annotation_bodies(instance: Any) -> List[Any]:
    bodies: List[Any] = []

    for attr_name in ("bodies", "kinematic_structure_entities"):
        try:
            values = getattr(instance, attr_name)
            if callable(values):
                values = values()
            bodies.extend(list(values))
        except Exception:
            pass

    for attr_name in (
        "root",
        "tip",
        "tool_frame",
        "torso",
        "base",
        "neck",
    ):
        value = getattr(instance, attr_name, None)
        if value is not None:
            bodies.append(value)

    for attr_name in (
        "manipulators",
        "sensors",
        "manipulator_chains",
        "sensor_chains",
        "arms",
    ):
        for child in _safe_iter(getattr(instance, attr_name, None)):
            bodies.extend(_iter_robot_annotation_bodies(child))

    result: List[Any] = []
    seen: Set[int] = set()
    for body in bodies:
        if body is None or id(body) in seen:
            continue
        seen.add(id(body))
        result.append(body)
    return result


def _safe_iter(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return []
    try:
        return list(value)
    except TypeError:
        return [value]
    except Exception:
        return []


def _is_structural_body(
    body: Any,
    body_name: str,
    semantic_structural_ids: Set[int],
    options: WorldSerializationOptions,
) -> bool:
    if options.structural_body_filter is not None:
        return options.structural_body_filter(body)
    if options.exclude_robot_structures and id(body) in semantic_structural_ids:
        return True
    return options.fallback_name_filter and _is_structural_name(body_name)


def _render_world_context(
    records: List[_BodyRecord],
    annotation_summary: Dict[str, List[str]],
    relation_lines: List[str],
    extra_context: str,
    options: WorldSerializationOptions,
) -> str:
    shown_records = records[: max(options.max_objects, 0)]
    unique_types = sorted(
        {ann_type for record in records for ann_type in record.semantic_types}
        | set(annotation_summary)
    )

    lines = [
        "## World State Summary",
        (
            f"Objects: {len(records)} visible"
            + (
                f" (showing {len(shown_records)})"
                if len(shown_records) != len(records)
                else ""
            )
            + f", Semantic types: {len(unique_types)}, Relations: {len(relation_lines)} shown"
        ),
        "",
        "## Grounding Instructions",
        "- Use exact body_name values for entity_description.name when possible.",
        "- Use semantic_type only from Available Semantic Types.",
        "- Use spatial_context for parent, surface, container, or proximity clues.",
        "- Use attributes only for visible distinguishing words such as color, size, or material.",
        "",
        "## Scene Objects",
    ]
    lines.extend(_render_scene_table(shown_records, options))

    if len(shown_records) < len(records):
        lines.append(
            f"- Truncated {len(records) - len(shown_records)} additional object(s)."
        )

    lines += ["", "## Available Semantic Types"]
    if annotation_summary:
        for ann_type, body_names in annotation_summary.items():
            lines.append(f"- {ann_type}: {', '.join(body_names)}")
    else:
        lines.append("- None found in this world.")

    lines += ["", "## Spatial Context"]
    spatial_lines = [
        f"- {record.body_name} is under/within parent {record.parent_name}"
        for record in shown_records
        if record.parent_name
    ]
    lines.extend(spatial_lines or ["- No parent or surface context found."])

    lines += ["", "## Symbol Relations"]
    lines.extend(relation_lines or ["- None found in this world."])

    if extra_context:
        lines += ["", "## Extra Context", extra_context]

    return "\n".join(lines)


def _render_scene_table(
    records: List[_BodyRecord],
    options: WorldSerializationOptions,
) -> List[str]:
    rows = [
        "| body_name | class | semantic_types | parent_or_surface | xyz | size | notes |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    if not records:
        rows.append(
            "| - | - | - | - | - | - | No scene objects found in SymbolGraph. |"
        )
        return rows

    for record in records:
        rows.append(
            "| "
            + " | ".join(
                [
                    record.body_name,
                    record.class_name,
                    ", ".join(record.semantic_types) if record.semantic_types else "-",
                    record.parent_name or "-",
                    _format_tuple(record.xyz, options.precision),
                    _format_tuple(record.size, options.precision),
                    ", ".join(record.notes) if record.notes else "-",
                ]
            )
            + " |"
        )
    return rows


def _nearest_parent_name(body: Any) -> Optional[str]:
    parent_conn = getattr(body, "parent_connection", None)
    parent = getattr(parent_conn, "parent", None) if parent_conn else None
    if parent is None:
        return None
    return body_display_name(parent) or f"{type(parent).__name__}@{id(parent):x}"


def _format_tuple(
    values: Optional[Tuple[float, float, float]],
    precision: int,
) -> str:
    if values is None:
        return "-"
    rounded = [round(value, precision) for value in values]
    return "(" + ", ".join(f"{value:g}" for value in rounded) + ")"


# ── Duck-type body helpers (public, reusable) ──────────────────────────────────
# These use plain getattr / duck typing.
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
            body.collision.as_bounding_box_collection_in_frame(ref)
            .bounding_box()
            .dimensions
        )
        return float(dims[0]), float(dims[1]), float(dims[2])
    except Exception:
        return None


# ── Symbol class / instance resolution ────────────────────────────────────────


def _camel_to_tokens(name: str) -> str:
    """Split a CamelCase class name into a lowercase token string for fuzzy matching."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()


def resolve_symbol_class(
    semantic_type: str,
    symbol_graph: Optional[SymbolGraph] = None,
) -> Optional[Type[Symbol]]:
    """Resolve a semantic-type string to a Symbol subclass via the SymbolGraph class diagram.

    Matches (in order) exact class name, camel-case token split, and any
    ``_synonyms`` classvar on the subclass.

    :param semantic_type: String from the LLM slot schema.
    :param symbol_graph: Existing SymbolGraph to query; defaults to the singleton.
    :return: Matching Symbol subclass, or ``None`` if nothing found.
    """
    query = semantic_type.strip().lower()
    query_tokens = query.replace("_", " ").replace("-", " ")

    try:
        class_diagram = (symbol_graph or SymbolGraph()).class_diagram
    except Exception:
        logger.debug(
            "SymbolGraph not yet initialised — cannot resolve '%s'.", semantic_type
        )
        return None

    for wrapped_cls in class_diagram.wrapped_classes:
        cls = wrapped_cls.clazz
        if cls.__name__.lower() == query:
            return cls
        if _camel_to_tokens(cls.__name__) == query_tokens:
            return cls
        synonyms = getattr(cls, "_synonyms", set())
        if any(s.lower() == query_tokens for s in synonyms):
            return cls

    return None


def get_instances(
    cls: Optional[type] = None,
    symbol_graph: Optional[SymbolGraph] = None,
) -> List[Any]:
    """Return SymbolGraph instances of *cls* (defaults to all :class:`Symbol` instances)."""
    try:
        graph = symbol_graph or SymbolGraph()
        return list(graph.get_instances_of_type(cls if cls is not None else Symbol))
    except Exception:
        return []
