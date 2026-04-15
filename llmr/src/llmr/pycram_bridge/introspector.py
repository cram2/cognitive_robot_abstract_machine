"""PycramIntrospector — classifies action dataclass fields by how they should be resolved.

FieldKind classification drives the slot-filler prompt and per-slot resolution in LLMBackend:
  ENTITY    Symbol subclass → SymbolGraph grounding, instance passed directly.
  POSE      Pose/HomogeneousTransformationMatrix → grounded entity's .global_pose.
  ENUM      Enum subclass → string-to-enum coercion.
  COMPLEX   Non-primitive dataclass (e.g. GraspDescription) → recursive sub-field construction.
  PRIMITIVE bool / int / float / str → taken directly from LLM output.
  TYPE_REF  Type[X] annotation → resolved to a Symbol subclass via SymbolGraph class diagram.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing_extensions import Any, ClassVar, Dict, List, Type

from krrood.class_diagrams.attribute_introspector import (
    AttributeIntrospector,
    DiscoveredAttribute,
)
from krrood.class_diagrams.class_diagram import ClassDiagram
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.symbol_graph.symbol_graph import Symbol

import logging

logger = logging.getLogger(__name__)


# ── Field classification ───────────────────────────────────────────────────────


class FieldKind(str, Enum):
    """How an action field should be resolved — drives prompt generation and slot resolution."""
    ENTITY    = "entity"
    POSE      = "pose"
    ENUM      = "enum"
    COMPLEX   = "complex"
    PRIMITIVE = "primitive"
    TYPE_REF  = "type_ref"


# Sentinel meaning "field has no default" — cannot use dataclasses.MISSING
# directly as a dataclass field default because the dataclasses machinery
# interprets MISSING as "no default", causing TypeError on class creation.
NO_DEFAULT = object()


@dataclass
class FieldSpec:
    """Metadata for one action dataclass field, used to build slot-filler prompts."""

    name: str
    raw_type: Any                           # resolved Python type (not a string)
    kind: FieldKind
    docstring: str = ""                     # attribute docstring from class source
    is_optional: bool = False
    default: Any = field(default_factory=lambda: NO_DEFAULT)  # NO_DEFAULT means required
    enum_members: List[str] = field(default_factory=list)      # for ENUM kind
    sub_fields: List["FieldSpec"] = field(default_factory=list) # for COMPLEX kind


@dataclass
class ActionSchema:
    """Full introspection result for one action class — action type, docstring, and per-field specs."""

    action_type: str       # e.g. "PickUpAction"
    action_cls: Any        # the actual class object
    docstring: str         # class-level docstring
    fields: List[FieldSpec]


# ── PycramIntrospector ─────────────────────────────────────────────────────────


@dataclass
class OwnDataclassIntrospector(AttributeIntrospector):
    """Discover only fields declared directly on the inspected dataclass."""

    def discover(self, owner_cls: Type) -> List[DiscoveredAttribute]:
        own_names = set(getattr(owner_cls, "__annotations__", {}))
        if not dataclasses.is_dataclass(owner_cls):
            return []
        return [
            DiscoveredAttribute(public_name=f.name, field=f)
            for f in dataclasses.fields(owner_cls)
            if f.name in own_names and not f.name.startswith("_") and f.init
        ]


@dataclass
class PycramIntrospector:
    """Reads an action dataclass and classifies each field into a FieldKind.

    Results drive the slot-filler prompt (which fields to ask the LLM about, what
    types/enum members to list) and LLMBackend resolution (how to ground or coerce
    each slot value returned by the LLM).

    Usage::

        schema = PycramIntrospector().introspect(PickUpAction)
    """

    # Names of types (and their subclasses via MRO) treated as spatial poses.
    # Matched against every class in the type's MRO — no world-package import needed.
    POSE_TYPE_NAMES: ClassVar[frozenset] = frozenset({"Pose", "HomogeneousTransformationMatrix"})

    introspector: AttributeIntrospector = field(
        default_factory=OwnDataclassIntrospector
    )

    def introspect(self, action_cls: type, _depth: int = 0) -> ActionSchema:
        """Return an :class:`ActionSchema` for *action_cls*.

        :param action_cls: An action dataclass.
        """
        if not dataclasses.is_dataclass(action_cls):
            raise TypeError(f"{action_cls!r} is not a dataclass")

        cls_doc = (inspect.getdoc(action_cls) or "").strip()
        field_docs = self._extract_field_docstrings(action_cls)
        class_diagram = ClassDiagram([action_cls], introspector=self.introspector)
        wrapped_class = class_diagram.get_wrapped_class(action_cls)

        field_specs: List[FieldSpec] = []
        for wrapped_field in wrapped_class.fields:
            field_specs.append(
                self._field_spec_from_wrapped_field(
                    wrapped_field=wrapped_field,
                    field_docs=field_docs,
                    depth=_depth,
                )
            )

        return ActionSchema(
            action_type=action_cls.__name__,
            action_cls=action_cls,
            docstring=cls_doc,
            fields=field_specs,
        )

    def _field_spec_from_wrapped_field(
        self,
        wrapped_field: WrappedField,
        field_docs: Dict[str, str],
        depth: int,
    ) -> FieldSpec:
        """Convert KRROOD field metadata into llmr's prompt-facing schema."""
        if wrapped_field.is_type_type:
            raw_type = wrapped_field.type_endpoint
            kind = FieldKind.TYPE_REF
        elif wrapped_field.is_container and not wrapped_field.is_optional:
            raw_type = wrapped_field.resolved_type
            kind = self.classify_type(raw_type, depth)
        else:
            raw_type = wrapped_field.type_endpoint
            kind = self.classify_type(raw_type, depth)
        dc_field = wrapped_field.field

        has_default = dc_field.default is not dataclasses.MISSING
        has_factory = dc_field.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        if has_default:
            field_default = dc_field.default
        elif has_factory:
            field_default = dc_field.default_factory  # type: ignore[misc]
        else:
            field_default = NO_DEFAULT

        spec = FieldSpec(
            name=wrapped_field.name,
            raw_type=raw_type,
            kind=kind,
            docstring=field_docs.get(wrapped_field.name, ""),
            is_optional=wrapped_field.is_optional or has_default or has_factory,
            default=field_default,
        )

        if kind == FieldKind.ENUM:
            spec.enum_members = list(raw_type.__members__.keys())

        elif kind == FieldKind.COMPLEX and depth < 2:
            try:
                sub_schema = self.introspect(raw_type, _depth=depth + 1)
                spec.sub_fields = sub_schema.fields
            except Exception as exc:
                logger.debug("Cannot introspect sub-fields of %s: %s", raw_type, exc)

        elif kind == FieldKind.TYPE_REF:
            args = typing.get_args(raw_type) or typing.get_args(wrapped_field.resolved_type)
            if args:
                spec.raw_type = args[0]

        return spec

    # ── Type classification ────────────────────────────────────────────────────

    def classify_type(self, t: Any, depth: int = 0) -> FieldKind:
        """Return the :class:`FieldKind` for a resolved Python type *t*.

        Symbol subclasses, including robot components such as manipulators and
        cameras, are classified as ENTITY and grounded from SymbolGraph.
        """
        if t is None or t is type(None):
            return FieldKind.PRIMITIVE

        # typing.get_origin tells us if it's Type[X]
        origin = typing.get_origin(t)
        if origin is type:
            return FieldKind.TYPE_REF

        # Primitive scalars
        if t in (bool, int, float, str, bytes):
            return FieldKind.PRIMITIVE

        if not isinstance(t, type):
            return FieldKind.PRIMITIVE

        # Enum (check before dataclass since Enum subclasses are not dataclasses)
        if issubclass(t, Enum):
            return FieldKind.ENUM

        # Spatial pose types, matched by class name to avoid importing a world package.
        if self._is_pose_type(t):
            return FieldKind.POSE

        # Symbol / Entity types (Body, Region, Manipulator, Camera, etc.)
        # All Symbol subclasses are grounded from SymbolGraph — no injection needed.
        if self._is_entity_type(t):
            return FieldKind.ENTITY

        # Complex dataclass → recursive construction
        if dataclasses.is_dataclass(t):
            return FieldKind.COMPLEX

        return FieldKind.PRIMITIVE

    # ── Type predicates ───────────────────────────────────────────────────────

    def _is_entity_type(self, t: type) -> bool:
        """True if *t* is a Symbol subclass — grounds to a SymbolGraph instance."""
        try:
            return issubclass(t, Symbol)
        except TypeError:
            return False

    def _is_pose_type(self, t: type) -> bool:
        """True if *t* is a spatial pose type — matched by name across the MRO, no world-package import."""
        try:
            return any(cls.__name__ in self.POSE_TYPE_NAMES for cls in t.__mro__)
        except (TypeError, AttributeError):
            return False

    # ── Attribute docstring extraction ─────────────────────────────────────────

    @staticmethod
    def _extract_field_docstrings(cls: type) -> Dict[str, str]:
        """Parse class source with AST to extract attribute-level docstrings.

        The action classes document each field as a string literal immediately following
        the annotated assignment::

            object_designator: Body
            \"\"\"
            Object designator describing the object to pick up.
            \"\"\"

        Standard Python introspection doesn't expose these, but AST parsing does.
        """
        try:
            raw_src = inspect.getsource(cls)
            source = textwrap.dedent(raw_src)
            tree = ast.parse(source)
        except Exception:
            return {}

        docs: Dict[str, str] = {}
        # Find the class body (first ClassDef in the module)
        class_body: List[ast.stmt] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_body = node.body
                break

        for i, node in enumerate(class_body):
            if not isinstance(node, ast.AnnAssign):
                continue
            target = node.target
            if not isinstance(target, ast.Name):
                continue
            fname = target.id
            if i + 1 < len(class_body):
                nxt = class_body[i + 1]
                if (
                    isinstance(nxt, ast.Expr)
                    and isinstance(nxt.value, ast.Constant)
                    and isinstance(nxt.value.value, str)
                ):
                    docs[fname] = nxt.value.value.strip()

        return docs


def introspect(action_cls: type) -> ActionSchema:
    """Introspect *action_cls* using a fresh introspector instance."""
    return PycramIntrospector().introspect(action_cls)
