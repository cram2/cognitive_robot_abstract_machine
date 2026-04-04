from __future__ import annotations

import inspect
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, Field, fields
from enum import Enum
from functools import lru_cache

import pydot
import rustworkx as rx
from typing_extensions import Any, Type, ClassVar, Dict, List, Optional, Tuple, Generator

from krrood.ripple_down_rules.datastructures.field_info import FieldInfo
from krrood.ripple_down_rules import logger
from krrood.ripple_down_rules.rules import Rule
from krrood.ripple_down_rules.utils import recursive_subclasses


class Direction(Enum):
    OUTBOUND = False
    INBOUND = True


class Relation(str, Enum):
    has = "has"
    isA = "isA"
    dependsOn = "dependsOn"


@dataclass(unsafe_hash=True)
class TrackedObjectMixin:
    """
    A class that is used as a base class to all classes that needs to be tracked for RDR inference, and reasoning.
    """
    _rdr_tracked_object_id: int = field(init=False, repr=False, hash=False,
                                        compare=False, default_factory=lambda: uuid.uuid4().int)
    """
    The unique identifier of the conclusion.
    """
    _dependency_graph: ClassVar[rx.PyDAG[Type[TrackedObjectMixin]]] = rx.PyDAG()
    """
    A graph that represents the relationships between all tracked objects.
    """
    _class_graph_indices: ClassVar[Dict[Type[TrackedObjectMixin], int]] = {}
    """
    The index of the current class in the dependency graph.
    """
    _edges: ClassVar[Dict[Relation, List[Tuple[int, int]]]] = defaultdict(list)
    """
    All the edges indexed by relation type.
    """
    _overridden_by: Type[TrackedObjectMixin] = field(init=False, repr=False, hash=False,
                                                     compare=False, default=None)
    """
    Whether the class has been overridden by a subclass.
    This is used to only include the new class in the dependency graph, not the overridden class.
    """
    _tracked_classes: ClassVar[List[Type]] = []
    """
    A list of all tracked classes.
    """

    @classmethod
    def _reset_dependency_graph(cls) -> None:
        """
        Reset the dependency graph and all class indices.
        """
        cls._dependency_graph = rx.PyDAG()
        cls._class_graph_indices = {}
        cls._edges = defaultdict(list)
        cls.parse_fields.cache_clear()
        cls.get_fields.cache_clear()
        cls.parse_field.cache_clear()

    @classmethod
    def _my_graph_idx(cls):
        return cls._class_graph_indices[cls]

    @classmethod
    def _get_classes_to_track(cls):
        if not cls._tracked_classes:
            subclasses = recursive_subclasses(cls)
            extra_classes = [Rule] + recursive_subclasses(Rule)
            for clazz in subclasses + extra_classes:
                cls._track_class(clazz)
        return cls._tracked_classes

    @classmethod
    def _track_class(cls, clazz):
        if clazz in cls._tracked_classes:
            return
        cls._tracked_classes.append(clazz)
        if not issubclass(clazz, TrackedObjectMixin):
            clazz._overridden_by = None

    @classmethod
    def _inheritance_relations(cls):
        for clazz in cls._get_classes_to_track():
            bases = [base for base in clazz.__bases__ if
                     base.__module__ not in ["builtins"] and base in cls._get_classes_to_track()]

            for base in bases:
                if hasattr(base, "_overridden_by") and base._overridden_by is not clazz:
                    yield clazz, base

    @classmethod
    def _composition_relations(cls):
        for clazz in cls._get_classes_to_track():
            if clazz.__module__ == "builtins":
                continue
            yield from TrackedObjectMixin.parse_fields(clazz)

    @staticmethod
    @lru_cache(maxsize=None)
    def parse_fields(clazz) -> Generator[Tuple[Type[TrackedObjectMixin], Type[TrackedObjectMixin]], None, None]:
        for f in TrackedObjectMixin.get_fields(clazz):

            logger.debug("=" * 80)
            logger.debug(f"Processing Field {clazz.__name__}.{f.name}: {f.type}.")

            # skip private fields
            if f.name.startswith("_"):
                logger.debug(f"Skipping since the field starts with _.")
                continue

            field_info = FieldInfo(clazz, f)
            yield from TrackedObjectMixin.parse_field(field_info)

    @staticmethod
    @lru_cache(maxsize=None)
    def get_fields(clazz) -> List[Field]:
        skip_fields = []
        bases = [base for base in clazz.__bases__ if issubclass(base, TrackedObjectMixin)]
        for base in bases:
            skip_fields.extend(TrackedObjectMixin.get_fields(base))

        result = [cls_field for cls_field in fields(clazz) if cls_field not in skip_fields]

        return result

    @staticmethod
    @lru_cache(maxsize=None)
    def parse_field(field_info: FieldInfo)\
            -> Generator[Tuple[Type[TrackedObjectMixin], Type[TrackedObjectMixin]], None, None]:
        parent = field_info.clazz
        field_cls: Optional[Type[TrackedObjectMixin]] = None
        if len(field_info.type) == 1 and issubclass(field_info.type[0], TrackedObjectMixin):
            field_cls = field_info.type[0]
        else:
            # TODO: Create a new TrackedObjectMixin class for new type
            logger.debug(f"Skipping unhandled field type: {field_info.type}")
            # logger.debug(f"Creating new TrackedObject type for builtin type {field_info.type}.")
            # field_cls = cls._create_tracked_object_class_for_field(field_info)

        if field_cls is not None:
            yield parent, field_cls

    @classmethod
    def add_edge(cls, parent_idx: int, child_idx: int, relation: Relation):
        if (parent_idx, child_idx) not in cls._edges[relation]:
            cls._dependency_graph.add_edge(parent_idx, child_idx, relation)
            cls._edges[relation].append((parent_idx, child_idx))

    @classmethod
    def _create_tracked_object_class_for_field(cls, field_info: FieldInfo):
        raise NotImplementedError

    @classmethod
    def to_dot(cls, filepath: str, format='svg') -> None:
        if not filepath.endswith(f".{format}"):
            filepath += f".{format}"
        dot_str = cls._dependency_graph.to_dot(
            lambda node: dict(
                color='black', fillcolor='lightblue', style='filled', label=node.__name__),
            lambda edge: dict(color='black', style='solid', label=edge))
        dot = pydot.graph_from_dot_data(dot_str)[0]
        try:
            dot.write(filepath, format=format)
        except FileNotFoundError:
            tmp_filepath = filepath.replace(f".{format}", ".dot")
            dot.write(tmp_filepath, format='raw')
            try:
                os.system(f"/usr/bin/dot -T{format} {tmp_filepath} -o {filepath}")
                os.remove(tmp_filepath)
            except Exception as e:
                logger.error(e)

    @classmethod
    def _add_class_to_dependency_graph(cls, class_to_add: Type[TrackedObjectMixin]) -> None:
        """
        Add a class to the dependency graph.
        """
        if class_to_add not in cls._class_graph_indices:
            cls_idx = cls._dependency_graph.add_node(class_to_add._overridden_by or class_to_add)
            cls._class_graph_indices[class_to_add] = cls_idx
            if class_to_add._overridden_by:
                cls._class_graph_indices[class_to_add._overridden_by] = cls_idx


annotations = TrackedObjectMixin.__annotations__
for val in [f.name for f in fields(TrackedObjectMixin)]:
    annotations.pop(val, None)

X = TrackedObjectMixin
