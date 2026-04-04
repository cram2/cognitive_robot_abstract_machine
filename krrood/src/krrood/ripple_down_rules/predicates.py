from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing_extensions import Generator, ClassVar

from typing_extensions import Type, TYPE_CHECKING, Tuple, Dict, List, TypeVar, Iterator, Union

from krrood.ripple_down_rules.datastructures.enums import InferMode
from krrood.ripple_down_rules.datastructures.tracked_object import TrackedObjectMixin, Direction, Relation
from krrood.ripple_down_rules.types import PredicateArgType, PredicateOutputType
from krrood.ripple_down_rules.utils import is_iterable

if TYPE_CHECKING:
    pass


@dataclass(eq=False)
class Predicate(TrackedObjectMixin, ABC):

    cached_results: ClassVar[Dict[Tuple, List]] = defaultdict(list)
    """
    Cached results between retrievals, makes future retrievals faster,
    """
    inferred_once: ClassVar[bool] = None
    """
    If inference has been performed at least once.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.inferred_once = False

    @classmethod
    @abstractmethod
    def relation(cls) -> Relation:
        """
        The relation type of the predicate.
        """

    def __call__(self, *args, infer: InferMode = InferMode.Auto, **kwargs):
        if self.should_infer(mode=infer):
            self.infer(*args, **kwargs)
            self.cached_results.clear()
            self.__class__.inferred_once = True
        key = (args, tuple(sorted(kwargs.items())))
        if key in self.cached_results:
            yield from self.cached_results[key]
        else:
            for value in self.retrieve(*args, **kwargs):
                self.cached_results[key].append(value)
                yield value

    @classmethod
    def should_infer(cls, mode: InferMode) -> bool:
        """
        Determine if the predicate relations should be inferred or just retrieve current relations.

        :param mode: The infer mode of the predicate (could be Auto, Always, Never)
        """
        match mode:
            case InferMode.Always:
                return True
            case InferMode.Never:
                return False
            case InferMode.Auto:
                return cls._should_infer()
            case _:
                raise ValueError(f"Invalid infer mode: {mode}")

    @classmethod
    @abstractmethod
    def _should_infer(cls) -> bool:
        """
        Predicate specific reasoning on when to infer relations.
        """

    @classmethod
    def infer(cls, *args, **kwargs):
        """
        Evaluate the predicate with the given arguments.
        This method should be implemented by subclasses.
        """
        results = cls._infer(*args, **kwargs)
        for parent, child in results:
            cls._add_class_to_dependency_graph(parent)
            cls._add_class_to_dependency_graph(child)
            cls.add_edge(cls._class_graph_indices[parent], cls._class_graph_indices[child], cls.relation())

    @classmethod
    @abstractmethod
    def _infer(cls, *args, **kwargs)\
            -> Generator[Tuple[Type[TrackedObjectMixin], Type[TrackedObjectMixin]], None, None]:
        """
        Evaluate the predicate with the given arguments.
        This method should be implemented by subclasses.
        """

    @classmethod
    def retrieve(cls, node1_type: PredicateArgType, node2_type: PredicateArgType,
                 recursive: bool = False, is_reversed: bool = False) -> PredicateOutputType:
        if not is_iterable(node1_type):
            node1_type = (node1_type,)
        if not is_iterable(node2_type):
            node2_type = (node2_type,)
        for n1 in node1_type:
            for n2 in node2_type:
                if n1 is not TrackedObjectMixin:
                    direction = Direction.INBOUND.value if is_reversed else Direction.OUTBOUND.value
                    owner_idx = n1._my_graph_idx()
                    neighbors = cls._dependency_graph.adj_direction(owner_idx, direction)
                    neighbor_generator = ((owner_idx, n) for n, e in neighbors.items()
                                          if (e == cls.relation() and issubclass(cls._dependency_graph.get_node_data(n),
                                                                                 n2))
                                          or (e == Relation.isA and any(cls.retrieve(cls._dependency_graph.get_node_data(n),
                                                                                     n2)))
                                          )
                    latest_results = []
                    for v in neighbor_generator:
                        res = (cls._dependency_graph.get_node_data(v[0]), cls._dependency_graph.get_node_data(v[1]))
                        latest_results.append(v)
                        yield res
                    if recursive:
                        for n in [n for n, e in neighbors.items() if e == cls.relation()]:
                            for v in cls.retrieve(cls._dependency_graph.get_node_data(n), n2, recursive=True,
                                                  is_reversed=is_reversed):
                                yield n1, v[1]
                elif n2 is not TrackedObjectMixin:
                    # owner type is TrackedObjectMixin
                    yield from map(lambda t: (t[1], t[0]),
                                   cls.retrieve(n2, n1, recursive=recursive, is_reversed=True))
                else:
                    # both are TrackedObjectMixin
                    yield from map(
                        lambda t: (cls._dependency_graph.get_node_data(t[0]), cls._dependency_graph.get_node_data(t[1])),
                        cls._edges[cls.relation()])

    def __hash__(self):
        return hash(self.__class__.__name__)

    def __eq__(self, other):
        if not isinstance(other, Predicate):
            return False
        return self.__class__ == other.__class__


@dataclass(eq=False)
class IsA(Predicate):
    """
    A predicate that checks if an object type is a subclass of another object type.
    """

    @classmethod
    def relation(cls):
        return Relation.isA

    @classmethod
    def _infer(cls, *args, **kwargs):
        yield from TrackedObjectMixin._inheritance_relations()

    @classmethod
    def _should_infer(cls) -> bool:
        return not cls.inferred_once


isA = IsA()


@dataclass(eq=False)
class Has(Predicate):
    """
    A predicate that checks if an object type has a certain member object type.
    """
    @classmethod
    def relation(cls):
        return Relation.has

    @classmethod
    def _infer(cls, *args, **kwargs):
        yield from TrackedObjectMixin._composition_relations()

    @classmethod
    def _should_infer(cls) -> bool:
        return not cls.inferred_once


has = Has()


@dataclass(eq=False)
class DependsOn(Predicate):
    """
    A predicate that checks if an object type depends on another object type.
    """

    @classmethod
    def relation(cls):
        return Relation.dependsOn

    @classmethod
    def _infer(cls, dependent: Type[TrackedObjectMixin],
                 dependency: Type[TrackedObjectMixin], recursive: bool = False) -> bool:
        raise NotImplementedError("Should be overridden in rdr meta")

    @classmethod
    def _should_infer(cls) -> bool:
        return not cls.inferred_once


dependsOn = DependsOn()
