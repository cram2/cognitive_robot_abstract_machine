from __future__ import annotations

import inspect

from krrood.adapters.json_serializer import recursive_subclasses
from typing_extensions import Dict, List, Tuple, Type

from probabilistic_model.probabilistic_circuit.tensorized.inner_layer import (
    Layer,
    LayerConverter,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import Unit


def layer_class_of(clazz: Type) -> Type[Layer]:
    """
    Find the layer class that corresponds to a class of the rustworkx implementation.

    An exact match wins over an inherited one. That distinction matters because the
    distributions form their own hierarchy: a truncated Gaussian is a Gaussian, so
    matching by ``issubclass`` alone would put it into whichever of the two layers the
    subclass iteration happens to reach first.

    :param clazz: The unit class or the distribution class of a leaf unit.
    :return: The matching layer class.
    """
    candidates = [
        subclass
        for subclass in recursive_subclasses(Layer)
        if not inspect.isabstract(subclass)
    ]

    for subclass in candidates:
        if clazz in subclass.get_generic_type_parameters():
            return subclass

    for subclass in candidates:
        bound_types = tuple(subclass.get_generic_type_parameters())
        if bound_types and issubclass(clazz, bound_types):
            return subclass

    raise TypeError(f"Could not find a layer class for {clazz}")


def _type_of_node(node: Unit) -> Type:
    """
    :param node: A unit of a rustworkx circuit.
    :return: The distribution class of a leaf unit, or the unit's own class otherwise.
    """
    return type(node.distribution) if node.is_leaf else type(node)


def create_layers_from_nodes(
    nodes: List[Unit],
    child_layers: List[LayerConverter],
    progress_bar: bool = False,
) -> List[LayerConverter]:
    """
    Group a list of units of a rustworkx circuit into layers.

    :param nodes: The units that form one level of the rustworkx circuit.
    :param child_layers: The converters of the level below.
    :param progress_bar: Whether to show a progress bar.
    :return: One converter per created layer.
    """
    result = []

    # grouping is by exact type, not by ``isinstance``: a truncated Gaussian leaf is
    # an instance of the Gaussian distribution and would otherwise be pulled into the
    # Gaussian group, whose layer cannot hold it
    groups: Dict[Tuple[Type, Tuple], List[Unit]] = {}
    for node in nodes:
        groups.setdefault(
            (_type_of_node(node), tuple(node.variables)), []
        ).append(node)

    for (node_type, _), group in groups.items():
        layer_type = layer_class_of(node_type)
        result.append(
            layer_type.create_layer_from_nodes_with_same_type_and_scope(
                group, child_layers, progress_bar
            )
        )

    return result
