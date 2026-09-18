from __future__ import annotations

import inspect

import tqdm
from krrood.adapters.json_serializer import recursive_subclasses
from sortedcontainers import SortedSet
from typing_extensions import Dict, List, Tuple, Type

from probabilistic_model.probabilistic_circuit.tensorized.inner_layer import (
    Layer,
    LayerConverter,
    QueryCache,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    Unit,
)


def import_layer_modules() -> None:
    """
    Import every module that defines a concrete layer.

    :func:`layer_class_of` finds a layer among the subclasses of :class:`Layer`, which
    only exist once the module defining them has been imported. That import lives here,
    and not in the ``__init__`` of the package, because the layer modules import this
    one.
    """
    from probabilistic_model.probabilistic_circuit.tensorized import (  # noqa: F401
        discrete_layer,
        gaussian_layer,
        input_layer,
        uniform_layer,
    )
    from probabilistic_model.probabilistic_circuit.tensorized import (  # noqa: F401
        inner_layer,
    )


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
    import_layer_modules()

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
        groups.setdefault((_type_of_node(node), tuple(node.variables)), []).append(node)

    for (node_type, _), group in groups.items():
        layer_type = layer_class_of(node_type)
        result.append(
            layer_type.create_layer_from_nodes_with_same_type_and_scope(
                group, child_layers, progress_bar
            )
        )

    return result


def root_layer_of_circuit(
    circuit: ProbabilisticCircuit, progress_bar: bool = False
) -> Layer:
    """
    Convert a circuit of the ``rx`` package into layers.

    :param circuit: The circuit to convert.
    :param progress_bar: Whether to show a progress bar.
    :return: The root layer of the converted circuit.
    :raises ValueError: If the circuit does not have exactly one root.
    """
    converters: List[LayerConverter] = []

    levels = list(circuit.layers)
    iterator = (
        tqdm.tqdm(reversed(levels), total=len(levels), desc="Creating layers")
        if progress_bar
        else reversed(levels)
    )

    for nodes in iterator:
        # every converter created so far is offered as a possible child, not only those
        # of the level directly below: the layering of the graph is by shortest distance
        # to the root, so an edge may skip levels
        converters = (
            create_layers_from_nodes(nodes, converters, progress_bar) + converters
        )

    root_converters = [
        converter for converter in converters if converter.nodes[0] is circuit.root
    ]
    if len(root_converters) != 1:
        raise ValueError("The circuit does not have exactly one root.")

    return root_converters[0].layer


def circuit_of_root_layer(
    root: Layer, variables: SortedSet, progress_bar: bool = False
) -> ProbabilisticCircuit:
    """
    Convert layers into a circuit of the ``rx`` package.

    :param root: The root layer to convert.
    :param variables: The variables of the circuit, in the order the layers index them.
    :param progress_bar: Whether to show a progress bar.
    :return: The converted circuit.
    """
    bar = (
        tqdm.tqdm(total=root.number_of_components, desc="Converting to rx")
        if progress_bar
        else None
    )
    result = ProbabilisticCircuit()
    root.to_rustworkx(variables, result, QueryCache(), bar)
    return result
