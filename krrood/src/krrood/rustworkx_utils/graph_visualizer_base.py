from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import rustworkx as rx
from typing_extensions import (
    Any,
    Callable,
    ClassVar,
    List,
    Optional,
    Tuple,
    Union,
    Dict,
)

NodePosition = Union[Tuple[float, float], Tuple[float, float, float]]
"""An explicit 2D or 3D node position, in whatever coordinate space the caller intends (for
example a world frame, so nodes end up placed at bodies' real spatial positions)."""


class GraphLayout(Enum):
    """The algorithm used to place nodes in the drawing."""

    SPRING = auto()
    """Force-directed placement that runs once and settles, working for any graph."""

    LAYERED = auto()
    """Top-to-bottom placement by distance from the source nodes, suited to trees and DAGs."""

    PHYSICS = auto()
    """Continuous force-directed placement where nodes keep simulating, self-organize and bounce."""

    FIXED = auto()
    """Nodes are placed at the explicit position :attr:`~GraphVisualizerBase.position_getter`
    returns for each one, instead of being simulated."""


class GraphVisualizerBackend(Enum):
    """The rendering technology used to display a live graph."""

    PLOTLY = auto()
    """A Dash application drawing the graph with Plotly."""

    CYTOSCAPE = auto()
    """A Flask application drawing the graph with Cytoscape.js."""

    VIS_NETWORK = auto()
    """A Flask application drawing the graph with vis-network."""

    THREE = auto()
    """A Flask application drawing the graph in 3D with 3d-force-graph (Three.js)."""


def label_by_string(payload: Any) -> str:
    """Label a node by the string representation of its payload."""
    return str(payload)


def describe_by_repr(payload: Any) -> List[str]:
    """Describe a node by the repr of its payload as a single detail line."""
    return [repr(payload)]


@dataclass
class GraphLayoutOptions:
    """
    Additional options that each visualization library needs to realise a specific layout.
    """

    layout_options: Dict[GraphLayout, Dict[str, Any]]
    """
    Additional options for each layout that are needed by the specific graph library.
    """

    def get_options(self, layout: GraphLayout) -> Dict[str, Any]:
        return self.layout_options.get(layout, {})

    def add_new_options(self, layout: GraphLayout, options: Dict[str, Any]) -> None:
        self.layout_options[layout] = options


@dataclass
class GraphVisualizerBase(ABC):
    """Common interface and node access for the live rustworkx graph visualizers.

    The graph is read on demand, so in-place mutations of the graph (nodes and edges added while
    it is being built) appear on the next refresh without restarting the application. Each concrete
    subclass renders the graph with a different backend but shares the same labelling, colouring and
    detail behaviour.
    """

    required_modules: ClassVar[Tuple[str, ...]] = ()
    """The importable modules the backend needs to run."""

    graph: Union[rx.PyDiGraph, rx.PyGraph]
    """The live graph to display; re-read whenever the drawing is refreshed."""

    information_getter: Callable[[Any], List[str]] = describe_by_repr
    """Maps a node payload to the detail lines shown when the node is clicked."""

    label_getter: Callable[[Any], str] = label_by_string
    """Maps a node payload to the label drawn on the node."""

    color_getter: Optional[Callable[[Any], str]] = None
    """Maps a node payload to its color; a constant color is used when not provided."""

    default_node_color: str = "#79a6d2"
    """The marker color used when no ``color_getter`` is provided."""

    border_color_getter: Optional[Callable[[Any], str]] = None
    """Maps a node payload to its border color; derived automatically from its fill color when not provided."""

    default_border_color: str = "#888888"
    """The border color used for nodes and edges when no border color is given."""

    position_getter: Optional[Callable[[Any], NodePosition]] = None
    """Maps a node payload to an explicit position; used when :attr:`layout` is
    :attr:`GraphLayout.FIXED`, ignored otherwise."""

    layout: GraphLayout = GraphLayout.SPRING
    """The algorithm used to place the nodes."""

    title: str = "Rustworkx Graph"
    """The title shown above the graph."""

    refresh_interval_seconds: float = 1.0
    """The delay between refreshes that pick up graph changes."""

    port: int = 8050
    """The port the server listens on."""

    open_browser: bool = True
    """Whether to open the application in a web browser when started."""

    def node_label(self, node_index: int) -> str:
        """
        :param node_index: The rustworkx index of the node.
        :return: The label drawn on the node.
        """
        return self.label_getter(self.graph[node_index])

    def node_color(self, node_index: int) -> str:
        """
        :param node_index: The rustworkx index of the node.
        :return: The color of the node.
        """
        if self.color_getter is None:
            return self.default_node_color
        return self.color_getter(self.graph[node_index])

    def node_border_color(self, node_index: int) -> Optional[str]:
        """
        :param node_index: The rustworkx index of the node.
        :return: The border color of the node, or ``None`` to derive one automatically from its fill color.
        """
        if self.border_color_getter is None:
            return None
        return self.border_color_getter(self.graph[node_index])

    def node_position(self, node_index: int) -> Optional[NodePosition]:
        """
        :param node_index: The rustworkx index of the node.
        :return: The node's explicit position, or ``None`` if no :attr:`position_getter` is set.
        """
        if self.position_getter is None:
            return None
        return self.position_getter(self.graph[node_index])

    def node_details(self, node_index: int) -> List[str]:
        """
        :param node_index: The rustworkx index of the clicked node.
        :return: The detail lines describing the node's payload.
        """
        return self.information_getter(self.graph[node_index])

    @classmethod
    def check_dependencies(cls) -> None:
        """
        Ensure the optional libraries the backend needs are installed.

        :raises ModuleNotFoundError: If a required module is missing.
        """
        for module_name in cls.required_modules:
            if importlib.util.find_spec(module_name) is None:
                raise ModuleNotFoundError(
                    f"{module_name} must be installed to visualize the graph. "
                    "(pip install krrood[visualization])",
                    name=module_name,
                )

    @abstractmethod
    def run(self) -> None:
        """Start the visualization and optionally open it in a browser."""
