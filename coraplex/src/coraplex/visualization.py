from __future__ import annotations

import logging
from typing_extensions import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    List,
)
from dataclasses import dataclass

import networkx as nx
from rustworkx import PyDiGraph

from coraplex.datastructures.enums import VisualizationLayout

if TYPE_CHECKING:
    from bokeh.document import Document
    from bokeh.models import GraphRenderer

logger = logging.getLogger(__name__)


@dataclass
class GraphVisualizer:
    """
    Handles the interactive visualization of a rustworkx graph using Bokeh.
    Supports dynamic updates by periodically checking for graph changes.
    """

    graph: Any
    """
    The rustworkx graph to visualize.
    """

    graph_source: Optional[Callable[[], Any]] = None
    """
    Optional callable returning the current graph, polled on every update tick.
    """

    node_params: Optional[Dict[int, Dict[str, Any]]] = None
    """
    Optional mapping from node index to the parameters shown when the node is clicked.
    """

    node_label: Optional[Callable[[int, Any], str]] = None
    """
    Optional callable mapping (index, payload) to the label shown for a node.
    """

    # attributes: Optional[Sequence[str]] = None # kroodd has problems with this
    attributes: List[str] = None
    """
    Optional attribute names to display; ``None`` shows all parameters.
    """

    layout: VisualizationLayout = VisualizationLayout.BFS
    """
    Layout algorithm to use: "spring", "kamada_kawai", or "bfs".
    """

    start: Optional[int] = None
    """
    Optional start node index for the "bfs" layout.
    """

    title: str = "Rustworkx Graph"
    """
    Title of the plot.
    """

    width: int = 1200
    """
    Figure width in pixels.
    """

    height: int = 800
    """
    Figure height in pixels.
    """

    update_interval: int = 1000
    """
    Interval in milliseconds between graph-change checks.
    """

    def _build_bokeh_app(self, doc: Document) -> None:
        # Local imports to keep dependency optional
        from bokeh.layouts import row
        from bokeh.models import (
            HoverTool,
            NodesAndLinkedEdges,
            TapTool,
            CustomJS,
            Div,
        )
        from bokeh.plotting import figure, from_networkx

        nx_graph = self._build_current_nx_graph()
        positions = calculate_layout_positions(self.layout.value, nx_graph, self.start)

        plot = figure(
            title=self.title,
            x_axis_location=None,
            y_axis_location=None,
            width=self.width,
            height=self.height,
            toolbar_location="below",
            background_fill_color="#efefef",
        )
        plot.grid.grid_line_color = None

        renderer = from_networkx(nx_graph, positions)
        renderer.node_renderer.glyph.update(size=18, fill_color="#79a6d2")

        hover = HoverTool(tooltips=[("label", "@label")])
        plot.add_tools(hover, TapTool())
        renderer.selection_policy = NodesAndLinkedEdges()
        renderer.inspection_policy = NodesAndLinkedEdges()

        info_panel = Div(
            text="<b>Click a node to see its parameters</b>",
            width=400,
            height=self.height,
        )

        node_source = renderer.node_renderer.data_source

        callback = CustomJS(
            args=dict(source=node_source, panel=info_panel),
            code="""
                const indices = source.selected.indices;
                if (indices.length === 0) {
                    panel.text = "<b>Click a node to see its parameters</b>";
                    return;
                }
                const index = indices[0];
                const label = source.data['label'][index];
                const parameters = source.data['param_text'][index] || '';
                panel.text = `<div><h3 style=\"margin:0 0 8px 0;\">${label}</h3>${parameters}</div>`;
            """,
        )
        node_source.selected.js_on_change("indices", callback)

        plot.renderers.append(renderer)

        last_snapshot = _graph_snapshot(nx_graph)

        def update_callback() -> None:
            nonlocal last_snapshot
            try:
                new_nx_graph = self._build_current_nx_graph()
                snapshot = _graph_snapshot(new_nx_graph)
                if snapshot != last_snapshot:
                    last_snapshot = snapshot
                    self._update_plot(renderer, new_nx_graph)
            except Exception:
                logger.exception("Failed to update the graph visualization")

        doc.add_periodic_callback(update_callback, self.update_interval)
        doc.add_root(row(plot, info_panel))

    def _build_current_nx_graph(self) -> nx.Graph:
        """
        Build the networkx graph to display, fetching a fresh graph from
        ``graph_source`` if one was given.
        """
        if self.graph_source is not None:
            self.graph = self.graph_source()
        return build_nx_graph(
            self.graph, self.node_params, self.attributes, self.node_label
        )

    def _update_plot(self, renderer: GraphRenderer, nx_graph: nx.Graph) -> None:
        from bokeh.plotting import from_networkx

        positions = calculate_layout_positions(self.layout, nx_graph, self.start)
        new_renderer = from_networkx(nx_graph, positions)

        renderer.node_renderer.data_source.data = dict(
            new_renderer.node_renderer.data_source.data
        )
        renderer.edge_renderer.data_source.data = dict(
            new_renderer.edge_renderer.data_source.data
        )
        renderer.layout_provider.graph_layout = dict(
            new_renderer.layout_provider.graph_layout
        )

    def show(self) -> None:
        """
        Launch the Bokeh server in a background thread and open the plot in
        the browser.

        The server runs on a daemon thread, so the calling process must stay
        alive for the visualization to remain reachable.
        """
        import asyncio
        import threading

        def run_server() -> None:
            from bokeh.server.server import Server

            # The server and its IOLoop must be created on the thread that
            # runs them, with an event loop bound to that thread.
            asyncio.set_event_loop(asyncio.new_event_loop())
            server = Server({"/": self._build_bokeh_app}, port=0)
            server.start()
            server.io_loop.add_callback(server.show, "/")
            server.io_loop.start()

        threading.Thread(
            target=run_server, daemon=True, name="GraphVisualizerServer"
        ).start()


def create_ordered_graph(plan) -> Tuple[PyDiGraph, Dict[int, int]]:
    """
    Build a new graph containing the plan's nodes in depth-first order.

    The node indices are remapped, so they differ from the indices in
    ``plan.plan_graph``.

    :param plan: The plan to build the graph from.
    :return: The new graph and a mapping from plan node indices to the
        indices in the new graph.
    """
    ordered_graph = PyDiGraph(multigraph=False)
    mapping = {}

    for node in plan.nodes:
        mapping[node.index] = ordered_graph.add_node(node)
    for node in plan.nodes:
        for child in node.children:
            ordered_graph.add_edge(mapping[node.index], mapping[child.index], None)
    return ordered_graph, mapping


def plot_rustworkx_interactive(
    graph: Any,
    *,
    graph_source: Optional[Callable[[], Any]] = None,
    node_params: Optional[Dict[int, Dict[str, Any]]] = None,
    node_label: Optional[Callable[[int, Any], str]] = None,
    attributes: Optional[Sequence[str]] = None,
    layout: VisualizationLayout = VisualizationLayout.BFS,
    start: Optional[int] = None,
    title: str = "Rustworkx Graph",
    width: int = 1200,
    height: int = 800,
):
    """
    Plot an interactive visualization of a rustworkx graph.

    - Click on a node to show its parameters in a side panel.
    - Hover shows the node label.
    - The plot is dynamically updated when the given graph is changing.

    Parameters
    ----------
    graph:
        A rustworkx.PyGraph or rustworkx.PyDiGraph instance.
    graph_source:
        Optional callable returning the current graph to display. If given,
        it is invoked on every update tick so that changes to the underlying
        data are picked up even when ``graph`` is a one-time snapshot.
    node_params:
        Optional mapping from node index to a dict of parameters to display when
        the node is clicked. If not provided and the node payload is a dict,
        those items will be used. If provided together with ``attributes``, the
        displayed parameters will be filtered to the given attribute names.
    node_label:
        Optional callable that takes (index, payload) and returns a label string
        for the node. By default it tries to use ``payload.get('label')`` or
        ``str(payload)``.
    attributes:
        Optional list of attribute names to show from the parameters. Ignored if
        parameters are not dict-like.
    layout:
        Layout algorithm to use: "spring", "kamada_kawai", or "bfs".
    start:
        Optional start node index for "bfs" layout.
    title:
        Plot title.
    width, height:
        Figure size in pixels.

    Notes
    -----
    This function imports bokeh lazily so that it does not add a hard runtime
    dependency unless you call it. Install with `pip install bokeh`.
    """
    try:
        import bokeh  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "plot_rustworkx_interactive requires bokeh. Install with 'pip install bokeh'."
        ) from exc

    visualizer = GraphVisualizer(
        graph=graph,
        graph_source=graph_source,
        node_params=node_params,
        node_label=node_label,
        attributes=attributes,
        layout=layout,
        start=start,
        title=title,
        width=width,
        height=height,
    )
    visualizer.show()


def calculate_layout_positions(
    layout: str, nx_g: nx.Graph, start: Optional[int] = None
) -> Dict[int, Tuple[float, float]]:
    """
    Calculates node positions based on the selected layout.
    :param layout: Layout name, e.g. "spring", "kamada_kawai", "bfs"
    :param nx_g: networkx graph
    :param start: Optional start node index for "bfs" layout.
    :return: A dictionary mapping node indices to 2d coordinates.
    """
    if len(nx_g) == 0:
        return {}
    if layout == "spring":
        pos = nx.spring_layout(nx_g, seed=42)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(nx_g)
    elif layout == "bfs":
        if start is None or start not in nx_g:
            start = next(iter(nx_g.nodes))
        try:
            pos = nx.bfs_layout(nx_g, start=start)
        except nx.NetworkXError:
            pos = nx.spring_layout(nx_g, seed=42)
    else:
        pos = nx.spring_layout(nx_g, seed=42)
    return pos


def build_nx_graph(graph: PyDiGraph, node_params, attributes, node_label) -> nx.Graph:
    """Convert a rustworkx graph to a networkx graph."""
    nx_g = nx.DiGraph() if isinstance(graph, PyDiGraph) else nx.Graph()

    attributes = list(attributes) if attributes is not None else None

    # Iterate node_indices() instead of range(num_nodes()): rustworkx indices
    # are not contiguous after node removals.
    for i in graph.node_indices():
        payload = graph[i]
        # Label
        if node_label is not None:
            label = node_label(i, payload)
        else:
            label = None
            if isinstance(payload, dict) and "label" in payload:
                label = str(payload.get("label"))
            if label is None:
                label = str(payload)
        # Parameters
        if node_params is not None:
            params = node_params.get(i)
        else:
            params = _object_params_with_properties(payload)
        # Filter attributes if requested
        if attributes is not None and isinstance(params, dict):
            params = {k: params.get(k) for k in attributes if k in params}
        # Attach as node attributes
        nx_g.add_node(
            i,
            label=label,
            param_text=_format_params(params),
        )

    for u, v in graph.edge_list():
        nx_g.add_edge(u, v)

    return nx_g


def _graph_snapshot(nx_graph: nx.Graph) -> Tuple[Any, Any]:
    """
    Return a hashable summary of the rendered graph used for change detection.

    Comparing the rendered labels and parameter texts detects in-place
    mutations of node payloads, which comparing the payload objects
    themselves cannot.
    """
    nodes = tuple(
        sorted(
            (i, data.get("label", ""), data.get("param_text", ""))
            for i, data in nx_graph.nodes(data=True)
        )
    )
    edges = tuple(sorted(nx_graph.edges()))
    return nodes, edges


def _object_params_with_properties(payload: Any) -> Optional[Dict[str, Any]]:
    """
    Build a parameter dictionary from a node payload by combining:
    - public attributes from payload.__dict__ (if present)
    - readable @property attributes defined on the payload's class
    - if payload is a dict, return it (excluding 'label')

    Private attributes (starting with '_') and the key 'label' are excluded.
    Values that raise on access are skipped. Callables are skipped.
    Explicit attributes take precedence over properties of the same name.
    """
    # If the payload is already a dict, filter and return it.
    if isinstance(payload, dict):
        return {k: v for k, v in payload.items() if k != "label"}

    if payload is None:
        return None

    params: Dict[str, Any] = {}

    # Collect from __dict__ if available
    try:
        if isinstance(getattr(payload, "__dict__", None), dict):
            for k, v in vars(payload).items():
                if k.startswith("_") or k == "label":
                    continue
                # Avoid adding callables
                if not callable(v):
                    params[k] = v
    except Exception:
        pass

    for name, value in _collect_properties(payload).items():
        params.setdefault(name, value)

    return params if params else None


def _collect_properties(payload) -> Dict[str, Any]:
    params = {}
    # Collect readable @property attributes on the class
    try:
        import inspect

        cls = type(payload)
        for name, member in inspect.getmembers(cls):
            if not isinstance(member, property):
                continue
            if name.startswith("_") or name == "label":
                continue
            # Access property value safely
            try:
                value = getattr(payload, name)
            except Exception:
                continue
            # Skip callables
            if callable(value):
                continue
            params[name] = value
    except Exception:
        # If inspection fails, just ignore properties
        pass
    return params


def _format_params(params: Optional[Dict[str, Any]]) -> str:
    """Return HTML for parameter dict suitable for the side panel."""
    if not params:
        return "<i>No parameters</i>"
    try:
        items = []
        for k, v in params.items():
            items.append(
                f"<tr><td style='padding-right:8px; white-space:nowrap;'><b>{_escape_html(k)}</b></td><td>{_escape_html(v)}</td></tr>"
            )
        return "<table>" + "".join(items) + "</table>"
    except Exception:
        return f"<pre>{_escape_html(params)}</pre>"


def _escape_html(value: Any) -> str:
    try:
        s = str(value)
    except Exception:
        s = repr(value)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
