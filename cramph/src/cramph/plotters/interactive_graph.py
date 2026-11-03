from __future__ import annotations

from dataclasses import dataclass

import rustworkx as rx
from typing_extensions import TYPE_CHECKING

from cramph.node import StatechartNode
from krrood.ormatic.utils import classproperty
from krrood.rustworkx_utils.graph_visualizer_base import (
    GraphLayout,
    GraphVisualizerBackend,
    GraphVisualizerBase,
)
from krrood.rustworkx_utils.visualization.cytoscape_graph_visualizer import (
    CytoscapeGraphVisualizer,
)
from krrood.rustworkx_utils.visualization.interactive_graph_visualizer import (
    InteractiveGraphVisualizer,
)
from krrood.rustworkx_utils.visualization.three_graph_visualizer import (
    ThreeGraphVisualizer,
)
from krrood.rustworkx_utils.visualization.visnetwork_graph_visualizer import (
    VisNetworkGraphVisualizer,
)

if TYPE_CHECKING:
    from cramph.statechart import Statechart


# %% interactive statechart graph


@dataclass
class StatechartGraphVisualizer:
    """
    An interactive drawing of the nodes of a statechart and their children.

    Nodes are labelled by their unique name, coloured by their life cycle state and
    reveal their state and run ticks when clicked, all updated while the statechart
    ticks.

    .. note:: The drawn nodes are those the statechart holds when the visualizer is
        created; create another one to see nodes added afterwards.
    """

    statechart: Statechart
    """
    The statechart to draw.
    """

    @classproperty
    def visualizer_classes(
        cls,
    ) -> dict[GraphVisualizerBackend, type[GraphVisualizerBase]]:
        """
        :return: The visualizer to use for each rendering backend.
        """
        return {
            GraphVisualizerBackend.PLOTLY: InteractiveGraphVisualizer,
            GraphVisualizerBackend.CYTOSCAPE: CytoscapeGraphVisualizer,
            GraphVisualizerBackend.VIS_NETWORK: VisNetworkGraphVisualizer,
            GraphVisualizerBackend.THREE: ThreeGraphVisualizer,
        }

    def create_visualizer(
        self, backend: GraphVisualizerBackend, layout: GraphLayout
    ) -> GraphVisualizerBase:
        """
        :param backend: The rendering technology to use.
        :param layout: The algorithm used to place the nodes.
        :return: A visualizer of the statechart, before it is started.
        """
        return self.visualizer_classes[backend](
            graph=self._parent_child_graph(),
            label_getter=lambda node: node.unique_name,
            information_getter=self._node_details,
            color_getter=lambda node: node.life_cycle_state.color.to_hex(),
            layout=layout,
            title=f"Statechart with {len(self.statechart.nodes)} nodes",
        )

    def _parent_child_graph(self) -> rx.PyDiGraph[StatechartNode]:
        """
        Unlike :attr:`~cramph.statechart.Statechart.rx_graph`, whose edges are
        transition condition dependencies, this graph's edges lead from each node to its
        children.

        :return: A graph of every node, each at its own :attr:`~StatechartNode.index`.
        """
        graph = rx.PyDiGraph()
        graph.add_nodes_from(self.statechart.nodes)
        graph.add_edges_from_no_data(
            [
                (node.index, child.index)
                for node in self.statechart.nodes
                for child in node.children
            ]
        )
        return graph

    def _node_details(self, node: StatechartNode) -> list[str]:
        """
        :param node: The node to describe.
        :return: The state of the node and the ticks of its current run as detail lines.
        """
        details = [
            f"life cycle: {node.life_cycle_state.name}",
            f"observation: {self.statechart.observation_state[node].name}",
        ]
        run = self.statechart.state_history.get_current_run_ticks_of_node(node)
        if run is None:
            return details
        return details + [f"start tick: {run.start_tick}", f"end tick: {run.end_tick}"]
