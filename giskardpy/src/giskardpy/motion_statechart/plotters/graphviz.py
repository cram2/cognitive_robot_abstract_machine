from __future__ import annotations

import re
from dataclasses import dataclass, field

import pydot
from typing_extensions import (
    List,
    Dict,
    Optional,
    Union,
    Set,
    TYPE_CHECKING,
)

from giskardpy.motion_statechart.graph_node import (
    MotionStatechartNode,
    TerminalNode,
)
from giskardpy.motion_statechart.graph_node import (
    Goal,
    TrinaryCondition,
)
from giskardpy.motion_statechart.plotters.plot_specs import (
    TRANSITION_SPECS,
    EdgeSpec,
    StateSelector,
)
from giskardpy.motion_statechart.plotters.styles import (
    RankSep,
    NodeSep,
    ObservationStateToColor,
    ObservationStateToSymbol,
    LiftCycleStateToColor,
    LiftCycleStateToSymbol,
    LineWidth,
    ConditionFont,
    FONT,
    Fontsize,
    GoalClusterStyle,
    ObservationStateToEdgeStyle,
    ArrowSize,
)

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart


def extract_node_names_from_condition(condition: str) -> Set[str]:
    """
    Collects the node names a condition expression refers to.

    :param condition: The condition expression to scan.
    :return: The names appearing in quotes inside the expression.
    """
    matches = re.findall(r'"(.*?)"|\'(.*?)\'', condition)
    return set(match for group in matches for match in group if match)


def format_condition(condition: str) -> str:
    """
    Rewrites a condition expression for display in an HTML label.

    Logical operators start a new line and trinary constants are spelled out.

    :param condition: The condition expression to rewrite.
    :return: The expression with graphviz line breaks and readable constants.
    """
    condition = condition.replace(" and ", "<BR/>       and ")
    condition = condition.replace(" or ", "<BR/>       or ")
    condition = condition.replace("1.0", "True")
    condition = condition.replace("0.0", "False")
    return condition


@dataclass
class MotionStatechartGraphviz:
    """
    Draws a motion statechart as a graphviz graph.

    Every node becomes a labelled box showing its current observation and life cycle
    state, every :class:`~giskardpy.motion_statechart.graph_node.Goal` becomes a cluster
    around its children, and every transition becomes a colored edge.

    ..note:: The drawing reflects the state the statechart is in when it is drawn.
    """

    motion_statechart: MotionStatechart
    """
    The statechart to draw, including the state it is currently in.
    """

    graph: pydot.Graph = field(init=False)
    """
    The graph the statechart is drawn into.
    """

    compact: bool = False
    """
    Whether nodes are drawn without their conditions and with tighter spacing.
    """

    _cluster_map: Dict[MotionStatechartNode, pydot.Cluster] = field(
        init=False, default_factory=dict
    )
    """
    Maps a goal to the cluster its children are drawn in, with ``None`` mapping to the
    top level graph.
    """

    def __post_init__(self):
        """
        Creates the empty graph the statechart is drawn into.
        """
        self.graph = pydot.Dot(
            graph_type="digraph",
            graph_name="",
            ranksep=RankSep if not self.compact else RankSep * 0.5,
            nodesep=NodeSep if not self.compact else NodeSep * 0.5,
            compound=True,
            ratio="compress",
        )

    def _format_motion_graph_node(
        self,
        node: MotionStatechartNode,
    ) -> str:
        """
        :param node: The node to label.
        :return: The HTML label showing the node's name, its observation and life cycle
            state and, outside of compact mode, its conditions.
        """
        obs_state = self.motion_statechart.observation_state[node]
        life_cycle_state = self.motion_statechart.life_cycle_state[node]
        obs_color = ObservationStateToColor[obs_state]
        obs_text = ObservationStateToSymbol[obs_state]
        life_color = LiftCycleStateToColor[life_cycle_state]
        life_symbol = LiftCycleStateToSymbol[life_cycle_state]
        label = (
            f'<<TABLE  BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            f"<TR>"
            f'  <TD WIDTH="100%" HEIGHT="{LineWidth}"></TD>'
            f"</TR>"
            f"<TR>"
            f"  <TD><B> {node.unique_name} </B></TD>"
            f"</TR>"
            f"<TR>"
            f'  <TD CELLPADDING="0">'
            f'    <TABLE BORDER="0" CELLBORDER="2" CELLSPACING="0" WIDTH="100%">'
            f"      <TR>"
            f'        <TD BGCOLOR="{obs_color}" WIDTH="50%" FIXEDSIZE="FALSE"><FONT FACE="monospace">{obs_text}</FONT></TD>'
            f"        <VR/>"
            f'        <TD BGCOLOR="{life_color}" WIDTH="50%" FIXEDSIZE="FALSE"><FONT FACE="monospace">{life_symbol}</FONT></TD>'
            f"      </TR>"
            f"    </TABLE>"
            f"  </TD>"
            f"</TR>"
        )
        if node.plot_specifications.collapse_children:
            label += self._build_hidden_node_count_block(node)
        if self.compact:
            label += (
                f"<TR>" f'  <TD WIDTH="100%" HEIGHT="{LineWidth*2.5}"></TD>' f"</TR>"
            )
        else:
            label += self._build_condition_block(node)
        label += f"</TABLE>>"
        return label

    def _build_hidden_node_count_block(self, node: MotionStatechartNode) -> str:
        """
        :param node: The node whose descendants are left out of the drawing.
        :return: The label row stating how many of them are hidden.
        """
        hidden_node_count = self._count_descendants(node)
        plural = "s" if hidden_node_count != 1 else ""
        return (
            f"<TR>"
            f'  <TD><FONT FACE="{ConditionFont}">'
            f"[+] {hidden_node_count} node{plural} hidden"
            f"</FONT></TD>"
            f"</TR>"
        )

    def _count_descendants(self, node: MotionStatechartNode) -> int:
        """
        :param node: The node to count below.
        :return: The number of nodes below it, nested goals included.
        """
        if not isinstance(node, Goal):
            return 0
        return sum(1 + self._count_descendants(child_node) for child_node in node.nodes)

    def _build_condition_block(
        self, node: MotionStatechartNode, line_color="black"
    ) -> str:
        """
        Builds the label rows listing the transition conditions of a node.

        Nodes that terminate the statechart only get their start condition, because the
        remaining conditions never fire for them.

        :param node: The node whose conditions are listed.
        :param line_color: The color of the lines separating the rows.
        :return: The condition rows of the label.
        """
        start_condition = format_condition(str(node._start_condition))
        pause_condition = format_condition(str(node._pause_condition))
        end_condition = format_condition(str(node._end_condition))
        reset_condition = format_condition(str(node._reset_condition))
        label = (
            f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
            f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">start:{start_condition}</FONT></TD></TR>'
        )
        if not isinstance(node, TerminalNode):
            label += (
                f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
                f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">pause:{pause_condition}</FONT></TD></TR>'
            )
            label += (
                f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
                f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">end  :{end_condition}</FONT></TD></TR>'
            )
            label += (
                f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
                f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">reset:{reset_condition}</FONT></TD></TR>'
            )
        return label

    def _escape_name(self, name: str) -> str:
        """
        :param name: The node name to escape.
        :return: The name in the quoted form pydot stores it under.
        """
        return f'"{name}"'

    def _get_cluster_of_node(
        self, node_name: str, graph: Union[pydot.Graph, pydot.Cluster]
    ) -> Optional[pydot.Cluster]:
        """
        :param node_name: The name of the node to look for.
        :param graph: The graph whose direct subgraphs are searched.
        :return: The subgraph holding the node, or ``None`` if none of them does.
        """
        node_cluster = None
        for cluster in graph.get_subgraphs():
            if (
                len(cluster.get_node(self._escape_name(node_name))) == 1
                or len(cluster.get_node(node_name)) == 1
            ):
                node_cluster = cluster
                break
        return node_cluster

    def _add_node(
        self,
        graph: pydot.Graph,
        node: MotionStatechartNode,
    ) -> pydot.Node:
        """
        Adds a node to a graph, wrapping it into one nested cluster per extra border
        style its plot specification asks for.

        :param graph: The graph the node is added to.
        :param node: The node to draw.
        :return: The added node.
        """
        pydot_node = self._create_pydot_node(node)
        if len(node.plot_specifications.extra_border_styles) == 0:
            graph.add_node(pydot_node)
            return pydot_node
        child = pydot_node
        for index, style in enumerate(node.plot_specifications.extra_border_styles):
            c = pydot.Cluster(
                graph_name=f"{node.unique_name}",
                penwidth=LineWidth,
                style=node.plot_specifications.extra_border_styles[index],
                color="black",
            )
            if index == 0:
                c.add_node(child)
            else:
                c.add_subgraph(child)
            child = c
        if len(node.plot_specifications.extra_border_styles) > 0:
            graph.add_subgraph(c)
        return pydot_node

    def _create_pydot_node(self, node: MotionStatechartNode) -> pydot.Node:
        """
        :param node: The node to draw.
        :return: A labelled pydot node shaped and styled by the node's plot
            specification.
        """
        label = self._format_motion_graph_node(node=node)
        pydot_node = pydot.Node(
            str(node.unique_name),
            label=label,
            shape=node.plot_specifications.shape,
            color="black",
            style=node.plot_specifications.style,
            margin=0,
            fillcolor="white",
            fontname=FONT,
            fontsize=Fontsize,
            penwidth=LineWidth,
        )
        return pydot_node

    def to_dot_graph(self) -> pydot.Graph:
        """
        Draws every visible node and transition of the statechart.

        :return: The drawn graph.
        """
        self._cluster_map[None] = self.graph
        top_level_nodes = [
            node for node in self.motion_statechart.nodes if not node.parent_node
        ]
        self._add_nodes(self.graph, top_level_nodes)
        self._add_edges()
        return self.graph

    def to_dot_graph_pdf(self, file_name: str):
        """
        Draws the statechart and writes it to a pdf.

        :param file_name: The path of the pdf to write.
        """
        self.to_dot_graph()
        file_name = file_name
        # create_path(file_name)
        self.graph.write_pdf(file_name)
        print(f"Saved task graph at {file_name}.")

    def _is_drawn(self, node: MotionStatechartNode) -> bool:
        """
        :param node: The node to check.
        :return: Whether the node appears in the drawing, which it does not if it or one
            of its ancestors is invisible, or if one of its ancestors collapses its
            children.
        """
        if not node.plot_specifications.visible:
            return False
        current = node.parent_node
        while current is not None:
            if (
                not current.plot_specifications.visible
                or current.plot_specifications.collapse_children
            ):
                return False
            current = current.parent_node
        return True

    def _add_nodes(
        self,
        parent_cluster: Union[pydot.Graph, pydot.Cluster],
        nodes: List[MotionStatechartNode],
    ):
        """
        Draws the given nodes, recursing into the children of every goal that does not
        collapse them.

        :param parent_cluster: The graph or cluster the nodes are drawn in.
        :param nodes: The nodes to draw.
        """
        for i, node in enumerate(nodes):
            # Skip invisible nodes entirely, as well as the children of a Goal that is
            # invisible or collapses them.
            if not self._is_drawn(node):
                continue

            if (
                isinstance(node, Goal)
                and not node.plot_specifications.collapse_children
            ):
                goal_cluster = self._add_cluster(node, parent_cluster)
                self._add_node(
                    graph=goal_cluster,
                    node=node,
                )
                self._add_nodes(goal_cluster, node.nodes)
                continue

            self._add_node(
                parent_cluster,
                node=node,
            )

    def _add_cluster(
        self,
        node: MotionStatechartNode,
        parent_cluster: Union[pydot.Graph, pydot.Cluster],
    ):
        """
        Opens the cluster that a goal and its children are drawn in.

        :param node: The goal to draw a border around.
        :param parent_cluster: The graph or cluster the new cluster is nested in.
        :return: The new cluster.
        """
        goal_cluster = pydot.Cluster(
            graph_name=str(node.unique_name),
            fontname=FONT,
            fontsize=Fontsize,
            style=GoalClusterStyle,
            color="black",
            fillcolor="white",
            penwidth=LineWidth,
        )
        parent_cluster.add_subgraph(goal_cluster)
        self._cluster_map[node] = goal_cluster
        return goal_cluster

    def _add_edges(self):
        """
        Draws an edge for every transition whose endpoints are both drawn in the same
        cluster.

        :raises ValueError: If a transition has a kind that has no edge specification.
        """
        transition: TrinaryCondition
        for edge_index, (
            parent_node_index,
            child_node_index,
            transition,
        ) in self.motion_statechart.rx_graph.edge_index_map().items():
            parent_node = self.motion_statechart.rx_graph.get_node_data(
                parent_node_index
            )
            child_node = self.motion_statechart.rx_graph.get_node_data(child_node_index)

            # Skip edges if either endpoint (or one of its ancestors) is not drawn
            if not self._is_drawn(parent_node):
                continue
            if not self._is_drawn(child_node):
                continue

            if not self._are_nodes_in_same_cluster(parent_node, child_node):
                continue
            spec = TRANSITION_SPECS.get(transition.kind)
            if spec is None:
                raise ValueError(f"Unhandled transition kind: {transition.kind}")
            self._add_condition_edge(parent_node, child_node, spec)

    def _are_nodes_in_same_cluster(
        self, parent_node: MotionStatechartNode, child_node: MotionStatechartNode
    ) -> bool:
        """
        :param parent_node: The node the transition points to.
        :param child_node: The node the transition belongs to.
        :return: Whether both nodes are drawn in the same cluster.
        """
        parent_node_parent = parent_node.parent_node
        child_node_parent = child_node.parent_node

        if parent_node_parent is None or child_node_parent is None:
            return parent_node_parent is child_node_parent

        return parent_node_parent.name == child_node_parent.name

    def _edge_clusters_kwargs(
        self,
        graph: Union[pydot.Graph, pydot.Cluster],
        src_name: str,
        dst_name: str,
    ) -> Dict[str, object]:
        """
        Determines the edge attributes that clip an edge at a cluster border instead of
        letting it reach into the cluster.

        :param graph: The graph or cluster the edge is drawn in.
        :param src_name: The name of the node the edge starts at.
        :param dst_name: The name of the node the edge ends at.
        :return: The ``ltail`` and ``lhead`` attributes for the endpoints that sit in a
            cluster.
        """
        kwargs: Dict[str, object] = {}
        dst_cluster = self._get_cluster_of_node(dst_name, graph)
        src_cluster = self._get_cluster_of_node(src_name, graph)
        if dst_cluster is not None:
            kwargs["lhead"] = dst_cluster.get_name()
        if src_cluster is not None:
            kwargs["ltail"] = src_cluster.get_name()
        return kwargs

    def _add_condition_edge(
        self,
        parent_node: MotionStatechartNode,
        child_node: MotionStatechartNode,
        spec: EdgeSpec,
    ) -> None:
        """
        Draws the edge of a single transition.

        Its direction and color come from the edge specification, while its line style
        reflects the observation state of the node the specification points at.

        :param parent_node: The node the transition points to.
        :param child_node: The node the transition belongs to.
        :param spec: The edge specification of the transition kind.
        """
        graph = self._cluster_map[parent_node.parent_node]

        def _select_node(
            parent: MotionStatechartNode,
            child: MotionStatechartNode,
            selector: StateSelector,
        ) -> MotionStatechartNode:
            """
            :return: The node the selector names.
            """
            return parent if selector == "parent" else child

        src_node = _select_node(parent_node, child_node, spec.src_selector)
        dst_node = _select_node(parent_node, child_node, spec.dst_selector)
        style_node = _select_node(parent_node, child_node, spec.state_selector)

        src_name = str(src_node.unique_name)
        dst_name = str(dst_node.unique_name)

        kwargs = self._edge_clusters_kwargs(graph, src_name, dst_name)

        observation_state = self.motion_statechart.observation_state[style_node]
        kwargs.update(ObservationStateToEdgeStyle[observation_state])

        kwargs.update(spec.extras())

        graph.add_edge(
            pydot.Edge(
                src=src_name,
                dst=dst_name,
                color=spec.color,
                arrowsize=ArrowSize,
                **kwargs,
            )
        )
