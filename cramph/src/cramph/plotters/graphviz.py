from __future__ import annotations

import re
from dataclasses import dataclass, field

import pydot
from typing_extensions import TYPE_CHECKING

from cramph.data_types import ObservationStateValues, TransitionKind
from cramph.node import StatechartNode, TerminalNode
from cramph.node import CompositeNode, TransitionCondition
from cramph.plotters.styles import (
    DrawingMetrics,
    Font,
    NodeDrawingStyle,
    ObservationDrawingStyle,
)
from semantic_digital_twin.world_description.geometry import Color

if TYPE_CHECKING:
    from cramph.statechart import Statechart


@dataclass(unsafe_hash=True)
class ConditionDependency:
    """
    One node reading another in its transition conditions.

    However many conditions read it, the drawing shows the pair once, because an arrow
    no longer says which condition it feeds.
    """

    condition_owner: StatechartNode
    """
    The node whose conditions read :attr:`observed_node`.
    """

    observed_node: StatechartNode
    """
    The node those conditions read.
    """

    conditions: list[TransitionCondition] = field(
        default_factory=list, compare=False, hash=False
    )
    """
    The conditions of :attr:`condition_owner` that read :attr:`observed_node`.
    """

    def minimum_rank_distance(self, metrics: DrawingMetrics) -> int:
        """
        A rank distance is a lower bound, so the separate arrows this one replaces
        already left graphviz solving for the largest of them.

        :param metrics: The measurements the drawing is laid out with.
        :return: How many rows below the observed node its owner is drawn at least.
        """
        return max(
            metrics.minimum_rank_distance(condition.kind)
            for condition in self.conditions
        )


def format_condition_text(text: str, color_constants: bool = False) -> str:
    """
    Rewrites the part of a condition that is not a term for display in an HTML label.

    Logical operators start a new line and constants are spelled out.

    :param text: The text to rewrite.
    :param color_constants: Whether boolean constants should be colored with their
        observation color.
    :return: The text with graphviz line breaks and readable constants.
    """
    text = text.replace(" and ", "<BR/>       and ")
    text = text.replace(" or ", "<BR/>       or ")
    text = text.replace("1.0", "True")
    text = text.replace("0.0", "False")
    if color_constants:
        true_color = ObservationDrawingStyle.TRUE.color.to_hex()
        false_color = ObservationDrawingStyle.FALSE.color.to_hex()
        text = re.sub(r"\bTrue\b", f'<FONT COLOR="{true_color}">True</FONT>', text)
        text = re.sub(r"\bFalse\b", f'<FONT COLOR="{false_color}">False</FONT>', text)
    return text


@dataclass
class StatechartGraphviz:
    """
    Draws a statechart as a graphviz graph.

    Every node becomes a labelled box showing its current observation and life cycle
    state, every :class:`~cramph.node.CompositeNode`
    becomes a cluster around its children, and every dependency between two nodes becomes an arrow colored
    by what the node it leaves observes.

    ..note:: The drawing reflects the state the statechart is in when it is drawn.
    """

    statechart: Statechart
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

    metrics: DrawingMetrics = field(default_factory=DrawingMetrics)
    """
    The measurements the drawing is laid out with.
    """

    disabled_condition_color: Color = field(
        default_factory=lambda: Color.from_hex("#9CA3AF")
    )
    """
    The color a condition that cannot become true any more is written in.
    """

    _cluster_map: dict[StatechartNode, pydot.Cluster] = field(
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
            ranksep=(
                self.metrics.rank_separation
                if not self.compact
                else self.metrics.rank_separation
                * self.metrics.compact_separation_factor
            ),
            nodesep=(
                self.metrics.node_separation
                if not self.compact
                else self.metrics.node_separation
                * self.metrics.compact_separation_factor
            ),
            compound=True,
            ratio="compress",
        )

    def _format_graph_node(
        self,
        node: StatechartNode,
    ) -> str:
        """
        :param node: The node to label.
        :return: The HTML label showing the node's name, its observation and life cycle
            state and, outside of compact mode, its conditions.
        """
        obs_state = self.statechart.observation_state[node]
        life_cycle_state = self.statechart.life_cycle_state[node]
        obs_color = obs_state.color.to_hex()
        obs_badge = obs_state.badge
        life_color = life_cycle_state.color.to_hex()
        life_badge = life_cycle_state.badge
        label = (
            f'<<TABLE  BORDER="0" CELLBORDER="0" CELLSPACING="0">'
            f"<TR>"
            f'  <TD WIDTH="100%" HEIGHT="{self.metrics.line_width}"></TD>'
            f"</TR>"
            f"<TR>"
            f"  <TD><B> {node.unique_name} </B></TD>"
            f"</TR>"
            f"<TR>"
            f'  <TD CELLPADDING="0">'
            f'    <TABLE BORDER="0" CELLBORDER="2" CELLSPACING="0" WIDTH="100%">'
            f"      <TR>"
            f'        <TD BGCOLOR="{obs_color}" WIDTH="50%" FIXEDSIZE="FALSE"><FONT FACE="{Font.MONOSPACE}">{obs_badge}</FONT></TD>'
            f"        <VR/>"
            f'        <TD BGCOLOR="{life_color}" WIDTH="50%" FIXEDSIZE="FALSE"><FONT FACE="{Font.MONOSPACE}">{life_badge}</FONT></TD>'
            f"      </TR>"
            f"    </TABLE>"
            f"  </TD>"
            f"</TR>"
        )
        if node.plot_specifications.collapse_children:
            label += self._build_hidden_node_count_block(node)
        if self.compact:
            label += (
                f"<TR>"
                f'  <TD WIDTH="100%" HEIGHT="{self.metrics.line_width * self.metrics.compact_bottom_padding_factor}"></TD>'
                f"</TR>"
            )
        else:
            label += self._build_condition_block(node)
        label += f"</TABLE>>"
        return label

    def _build_hidden_node_count_block(self, node: StatechartNode) -> str:
        """
        :param node: The node whose descendants are left out of the drawing.
        :return: The label row stating how many of them are hidden.
        """
        hidden_node_count = len(node.descendants)
        plural = "s" if hidden_node_count != 1 else ""
        return (
            f"<TR>"
            f'  <TD><FONT FACE="{Font.MONOSPACE}">'
            f"[+] {hidden_node_count} node{plural} hidden"
            f"</FONT></TD>"
            f"</TR>"
        )

    def _build_condition_block(
        self, node: StatechartNode, line_color: str = "black"
    ) -> str:
        """
        Builds the label rows listing the transition conditions of a node.

        Nodes that terminate the statechart only get their start condition, because the
        remaining conditions never fire for them.

        :param node: The node whose conditions are listed.
        :param line_color: The color of the lines separating the rows.
        :return: The condition rows of the label.
        """
        life_cycle_state = self.statechart.life_cycle_state[node]
        return "".join(
            self._build_condition_row(
                prefix=self.condition_prefix(condition.kind),
                condition=condition,
                is_active=condition.kind.can_trigger_from(life_cycle_state),
                line_color=line_color,
            )
            for condition in node.effective_conditions
            if condition.kind is TransitionKind.START
            or not isinstance(node, TerminalNode)
        )

    @staticmethod
    def condition_prefix(transition_kind: TransitionKind) -> str:
        """
        :param transition_kind: The transition whose condition a row lists.
        :return: The text a condition row starts with.
        """
        return transition_kind.name.lower()

    def _build_condition_row(
        self,
        prefix: str,
        condition: TransitionCondition,
        is_active: bool,
        line_color: str,
    ) -> str:
        """
        :param prefix: The label prefix for this condition.
        :param condition: The condition to render.
        :param is_active: Whether this condition can trigger from the current lifecycle state.
        :param line_color: The color of the line separating rows.
        :return: The HTML table rows for this condition.
        """
        rendered = self._render_condition(condition, grayed_out=not is_active)
        if is_active:
            font_tag = f'<FONT FACE="{Font.MONOSPACE}">{prefix}:{rendered}</FONT>'
        else:
            font_tag = f'<FONT FACE="{Font.MONOSPACE}" COLOR="{self.disabled_condition_color.to_hex()}">{prefix}:{rendered}</FONT>'
        return (
            f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{self.metrics.line_width}"></TD></TR>'
            f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{self.metrics.line_width}">{font_tag}</TD></TR>'
        )

    def _render_condition(
        self, condition: TransitionCondition, grayed_out: bool = False
    ) -> str:
        """
        Writes a condition for display, coloring every term in the value that term
        takes.

        When grayed out, individual term status coloring is omitted so the outer
        disabled color applies uniformly.

        The value is the term's own, not the observation of the node it names: an
        ``is_succeeded`` term is false until that node succeeded, however decisive that
        node's observation already is.

        The terms are cut out before the rest is reformatted, so that a node whose name
        reads as a logical operator is still recognised as one term.

        :param condition: The condition to write.
        :param grayed_out: Whether to render without term coloring.
        :return: The condition as an HTML label fragment.
        """
        text = str(condition)
        if grayed_out:
            return format_condition_text(text, color_constants=False)
        values_by_term = {
            f'"{variable.display_name}"': variable.resolve()
            for variable in condition.variables
        }
        if not values_by_term:
            return format_condition_text(text, color_constants=True)
        terms = re.compile(f"({'|'.join(re.escape(term) for term in values_by_term)})")
        return "".join(
            (
                self._color_term(part, values_by_term[part])
                if part in values_by_term
                else format_condition_text(part, color_constants=True)
            )
            for part in terms.split(text)
        )

    def _color_term(self, term: str, value: ObservationStateValues) -> str:
        """
        :param term: The term as it is written in the condition, quotes included.
        :param value: The value that term takes.
        :return: The term in the color of that value.
        """
        style = ObservationDrawingStyle.of_observation(value)
        return f'<FONT COLOR="{style.color.to_hex()}">{term}</FONT>'

    def _escape_name(self, name: str) -> str:
        """
        :param name: The node name to escape.
        :return: The name in the quoted form pydot stores it under.
        """
        return f'"{name}"'

    def _get_cluster_of_node(
        self, node_name: str, graph: pydot.Graph | pydot.Cluster
    ) -> pydot.Cluster | None:
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
        node: StatechartNode,
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
                penwidth=self.metrics.line_width,
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

    def _create_pydot_node(self, node: StatechartNode) -> pydot.Node:
        """
        :param node: The node to draw.
        :return: A labelled pydot node shaped and styled by the node's plot
            specification.
        """
        label = self._format_graph_node(node=node)
        pydot_node = pydot.Node(
            str(node.unique_name),
            label=label,
            shape=node.plot_specifications.shape,
            color="black",
            style=node.plot_specifications.style,
            margin=0,
            fillcolor="white",
            fontname=Font.SANS_SERIF,
            fontsize=self.metrics.font_size,
            penwidth=self.metrics.line_width,
        )
        return pydot_node

    def to_dot_graph(self) -> pydot.Graph:
        """
        Draws every visible node and transition of the statechart.

        :return: The drawn graph.
        """
        self._cluster_map[None] = self.graph
        self._add_nodes(self.graph, self.statechart.top_level_nodes)
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

    def _is_drawn(self, node: StatechartNode) -> bool:
        """
        :param node: The node to check.
        :return: Whether the node appears in the drawing, which it does not if it or one
            of its ancestors is invisible, or if one of its ancestors collapses its
            children.
        """
        if not node.plot_specifications.visible:
            return False
        return all(
            ancestor.plot_specifications.visible
            and not ancestor.plot_specifications.collapse_children
            for ancestor in node.path
        )

    def _add_nodes(
        self,
        parent_cluster: pydot.Graph | pydot.Cluster,
        nodes: list[StatechartNode],
    ):
        """
        Draws the given nodes, recursing into the children of every goal that does not
        collapse them.

        :param parent_cluster: The graph or cluster the nodes are drawn in.
        :param nodes: The nodes to draw.
        """
        for i, node in enumerate(nodes):
            # Skip invisible nodes entirely, as well as the children of a
            # CompositeNode that is invisible or collapses them.
            if not self._is_drawn(node):
                continue

            if (
                isinstance(node, CompositeNode)
                and not node.plot_specifications.collapse_children
            ):
                goal_cluster = self._add_cluster(node, parent_cluster)
                self._add_node(
                    graph=goal_cluster,
                    node=node,
                )
                self._add_nodes(goal_cluster, node.children)
                continue

            self._add_node(
                parent_cluster,
                node=node,
            )

    def _add_cluster(
        self,
        node: StatechartNode,
        parent_cluster: pydot.Graph | pydot.Cluster,
    ):
        """
        Opens the cluster that a goal and its children are drawn in.

        :param node: The goal to draw a border around.
        :param parent_cluster: The graph or cluster the new cluster is nested in.
        :return: The new cluster.
        """
        goal_cluster = pydot.Cluster(
            graph_name=str(node.unique_name),
            fontname=Font.SANS_SERIF,
            fontsize=self.metrics.font_size,
            style=NodeDrawingStyle.COMPOSITE.style,
            color="black",
            fillcolor="white",
            penwidth=self.metrics.line_width,
        )
        parent_cluster.add_subgraph(goal_cluster)
        self._cluster_map[node] = goal_cluster
        return goal_cluster

    def _add_edges(self):
        """
        Draws one arrow per dependency, from the node a condition reads to the node that
        reads it.
        """
        for dependency in self._dependencies_to_draw():
            self._add_dependency_edge(dependency)

    def _dependencies_to_draw(self) -> list[ConditionDependency]:
        """
        :return: Every dependency the drawing shows, once per pair of nodes however many
            of the owner's conditions read it, leaving out those with an endpoint that is
            not drawn or that sits in another cluster.
        """
        dependencies: dict[ConditionDependency, ConditionDependency] = {}
        condition: TransitionCondition
        for (
            owner_index,
            observed_index,
            condition,
        ) in self.statechart.rx_graph.edge_index_map().values():
            condition_owner = self.statechart.rx_graph.get_node_data(owner_index)
            observed_node = self.statechart.rx_graph.get_node_data(observed_index)
            if not self._is_drawn(condition_owner) or not self._is_drawn(observed_node):
                continue
            if not self._are_nodes_in_same_cluster(condition_owner, observed_node):
                continue
            new_dependency = ConditionDependency(condition_owner, observed_node)
            dependency = dependencies.setdefault(new_dependency, new_dependency)
            dependency.conditions.append(condition)
        return list(dependencies.values())

    def _are_nodes_in_same_cluster(
        self, condition_owner: StatechartNode, observed_node: StatechartNode
    ) -> bool:
        """
        :param condition_owner: The node whose condition reads the other.
        :param observed_node: The node that condition reads.
        :return: Whether both nodes are drawn in the same cluster.
        """
        owner_parent = condition_owner.parent_node
        observed_parent = observed_node.parent_node

        if owner_parent is None or observed_parent is None:
            return owner_parent is observed_parent

        return owner_parent.name == observed_parent.name

    def _edge_clusters_kwargs(
        self,
        graph: pydot.Graph | pydot.Cluster,
        src_name: str,
        dst_name: str,
    ) -> dict[str, object]:
        """
        Determines the edge attributes that clip an edge at a cluster border instead of
        letting it reach into the cluster.

        :param graph: The graph or cluster the edge is drawn in.
        :param src_name: The name of the node the edge starts at.
        :param dst_name: The name of the node the edge ends at.
        :return: The ``ltail`` and ``lhead`` attributes for the endpoints that sit in a
            cluster.
        """
        kwargs: dict[str, object] = {}
        dst_cluster = self._get_cluster_of_node(dst_name, graph)
        src_cluster = self._get_cluster_of_node(src_name, graph)
        if dst_cluster is not None:
            kwargs["lhead"] = dst_cluster.get_name()
        if src_cluster is not None:
            kwargs["ltail"] = src_cluster.get_name()
        return kwargs

    def _add_dependency_edge(self, dependency: ConditionDependency) -> None:
        """
        Draws the arrow of a single dependency.

        It carries the observation of the node it leaves and nothing else: which of the
        owner's conditions read that node is written in the owner's condition rows.

        :param dependency: The dependency to draw.
        """
        graph = self._cluster_map[dependency.condition_owner.parent_node]
        source_name = str(dependency.observed_node.unique_name)
        destination_name = str(dependency.condition_owner.unique_name)

        kwargs = self._edge_clusters_kwargs(graph, source_name, destination_name)
        style = ObservationDrawingStyle.UNKNOWN
        color = (
            style.color.to_hex()
            if self._is_dependency_active(dependency)
            else self.disabled_condition_color.to_hex()
        )

        graph.add_edge(
            pydot.Edge(
                src=source_name,
                dst=destination_name,
                color=color,
                penwidth=style.line_width(self.metrics),
                minlen=dependency.minimum_rank_distance(self.metrics),
                arrowsize=self.metrics.arrow_size,
                **kwargs,
            )
        )

    def _is_dependency_active(self, dependency: ConditionDependency) -> bool:
        """
        :param dependency: The dependency to check.
        :return: Whether at least one of the conditions this dependency bundles can
            currently trigger from its owner's life cycle state.
        """
        life_cycle_state = self.statechart.life_cycle_state[dependency.condition_owner]
        return any(
            condition.kind.can_trigger_from(life_cycle_state)
            for condition in dependency.conditions
        )
