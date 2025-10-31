from __future__ import annotations

import re
from dataclasses import dataclass, field

import pydot
from typing_extensions import List, Dict, Optional, Union, Set, TYPE_CHECKING

from giskardpy.motion_statechart.data_types import LifeCycleValues, ObservationState

from giskardpy.motion_statechart.graph_node import (
    EndMotion,
    CancelMotion,
    MotionStatechartNode,
)

if TYPE_CHECKING:
    from giskardpy.motion_statechart.motion_statechart import MotionStatechart


def extract_node_names_from_condition(condition: str) -> Set[str]:
    matches = re.findall(r'"(.*?)"|\'(.*?)\'', condition)
    return set(match for group in matches for match in group if match)


def format_condition(condition: str) -> str:
    condition = condition.replace(" and ", "<BR/>       and ")
    condition = condition.replace(" or ", "<BR/>       or ")
    condition = condition.replace("1.0", "True")
    condition = condition.replace("0.0", "False")
    return condition


NotStartedColor = "#9F9F9F"
MyBLUE = "#0000DD"
MyGREEN = "#006600"
MyORANGE = "#996900"
MyRED = "#993000"
MyGRAY = "#E0E0E0"

ChatGPTGreen = "#28A745"
ChatGPTOrange = "#E6AC00"
ChatGPTRed = "#DC3545"
ChatGPTBlue = "#007BFF"
ChatGPTGray = "#8F959E"

StartCondColor = ChatGPTGreen
PauseCondColor = ChatGPTOrange
EndCondColor = ChatGPTRed
ResetCondColor = ChatGPTGray

MonitorTrueGreen = "#B6E5A0"
MonitorFalseRed = "#FF5024"
FONT = "sans-serif"
LineWidth = 4
NodeSep = 1
RankSep = 1
ArrowSize = 1
Fontsize = 15
GoalNodeStyle = "filled"
GoalNodeShape = "none"
GoalClusterStyle = "filled"
MonitorStyle = "filled, rounded"
MonitorShape = "rectangle"
TaskStyle = "filled, diagonals"
TaskShape = "rectangle"
ConditionFont = "monospace"

ResetSymbol = "⟲"

ObservationStateToColor: Dict[ObservationState, str] = {
    ObservationState.unknown: ResetCondColor,
    ObservationState.true: MonitorTrueGreen,
    ObservationState.false: MonitorFalseRed,
}

ObservationStateToSymbol: Dict[ObservationState, str] = {
    ObservationState.unknown: "?",
    ObservationState.true: "True",
    ObservationState.false: "False",
}

ObservationStateToEdgeStyle: Dict[ObservationState, Dict[str, str]] = {
    ObservationState.unknown: {
        "penwidth": (LineWidth * 1.5) / 2,
        # 'label': '<<FONT FACE="monospace"><B>?</B></FONT>>',
        "fontsize": Fontsize * 1.333,
    },
    ObservationState.true: {"penwidth": LineWidth * 1.5},
    ObservationState.false: {"style": "dashed", "penwidth": LineWidth * 1.5},
}

LiftCycleStateToColor: Dict[LifeCycleValues, str] = {
    LifeCycleValues.NOT_STARTED: ResetCondColor,
    LifeCycleValues.RUNNING: StartCondColor,
    LifeCycleValues.PAUSED: PauseCondColor,
    LifeCycleValues.DONE: EndCondColor,
    LifeCycleValues.FAILED: "red",
}

LiftCycleStateToSymbol: Dict[LifeCycleValues, str] = {
    # LifeCycleState.not_started: '○',
    LifeCycleValues.NOT_STARTED: "—",
    LifeCycleValues.RUNNING: "▶",
    # LifeCycleState.paused: '⏸',
    LifeCycleValues.PAUSED: "<B>||</B>",
    LifeCycleValues.DONE: "■",
    LifeCycleValues.FAILED: "red",
}


@dataclass
class MotionStatechartGraphviz:
    motion_statechart: MotionStatechart
    graph: pydot.Graph = field(init=False)
    compact: bool = False

    def __post_init__(self):
        self.graph = pydot.Dot(
            graph_type="digraph",
            graph_name="",
            ranksep=RankSep if not self.compact else RankSep * 0.5,
            nodesep=NodeSep if not self.compact else NodeSep * 0.5,
            compound=True,
            ratio="compress",
        )

    def format_motion_graph_node(
        self,
        node: MotionStatechartNode,
    ) -> str:
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
            f"  <TD><B> {node.name} </B></TD>"
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
        if self.compact:
            label += (
                f"<TR>" f'  <TD WIDTH="100%" HEIGHT="{LineWidth*2.5}"></TD>' f"</TR>"
            )
        else:
            label += self._build_condition_block(node)
        label += f"</TABLE>>"
        return label

    def _build_condition_block(
        self, node: MotionStatechartNode, line_color="black"
    ) -> str:
        start_condition = format_condition(str(node._start_condition))
        pause_condition = format_condition(str(node._pause_condition))
        end_condition = format_condition(str(node._end_condition))
        reset_condition = format_condition(str(node._reset_condition))
        label = (
            f'<TR><TD WIDTH="100%" BGCOLOR="{line_color}" HEIGHT="{LineWidth}"></TD></TR>'
            f'<TR><TD ALIGN="LEFT" BALIGN="LEFT" CELLPADDING="{LineWidth}"><FONT FACE="{ConditionFont}">start:{start_condition}</FONT></TD></TR>'
        )
        if not isinstance(node, (EndMotion, CancelMotion)):
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

    def escape_name(self, name: str) -> str:
        return f'"{name}"'

    def get_cluster_of_node(
        self, node_name: str, graph: Union[pydot.Graph, pydot.Cluster]
    ) -> Optional[pydot.Cluster]:
        node_cluster = None
        for cluster in graph.get_subgraphs():
            if (
                len(cluster.get_node(self.escape_name(node_name))) == 1
                or len(cluster.get_node(node_name)) == 1
            ):
                node_cluster = cluster
                break
        return node_cluster

    def add_node(
        self,
        graph: pydot.Graph,
        node: MotionStatechartNode,
    ) -> pydot.Node:
        pydot_node = self._create_pydot_node(node)
        if len(node._plot_extra_boarder_styles) == 0:
            graph.add_node(pydot_node)
            return pydot_node
        child = pydot_node
        for index, style in enumerate(node._plot_extra_boarder_styles):
            c = pydot.Cluster(
                graph_name=f"{node.name}",
                penwidth=LineWidth,
                style=node._plot_extra_boarder_styles[index],
                color="black",
            )
            if index == 0:
                c.add_node(child)
            else:
                c.add_subgraph(child)
            child = c
        if len(node._plot_extra_boarder_styles) > 0:
            graph.add_subgraph(c)
        return pydot_node

    def _create_pydot_node(self, node: MotionStatechartNode) -> pydot.Node:
        label = self.format_motion_graph_node(node=node)
        pydot_node = pydot.Node(
            str(node.name),
            label=label,
            shape=node._plot_shape,
            color="black",
            style=node._plot_style,
            margin=0,
            fillcolor="white",
            fontname=FONT,
            fontsize=Fontsize,
            penwidth=LineWidth,
        )
        return pydot_node

    def cluster_name_to_goal_name(self, name: str) -> str:
        if name == "":
            return name
        if '"' in name:
            return name[9:-1]
        return name[8:]

    def to_dot_graph(self) -> pydot.Graph:
        self.add_goal_cluster(self.graph)
        return self.graph

    def add_goal_cluster(
        self,
        parent_cluster: Union[pydot.Graph, pydot.Cluster],
    ):
        for i, node in enumerate(self.motion_statechart.nodes):
            # todo check parent cluster
            self.add_node(
                parent_cluster,
                node=node,
            )

        # for i, goal in enumerate(self.execution_state.goals):
        #     if self.execution_state.goal_parents[i] == self.cluster_name_to_goal_name(
        #         parent_cluster.get_name()
        #     ):
        #         obs_state = self.execution_state.goal_state[i]
        #         goal_cluster = pydot.Cluster(
        #             graph_name=goal.name,
        #             fontname=FONT,
        #             fontsize=Fontsize,
        #             style=GoalClusterStyle,
        #             color="black",
        #             fillcolor="white",
        #             penwidth=LineWidth,
        #         )
        #         self.add_node(
        #             graph=goal_cluster,
        #             node_msg=goal,
        #             style=GoalNodeStyle,
        #             shape=GoalNodeShape,
        #             obs_state=obs_state,
        #             life_cycle_state=self.execution_state.goal_life_cycle_state[i],
        #         )
        #         obs_states[goal.name] = obs_state
        #         parent_cluster.add_subgraph(goal_cluster)
        #         self.add_goal_cluster(goal_cluster, obs_states)
        #         my_goals.append(goal)
        # %% add edges
        # self.add_edges(parent_cluster, my_tasks, my_monitors, my_goals, obs_states)

    def add_edges(
        self,
        graph: Union[pydot.Graph, pydot.Cluster],
        tasks: List[MotionStatechartNode],
        monitors: List[MotionStatechartNode],
        goals: List[MotionStatechartNode],
        obs_states: Dict[str, ObservationState],
    ) -> pydot.Graph:
        all_nodes = tasks + monitors + goals
        all_node_name = [
            node.name for node in all_nodes
        ]  # + [self.cluster_name_to_goal_name(graph.get_name())]
        for node in all_nodes:
            node_name = node.name
            node_cluster = self.get_cluster_of_node(node_name, graph)
            for sub_node_name in extract_node_names_from_condition(
                node.start_condition
            ):
                if sub_node_name not in all_node_name:
                    continue
                sub_node_cluster = self.get_cluster_of_node(sub_node_name, graph)
                kwargs = {}
                if node_cluster is not None:
                    kwargs["lhead"] = node_cluster.get_name()
                if sub_node_cluster is not None:
                    kwargs["ltail"] = sub_node_cluster.get_name()
                kwargs.update(ObservationStateToEdgeStyle[obs_states[sub_node_name]])
                graph.add_edge(
                    pydot.Edge(
                        src=sub_node_name,
                        dst=node_name,
                        color=StartCondColor,
                        arrowsize=ArrowSize,
                        **kwargs,
                    )
                )
            for sub_node_name in extract_node_names_from_condition(
                node.pause_condition
            ):
                if sub_node_name not in all_node_name:
                    continue
                sub_node_cluster = self.get_cluster_of_node(sub_node_name, graph)
                kwargs = {}
                if node_cluster is not None:
                    kwargs["lhead"] = node_cluster.get_name()
                if sub_node_cluster is not None:
                    kwargs["ltail"] = sub_node_cluster.get_name()
                kwargs.update(ObservationStateToEdgeStyle[obs_states[sub_node_name]])
                graph.add_edge(
                    pydot.Edge(
                        sub_node_name,
                        node_name,
                        color=PauseCondColor,
                        minlen=0,
                        arrowsize=ArrowSize,
                        **kwargs,
                    )
                )
            for sub_node_name in extract_node_names_from_condition(node.end_condition):
                if sub_node_name not in all_node_name:
                    continue
                sub_node_cluster = self.get_cluster_of_node(sub_node_name, graph)
                kwargs = {}
                if node_cluster is not None:
                    kwargs["ltail"] = node_cluster.get_name()
                if sub_node_cluster is not None:
                    kwargs["lhead"] = sub_node_cluster.get_name()
                kwargs.update(ObservationStateToEdgeStyle[obs_states[sub_node_name]])
                graph.add_edge(
                    pydot.Edge(
                        node_name,
                        sub_node_name,
                        color=EndCondColor,
                        arrowhead="none",
                        arrowtail="normal",
                        dir="both",
                        arrowsize=ArrowSize,
                        **kwargs,
                    )
                )
            for sub_node_name in extract_node_names_from_condition(
                node.reset_condition
            ):
                if sub_node_name not in all_node_name:
                    continue
                sub_node_cluster = self.get_cluster_of_node(sub_node_name, graph)
                kwargs = {}
                if node_cluster is not None:
                    kwargs["ltail"] = node_cluster.get_name()
                if sub_node_cluster is not None:
                    kwargs["lhead"] = sub_node_cluster.get_name()
                kwargs.update(ObservationStateToEdgeStyle[obs_states[sub_node_name]])
                graph.add_edge(
                    pydot.Edge(
                        node_name,
                        sub_node_name,
                        color=ResetCondColor,
                        arrowhead="none",
                        arrowtail="normal",
                        minlen=0,
                        dir="both",
                        arrowsize=ArrowSize,
                        **kwargs,
                    )
                )

        return graph
