from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import ClassVar, Optional, Dict

from .rules.conclusion import Conclusion
from .rules.conclusion_selector import ConclusionSelector
from .query.query import (
    Query,
)
from .query.operations import OrderedBy, GroupedBy
from .query.quantifiers import ResultQuantifier
from .operators.concatenation import Concatenation
from .operators.aggregators import Aggregator
from .operators.core_logical_operators import LogicalOperator
from .core.base_expressions import SymbolicExpression, Filter
from .core.variable import Variable, Literal
from .core.mapped_variable import MappedVariable
from .operators.comparator import Comparator

try:
    from rustworkx_utils import (
        GraphVisualizer,
        RWXNode as RXUtilsNode,
        ColorLegend as RXUtilsColorLegend,
    )
except ImportError:
    GraphVisualizer = None
    RXUtilsNode = None
    RXUtilsColorLegend = None

import rustworkx as rx


@dataclass
class QueryGraph:
    """
    Represents a query graph for visualizing and introspecting query structures.
    """

    query: SymbolicExpression
    """
    An expression representing the query.
    """
    graph: rx.PyDAG = field(init=False, default_factory=rx.PyDAG)
    """
    The graph representation of the query, used for visualization and introspection.
    """
    expression_node_map: Dict[SymbolicExpression, QueryNode] = field(
        init=False, default_factory=dict
    )
    """
    A mapping from symbolic expressions to their corresponding nodes in the graph.
    """

    def __post_init__(self):
        self.construct_graph()

    def visualize(
        self,
        figure_size=(35, 30),
        node_size=7000,
        font_size=25,
        spacing_x: float = 4,
        spacing_y: float = 4,
        curve_scale: float = 0.5,
        layout: str = "tidy",
        edge_style: str = "orthogonal",
        label_max_chars_per_line: Optional[int] = 13,
    ):
        """
        Visualizes the graph using the specified layout and style options.

        Provides a graphical visualization of the graph with customizable options for
        size, layout, spacing, and labeling. Requires the rustworkx_utils library for
        execution.

        :returns: The rendered visualization object.
        :raises: `ModuleNotFoundError` If rustworkx_utils is not installed.
        """
        if not GraphVisualizer:
            raise ModuleNotFoundError(
                "rustworkx_utils is not installed. Please install it with `pip install rustworkx_utils`"
            )
        visualizer = GraphVisualizer(
            node=self.expression_node_map[self.query._root_],
            figsize=figure_size,
            node_size=node_size,
            font_size=font_size,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            curve_scale=curve_scale,
            layout=layout,
            edge_style=edge_style,
            label_max_chars_per_line=label_max_chars_per_line,
        )
        return visualizer.render()

    def construct_graph(
        self,
        expression: Optional[SymbolicExpression] = None,
    ) -> QueryNode:
        """
        Construct the graph representation of the query, used for visualization and introspection.
        """
        expression = expression if expression is not None else self.query._root_

        if expression in self.expression_node_map:
            return self.expression_node_map[expression]

        node = QueryNode(
            expression._name_,
            self.graph,
            color=ColorLegend.from_expression(expression),
            data=expression,
        )
        self.expression_node_map[expression] = node

        if isinstance(expression, ResultQuantifier):
            node.wrap_subtree = True

        self._add_children_to_graph(node)

        return node

    def _add_children_to_graph(
        self,
        parent_node: QueryNode,
    ):
        """
        Adds child nodes to the graph recursively.

        :param parent_node: The parent node of the children to add.
        """
        parent_expression = parent_node.data
        for child in parent_expression._children_:
            child_node = self.construct_graph(child)
            if isinstance(parent_expression, Query):
                if child._binding_id_ in [
                    v._binding_id_ for v in parent_expression._selected_variables_
                ]:
                    child_node.enclosed = True
            child_node.parent = parent_node


@dataclass
class ColorLegend(RXUtilsColorLegend):
    """
    Represents a color legend entry for visualizing query graph nodes. Maps each expression type to a color.
    """

    @classmethod
    def from_expression(cls, expression: SymbolicExpression) -> ColorLegend:
        name = expression.__class__.__name__
        color = "white"
        match expression:
            case Filter() | OrderedBy() | GroupedBy():
                color = "#17becf"
            case Aggregator():
                name = "Aggregator"
                color = "#F54927"
            case ResultQuantifier():
                name = "ResultQuantifier"
                color = "#9467bd"
            case Query():
                name = "QueryObjectDescriptor"
                color = "#d62728"
            case Literal():
                color = "#949292"
            case Variable():
                color = "cornflowerblue"
            case Concatenation():
                name = "Union"
                color = "#949292"
            case MappedVariable():
                name = "DomainMapping"
                color = "#8FC7B8"
            case Comparator():
                name = "Comparator"
                color = "#ff7f0e"
            case ConclusionSelector():
                name = "ConclusionSelector"
                color = "#eded18"
            case LogicalOperator():
                name = "LogicalOperator"
                color = "#2ca02c"
            case Conclusion():
                name = "Conclusion"
                color = "#8cf2ff"
        return cls(name=name, color=color)


@dataclass
class QueryNode(RXUtilsNode):
    """
    A node in the query graph. Overrides the default enclosed name to "Selected Variable".
    """

    enclosed_name: ClassVar[str] = "Selected Variable"
