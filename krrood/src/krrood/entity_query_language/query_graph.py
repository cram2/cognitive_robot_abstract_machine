from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from ordered_set import OrderedSet
from typing_extensions import ClassVar, Optional, Dict, Any

from krrood.entity_query_language.rules.conclusion import Conclusion
from krrood.entity_query_language.rules.conclusion_selector import ConclusionSelector
from krrood.entity_query_language.query.query import (
    Query,
)
from krrood.entity_query_language.query.operations import OrderedBy, GroupedBy
from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.operators.concatenation import Concatenation
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.operators.core_logical_operators import (
    LogicalOperator,
)
from krrood.entity_query_language.core.base_expressions import (
    SymbolicExpression,
    Filter,
)
from krrood.entity_query_language.evaluation import is_condition_participant
from krrood.entity_query_language.core.variable import (
    Variable,
    Literal,
    InstantiatedVariable,
)
from krrood.entity_query_language.core.mapped_variable import (
    MappedVariable,
    Attribute,
    Index,
    FlatVariable,
    Call,
)
from krrood.entity_query_language.operators.comparator import Comparator

from krrood.rustworkx_utils import (
    GraphVisualizer,
    RWXNode as RXUtilsNode,
    ColorLegend as RXUtilsColorLegend,
)

import rustworkx as rx

_UNSATISFIED_BORDER_COLOR: str = "red"


def _is_faded_gate(node, satisfied_condition_ids: OrderedSet[UUID]) -> bool:
    """
    Return True if *node* is an unsatisfied condition participant.

    Such nodes act as "gates" that the BFS in
    :meth:`QueryGraph._propagate_faded_subtrees` refuses to traverse through.
    """
    expr = node.data
    if expr is None:
        return False
    if not is_condition_participant(expr):
        return False
    return expr._id_ not in satisfied_condition_ids


@dataclass
class QueryGraph:
    """
    Represents a query graph for visualizing and introspecting query structures.
    """

    query: SymbolicExpression
    """
    An expression representing the query.
    """

    satisfied_condition_ids: Optional[OrderedSet[UUID]] = None
    """
    Optional frozenset of satisfied condition UUIDs for coloring condition nodes.

    When provided, unsatisfied condition nodes are colored grey, while satisfied
    condition nodes keep their type-based color.
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
        if GraphVisualizer is None:
            raise ModuleNotFoundError(
                "rustworkx_utils is not installed. Please install it with `pip install rustworkx_utils`"
            )
        if isinstance(self.query, Query):
            self.query.build()
        self.construct_graph()
        if self.satisfied_condition_ids is not None:
            self._propagate_faded_subtrees()

    def _propagate_faded_subtrees(self):
        """
        Mark unsatisfied condition nodes and their exclusive descendants as faded.

        A node is faded when every path from the root to that node passes through
        at least one unsatisfied condition node.  We compute this by BFS from the
        root, refusing to traverse *through* unsatisfied condition nodes, then
        marking every node the BFS did **not** reach as faded.

        Exception: descendants of a *satisfied* QuantifiedConditional (Exists/ForAll)
        are always reachable even though their internal condition IDs are not tracked
        in the outer satisfied_condition_ids set.  The BFS carries a skip_gate flag
        for exactly this case.
        """
        from krrood.entity_query_language.operators.logical_quantifiers import (
            QuantifiedConditional,
        )

        root_node = self.expression_node_map.get(self.query._root_)
        if root_node is None:
            return

        reachable: set = set()
        queue: list = [(root_node, False)]  # (node, skip_gate)
        while queue:
            node, skip_gate = queue.pop(0)
            if node.id in reachable:
                continue
            if not skip_gate and _is_faded_gate(node, self.satisfied_condition_ids):
                continue
            reachable.add(node.id)
            child_skip_gate = skip_gate or (
                isinstance(node.data, QuantifiedConditional)
                and node.data._id_ in self.satisfied_condition_ids
            )
            for child in node.children:
                if child.id not in reachable:
                    queue.append((child, child_skip_gate))

        for node in self.expression_node_map.values():
            if node.id not in reachable:
                node.faded = True
                node.border_color = _UNSATISFIED_BORDER_COLOR

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
        filename: str = "query_graph.pdf",
    ):
        """
        Visualizes the graph using the specified layout and style options.

        Provides a graphical visualization of the graph with customizable options for
        size, layout, spacing, and labeling. Requires the rustworkx_utils library for
        execution.

        :returns: The rendered visualization object.
        :raises:`ModuleNotFoundError` If rustworkx_utils is not installed.
        """
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
            filename=filename,
        )
        return visualizer.render()

    @staticmethod
    def get_expression_name(expression: Any) -> str:
        """
        Retrieves the name of a symbolic expression for visualization purposes.

        :param expression: The symbolic expression to get the name for.
        :type expression: SymbolicExpression
        :return: The name of the expression.
        """
        match expression:
            case Attribute():
                return f".{expression._name_.split('.')[-1]}"
            case Index():
                return f"[{expression._name_.split('[')[-1]}"
            case FlatVariable():
                return "Flatten"
            case Call():
                expression: Call
                args = [
                    QueryGraph.get_expression_name(arg) for arg in expression._args_
                ]
                kwargs = [f"{k}={v}" for k, v in expression._kwargs_.items()]
                args_and_kwargs = args + kwargs
                return f"({','.join(args_and_kwargs)})"
            case Query():
                return f"({','.join(QueryGraph.get_expression_name(v) for v in expression._selected_variables_)})"
            case SymbolicExpression():
                return expression._name_
            case _:
                return repr(expression)

    def construct_graph(
        self,
        expression: Optional[SymbolicExpression] = None,
    ) -> QueryNode:
        """
        Construct the graph representation of the query, used for visualization and
        introspection.
        """
        expression = expression if expression is not None else self.query._root_

        if expression in self.expression_node_map:
            return self.expression_node_map[expression]

        is_satisfied = (
            self.satisfied_condition_ids is not None
            and is_condition_participant(expression)
            and expression._id_ in self.satisfied_condition_ids
        )
        node = QueryNode(
            self.get_expression_name(expression),
            self.graph,
            color=ColorLegend.from_expression(expression, self.satisfied_condition_ids),
            data=expression,
            is_satisfied=is_satisfied,
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
        selected_var_ids = (
            [v._id_ for v in parent_expression._selected_variables_]
            if isinstance(parent_expression, Query)
            else []
        )
        for child in parent_expression._children_:
            child_node = self.construct_graph(child)
            if child._id_ in selected_var_ids:
                child_node.enclosed = True
            child_node.parent = parent_node


@dataclass
class ColorLegend(RXUtilsColorLegend):
    """
    Represents a color legend entry for visualizing query graph nodes.

    Maps each expression type to a color.
    """

    @classmethod
    def from_expression(
        cls,
        expression: SymbolicExpression,
        satisfied_condition_ids: Optional[OrderedSet[UUID]] = None,
    ) -> ColorLegend:
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
    A node in the query graph.

    Overrides the default enclosed name to "Selected Variable".
    """

    enclosed_name: ClassVar[str] = "Selected Variable"
    is_satisfied: bool = field(default=False)
    """
    True if this node's expression is a condition participant whose evaluation result
    was True.

    Grounded directly on satisfied_condition_ids, not derived from the faded propagation
    pass.
    """
