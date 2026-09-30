"""
Running one EQL query over a set of declared domains and rendering its result.

Knows nothing about where the domains came from: a recorded episode and a running demo
are queried through the same runner.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from datetime import datetime

from typing_extensions import (
    AbstractSet,
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    runtime_checkable,
)

from krrood.entity_query_language import factories as eql_factories
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.scope import eql_factory_namespace
from semantic_digital_twin.spatial_types import Point3, Pose

from cramera.body_geometry import NumericPose, pose_label, position_label
from cramera.knowledge.entity import NamedEntity
from cramera.knowledge.query_domain import QueryDomain
from cramera.knowledge.replay import ReplayWindow
from cramera.knowledge.query_verbalization import QueryVerbalization
from cramera.knowledge.query_vocabulary import QueryVocabulary
from cramera.knowledge.queryable_knowledge import InMemoryEvaluation, QueryEvaluation
from cramera.knowledge.workspace_classes import (
    WorkspaceClassIndex,
    WorkspaceClassNamespace,
)
from cramera.payload import CrameraPayload

DEFAULT_ROW_LIMIT = 200
"""
Maximum number of answer rows a query returns unless the caller asks for fewer.
"""


@runtime_checkable
class CarriesATimestamp(Protocol):
    """
    An entity recording when it happened, such as a detected event.
    """

    timestamp: datetime


@runtime_checkable
class HighlightsRelatedNodes(Protocol):
    """
    An entity whose answer row lights up graph nodes besides its own.
    """

    def related_highlight_ids(self) -> List[str]:
        """
        Ids of the further graph nodes a row for this entity highlights.
        """


@dataclass(kw_only=True)
class RenderResult(CrameraPayload):
    """
    The rendered result of one EQL query.

    Not a :class:`~cramera.knowledge.subgraph.GraphPanelPayload`: the EQL panel shows
    answer rows, not a graph, so this carries no nodes or edges.
    """

    kind: str
    """
    ``"rows"`` for arbitrary answer rows, ``"entities"`` when every row names an entity.
    """

    rows: List[Dict[str, Any]]
    """
    The query's answer rows; each row's own keys depend on what the query asked for.
    """

    count: int
    """
    Number of rows returned (``len(rows)``).
    """

    more: bool
    """
    Whether the result was truncated at ``limit``.
    """

    highlight: List[str]
    """
    Ids of the graph nodes this result should highlight, sorted and deduplicated.
    """

    replay: List[Optional[ReplayWindow]] = field(default_factory=list)
    """
    The window worth replaying around each row's moment, one entry per row and None for
    a row naming no moment.

    Beside the rows rather than in them: a row holds what its query asked for, so a
    viewer that knows nothing of replay shows the answer unchanged instead of rendering
    the window as a column.
    """

    verbalization: Optional[QueryVerbalization] = None
    """
    The query read back as English, or None when there was no expression to word.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the frontend's EQL panel expects.
        """
        return {
            "ok": self.ok,
            "kind": self.kind,
            "rows": self.rows,
            "count": self.count,
            "more": self.more,
            "highlight": self.highlight,
            "replay": [
                window.to_payload() if window is not None else None
                for window in self.replay
            ],
            "verbalization": (
                self.verbalization.to_payload()
                if self.verbalization is not None
                else None
            ),
        }


@dataclass
class AnswerRow:
    """
    One rendered answer row and what the viewer may do with it.
    """

    values: Dict[str, Any]
    """
    The row's own columns, keyed as the panel shows them.
    """

    replay: Optional[ReplayWindow] = None
    """
    The window of the demo recording worth replaying around this row's moment, or None
    when the row names no moment.
    """


@dataclass
class _RenderedRows:
    """
    A query result rendered into answer rows, before it is wrapped as a RenderResult.
    """

    rows: List[AnswerRow]
    """
    The rendered answer rows.
    """

    highlight: List[str]
    """
    Ids of the graph nodes to highlight, collected while rendering.
    """

    more: bool
    """
    Whether rendering stopped early because ``limit`` was reached.
    """


@dataclass
class RowRenderer:
    """
    Renders one evaluated EQL result into the answer rows the panel shows.

    Collects the graph-node ids to highlight while rendering, so the walk does not have
    to thread an output list through every level.
    """

    limit: int = DEFAULT_ROW_LIMIT
    """
    Maximum number of answer rows to render.
    """

    entity_types: Tuple[Type[Any], ...] = ()
    """
    Types whose instances are titled and highlighted as entities.

    Carrying a ``name`` is not enough: unrelated dataclasses have one too, and a row for
    one of those is a plain value rather than something the graph can light up.
    """

    highlightable_ids: AbstractSet[str] = frozenset()
    """
    Ids the viewer can light up, such as the scene objects it currently shows.

    Any string answer value equal to one of these is highlighted, whatever the query
    asked for; answer values naming nothing the viewer shows are left alone.
    """

    highlight: List[str] = field(default_factory=list)
    """
    Ids of the graph nodes the rendered rows should highlight.
    """

    def rows_of(self, result: Any) -> _RenderedRows:
        """
        Render a query result into answer rows.

        :param result: The evaluated query result to render.
        """
        rows: List[AnswerRow] = []
        if result is None:
            return _RenderedRows(rows, self.highlight, False)
        if isinstance(result, (str, int, float, bool, Point3, Pose, NumericPose)):
            rows.append(AnswerRow({"value": self._jsonable(result)}))
            return _RenderedRows(rows, self.highlight, False)
        if is_dataclass(result) and not isinstance(result, type):
            rows.append(self._entity_row(result))
            return _RenderedRows(rows, self.highlight, False)
        try:
            iterator = iter(result)
        except TypeError:
            rows.append(AnswerRow({"value": self._jsonable(result)}))
            return _RenderedRows(rows, self.highlight, False)
        for item in iterator:
            if len(rows) >= self.limit:
                return _RenderedRows(rows, self.highlight, True)
            rows.append(self._item_row(item))
        return _RenderedRows(rows, self.highlight, False)

    def _item_row(self, item: Any) -> AnswerRow:
        """
        One arbitrary query result item as an answer row.

        :param item: The query result item to render as a row.
        """
        if isinstance(item, (Point3, Pose, NumericPose)):
            return AnswerRow({"value": self._jsonable(item)})
        if is_dataclass(item) and not isinstance(item, type):
            return self._entity_row(item)
        if isinstance(item, Mapping):  # a unification row from set_of()
            columns = self._column_names([str(key) for key in item])
            values = {}
            window: Optional[ReplayWindow] = None
            for column, value in zip(columns, item.values()):
                name = self._row_title(value)
                if name:
                    self.highlight.append(name)
                if window is None and isinstance(value, datetime):
                    window = ReplayWindow.around(value)
                values[column] = self._jsonable(value)
            return AnswerRow(values, window)
        return AnswerRow({"value": self._jsonable(item)})

    @staticmethod
    def _column_names(keys: List[str]) -> List[str]:
        """
        The headings a set of asked-for values is shown under.

        A selected attribute is named after its own type (``ShapeUnderTest.name``), which
        is the answer's subject and reads as noise repeated in every heading. Dropping it
        is only safe while the shortened headings stay distinct.

        :param keys: The selected values' own names, in the order they were asked for.
        """
        shortened = [key.rsplit(".", 1)[-1] for key in keys]
        return shortened if len(set(shortened)) == len(keys) else keys

    def _entity_row(self, item: Any) -> AnswerRow:
        """
        One entity as an answer row, collecting the ids it lights up.

        A ``repr=False`` field is internal bookkeeping and stays out of the row. A
        field declared ``init=False`` with a plain default lives on the class rather
        than in the instance ``__dict__``, so its value is read from the field's
        default instead.

        :param item: The entity to render as a row.
        """
        name = self._row_title(item)
        if name:
            self.highlight.append(name)
        if isinstance(item, HighlightsRelatedNodes):
            self.highlight.extend(item.related_highlight_ids())
        values = {"__entity__": name or repr(item), "__type__": type(item).__name__}
        instance_values = vars(item)
        for entity_field in fields(item):
            if entity_field.name == "name" or not entity_field.repr:
                continue
            if entity_field.name in instance_values:
                values[entity_field.name] = self._jsonable(
                    instance_values[entity_field.name]
                )
            elif entity_field.default is not MISSING:
                values[entity_field.name] = self._jsonable(entity_field.default)
        window = (
            ReplayWindow.around(item.timestamp)
            if isinstance(item, CarriesATimestamp)
            and isinstance(item.timestamp, datetime)
            else None
        )
        return AnswerRow(values, window)

    def _jsonable(self, value: Any) -> Any:
        """
        A JSON-serializable rendering of one query result value.

        :param value: The raw query result value to render.
        """
        if isinstance(value, NumericPose):
            return value.label
        if isinstance(value, Point3):
            return position_label(value)
        if isinstance(value, Pose):
            return pose_label(value)
        if isinstance(value, datetime):
            return value.isoformat(sep=" ", timespec="seconds")
        if is_dataclass(value) and not isinstance(value, type):
            return self._row_title(value) or repr(value)
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, str):
            if value in self.highlightable_ids:
                self.highlight.append(value)
            return value
        if isinstance(value, (int, bool)) or value is None:
            return value
        return repr(value)

    def _row_title(self, value: Any) -> Optional[str]:
        """
        The name a row for this value is titled and highlighted by, or None.

        :param value: The query result value to title.
        """
        if isinstance(value, self.entity_types):
            return str(value.name)
        return self._entity_name(value)

    @staticmethod
    def _entity_name(value: Any) -> Optional[str]:
        """
        The entity's name, or None for a value that is not one of our entities.

        :param value: The query result value to name.
        """
        return str(value.name) if isinstance(value, NamedEntity) else None


@dataclass
class EqlQueryRunner:
    """
    Executes EQL query strings against a fixed set of domains.
    """

    domains: List[QueryDomain]
    """
    The ready-made variables every query of this runner may range over.
    """

    extra_names: Dict[str, Any] = field(default_factory=dict)
    """
    Further names a query may use, such as constants or the raw domain lists.
    """

    evaluation: QueryEvaluation = field(default_factory=InMemoryEvaluation)
    """
    Where a query of this runner is worked out.
    """

    class_index: WorkspaceClassIndex = field(
        default_factory=WorkspaceClassIndex.of_repository
    )
    """
    The workspace classes a query of this runner may name besides its own domains.
    """

    highlightable_ids: FrozenSet[str] = frozenset()
    """
    Ids the viewer can light up; see :attr:`RowRenderer.highlightable_ids`.
    """

    @property
    def entity_types(self) -> Tuple[Type[Any], ...]:
        """
        The types this runner's answers are rendered as entities.
        """
        return tuple(domain.entity_type for domain in self.domains)

    def vocabulary(self) -> QueryVocabulary:
        """
        Everything a query of this runner may name, for a query box to offer.
        """
        return QueryVocabulary(
            domains=self.domains,
            extra_names=self.extra_names,
            class_index=self.class_index,
        )

    def namespace(self) -> Dict[str, Any]:
        """
        A namespace for evaluating one EQL query (fresh variables each time).
        """
        namespace = WorkspaceClassNamespace(index=self.class_index)
        namespace.update(eql_factory_namespace())
        for domain in self.domains:
            namespace[domain.entity_type.__name__] = domain.entity_type
        for domain in self.domains:
            namespace[domain.name] = (
                eql_factories.variable(domain.entity_type)
                if domain.objects is None
                else eql_factories.variable(domain.entity_type, domain=domain.objects)
            )
        namespace.update(self.extra_names)
        return namespace

    def build(self, code: str) -> Any:
        """
        Build an EQL query string into the expression it stands for, unevaluated.

        The last expression of ``code`` is the query; preceding statements are executed
        as setup.

        :param code: The EQL query source.
        """
        namespace = self.namespace()
        tree = ast.parse(code, mode="exec")
        if not tree.body:
            raise ValueError("empty query")
        last = tree.body[-1]
        if isinstance(last, ast.Expr):
            if len(tree.body) > 1:
                preamble = ast.Module(body=tree.body[:-1], type_ignores=[])
                exec(compile(preamble, "<eql>", "exec"), namespace)
            return eval(compile(ast.Expression(last.value), "<eql>", "eval"), namespace)
        exec(compile(tree, "<eql>", "exec"), namespace)
        return namespace.get("result")

    def verbalize(self, code: str) -> Optional[QueryVerbalization]:
        """
        Read an EQL query string back as English without evaluating it.

        A sentence is a nicety: a query whose code cannot even be built still gets
        offered (running it reports its own error), so nothing here is allowed to
        raise.

        :param code: The EQL query source.
        :return: Both renderings, or None when the code does not build or does not
            build into a query.
        """
        try:
            expression = self.build(code)
        except Exception:
            return None
        if not isinstance(expression, Evaluable):
            return None
        return QueryVerbalization.of_expression(expression)

    def run(self, code: str, limit: int = DEFAULT_ROW_LIMIT) -> RenderResult:
        """
        Execute an EQL query string and return its rendered result.

        The last expression of ``code`` is the query; preceding statements are executed
        as setup.

        :param code: The EQL query source.
        :param limit: Maximum number of result rows to return.
        """
        result = self.build(code)
        verbalization = None
        if isinstance(result, Evaluable):
            # worded before evaluating: building the sentence leaves the expression
            # evaluable, whereas the evaluated result is rows and no longer a question
            verbalization = QueryVerbalization.of_expression(result)
            result = self.evaluation.evaluate(result)
        rendered = RowRenderer(
            limit=limit,
            entity_types=self.entity_types,
            highlightable_ids=self.highlightable_ids,
        ).rows_of(result)
        kind = (
            "rows"
            if rendered.rows and "__entity__" not in rendered.rows[0].values
            else "entities"
        )
        return RenderResult(
            kind=kind,
            rows=[row.values for row in rendered.rows],
            count=len(rendered.rows),
            more=rendered.more,
            highlight=sorted(set(rendered.highlight)),
            replay=[row.replay for row in rendered.rows],
            verbalization=verbalization,
        )
