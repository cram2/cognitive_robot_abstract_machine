"""
Running one EQL query against the knowledge base and rendering its result.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, fields, is_dataclass

from typing_extensions import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from krrood.entity_query_language import factories as eql_factories
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.scope import eql_factory_namespace

from cram_viz.knowledge.architecture_entities import Package, PythonClass, SubPackage
from cram_viz.knowledge.entities import (
    ActionEpisode,
    Arm,
    BenchObject,
    Gripper,
    JointMotion,
    Position,
    Robot,
)
from cram_viz.knowledge.knowledge_base import get_knowledge_base


@dataclass
class RenderResult:
    """
    The rendered result of one EQL query.
    """

    ok: bool
    """
    Always ``True`` — a query that cannot run raises instead of returning this.
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
        }


@dataclass
class _RenderedRows:
    """
    A query result rendered into answer rows, before it is wrapped as a RenderResult.
    """

    rows: List[Dict[str, Any]]
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


@runtime_checkable
class NamesAnEntity(Protocol):
    """
    A query result that carries an entity name the viewer can highlight.
    """

    name: Any


@runtime_checkable
class MapsNamesToValues(Protocol):
    """
    A result row that binds names to values, such as EQL's unification dictionary.
    """

    def items(self) -> Any:
        """
        The name/value pairs of this row.
        """


def fresh_namespace() -> Dict[str, Any]:
    """
    A namespace for evaluating one EQL query (fresh variables each time).
    """
    kb = get_knowledge_base()
    namespace: Dict[str, Any] = eql_factory_namespace()
    namespace.update(
        Position=Position,
        Gripper=Gripper,
        Arm=Arm,
        Robot=Robot,
        BenchObject=BenchObject,
        ActionEpisode=ActionEpisode,
        JointMotion=JointMotion,
        Package=Package,
        SubPackage=SubPackage,
        PythonClass=PythonClass,
        objects=kb.objects,
        episodes=kb.episodes,
        arms=kb.arms,
        grippers=kb.grippers,
        joints=kb.joints,
        robots=[kb.robot],
        packages=kb.packages,
        subpackages=kb.subpackages,
        classes=kb.classes,
    )
    # ready-made query variables so one-liners stay short
    namespace["obj"] = eql_factories.variable(BenchObject, domain=kb.objects)
    namespace["ep"] = eql_factories.variable(ActionEpisode, domain=kb.episodes)
    namespace["arm"] = eql_factories.variable(Arm, domain=kb.arms)
    namespace["j"] = eql_factories.variable(JointMotion, domain=kb.joints)
    namespace["rob"] = eql_factories.variable(Robot, domain=[kb.robot])
    namespace["pkg"] = eql_factories.variable(Package, domain=kb.packages)
    namespace["sub"] = eql_factories.variable(SubPackage, domain=kb.subpackages)
    namespace["cls"] = eql_factories.variable(PythonClass, domain=kb.classes)
    return namespace


def _entity_name(value: Any) -> Optional[str]:
    """
    The entity's name attribute, or None for non-entities.

    :param value: The value to check for a name attribute.
    """
    return str(value.name) if isinstance(value, NamesAnEntity) else None


def _jsonable(value: Any) -> Any:
    """
    A JSON-serializable rendering of one query result value.

    :param value: The raw query result value to render.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return _entity_name(value) or repr(value)
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return repr(value)


def run_query(code: str, limit: int = 200) -> RenderResult:
    """
    Execute an EQL query string and return its rendered result.

    The last expression of ``code`` is the query; preceding statements are executed as
    setup.

    :param code: The EQL query source.
    :param limit: Maximum number of result rows to return.
    """
    namespace = fresh_namespace()
    tree = ast.parse(code, mode="exec")
    if not tree.body:
        raise ValueError("empty query")
    last = tree.body[-1]
    if isinstance(last, ast.Expr):
        if len(tree.body) > 1:
            preamble = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(preamble, "<eql>", "exec"), namespace)
        result = eval(compile(ast.Expression(last.value), "<eql>", "eval"), namespace)
    else:
        exec(compile(tree, "<eql>", "exec"), namespace)
        result = namespace.get("result")

    if isinstance(result, Evaluable):
        result = result.evaluate()
    rendered = _result_rows(result, limit)
    kind = (
        "rows" if rendered.rows and "__entity__" not in rendered.rows[0] else "entities"
    )
    return RenderResult(
        ok=True,
        kind=kind,
        rows=rendered.rows,
        count=len(rendered.rows),
        more=rendered.more,
        highlight=sorted(set(rendered.highlight)),
    )


def _result_rows(result: Any, limit: int) -> _RenderedRows:
    """
    Render a query result into answer rows.

    :param result: The evaluated query result to render.
    :param limit: Maximum number of result rows to return.
    """
    rows: List[Dict[str, Any]] = []
    highlight: List[str] = []
    if result is None:
        return _RenderedRows(rows, highlight, False)
    if isinstance(result, (str, int, float, bool)):
        rows.append({"value": _jsonable(result)})
        return _RenderedRows(rows, highlight, False)
    if is_dataclass(result) and not isinstance(result, type):
        rows.append(_entity_row(result, highlight))
        return _RenderedRows(rows, highlight, False)
    try:
        iterator = iter(result)
    except TypeError:
        rows.append({"value": _jsonable(result)})
        return _RenderedRows(rows, highlight, False)
    for item in iterator:
        if len(rows) >= limit:
            return _RenderedRows(rows, highlight, True)
        rows.append(_item_row(item, highlight))
    return _RenderedRows(rows, highlight, False)


def _entity_row(item: Any, highlight: List[str]) -> Dict[str, Any]:
    """
    One entity as an answer row; collects the ids to highlight.

    :param item: The entity to render as a row.
    :param highlight: Output list entity/package ids to highlight are appended to.
    """
    name = _entity_name(item)
    if name:
        highlight.append(name)
    if isinstance(item, PythonClass):
        # classes aren't graph nodes — light up their subpackage + package instead
        highlight.append(item.subpackage)
        highlight.append(item.package)
    row = {"__entity__": name or repr(item), "__type__": type(item).__name__}
    for entity_field in fields(item):
        if entity_field.name != "name":
            row[entity_field.name] = _jsonable(vars(item)[entity_field.name])
    return row


def _item_row(item: Any, highlight: List[str]) -> Dict[str, Any]:
    """
    One arbitrary query result item as an answer row.

    :param item: The query result item to render as a row.
    :param highlight: Output list entity ids to highlight are appended to.
    """
    if is_dataclass(item) and not isinstance(item, type):
        return _entity_row(item, highlight)
    if isinstance(item, MapsNamesToValues):  # a unification row from set_of()
        row = {}
        for key, value in item.items():
            if (
                is_dataclass(value)
                and not isinstance(value, type)
                and _entity_name(value)
            ):
                highlight.append(_entity_name(value))
            row[str(key)] = _jsonable(value)
        return row
    return {"value": _jsonable(item)}
