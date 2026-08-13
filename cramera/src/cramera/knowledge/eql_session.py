"""
Running one EQL query against the knowledge base and rendering its result.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass

from typing_extensions import (
    Any,
    Dict,
    List,
    Optional,
)

from krrood.entity_query_language import factories as eql_factories
from krrood.entity_query_language.evaluable import Evaluable
from krrood.entity_query_language.scope import eql_factory_namespace
from semantic_digital_twin.spatial_types import Point3

from cramera.body_geometry import position_label
from cramera.knowledge.architecture_entities import Package, PythonClass, SubPackage
from cramera.payload import CrameraPayload
from cramera.knowledge.entity import NamedEntity
from cramera.knowledge.entities import (
    ActionEpisode,
    Arm,
    BenchObject,
    Gripper,
    JointMotion,
    Robot,
)
from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase


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


@dataclass
class RowRenderer:
    """
    Renders one evaluated EQL result into the answer rows the panel shows.

    Collects the graph-node ids to highlight while rendering, so the walk does not have
    to thread an output list through every level.
    """

    limit: int = 200
    """
    Maximum number of answer rows to render.
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
        rows: List[Dict[str, Any]] = []
        if result is None:
            return _RenderedRows(rows, self.highlight, False)
        if isinstance(result, (str, int, float, bool, Point3)):
            rows.append({"value": self._jsonable(result)})
            return _RenderedRows(rows, self.highlight, False)
        if is_dataclass(result) and not isinstance(result, type):
            rows.append(self._entity_row(result))
            return _RenderedRows(rows, self.highlight, False)
        try:
            iterator = iter(result)
        except TypeError:
            rows.append({"value": self._jsonable(result)})
            return _RenderedRows(rows, self.highlight, False)
        for item in iterator:
            if len(rows) >= self.limit:
                return _RenderedRows(rows, self.highlight, True)
            rows.append(self._item_row(item))
        return _RenderedRows(rows, self.highlight, False)

    def _item_row(self, item: Any) -> Dict[str, Any]:
        """
        One arbitrary query result item as an answer row.

        :param item: The query result item to render as a row.
        """
        if isinstance(item, Point3):
            return {"value": self._jsonable(item)}
        if is_dataclass(item) and not isinstance(item, type):
            return self._entity_row(item)
        if isinstance(item, Mapping):  # a unification row from set_of()
            row = {}
            for key, value in item.items():
                name = self._entity_name(value)
                if name and is_dataclass(value) and not isinstance(value, type):
                    self.highlight.append(name)
                row[str(key)] = self._jsonable(value)
            return row
        return {"value": self._jsonable(item)}

    def _entity_row(self, item: Any) -> Dict[str, Any]:
        """
        One entity as an answer row, collecting the ids it lights up.

        :param item: The entity to render as a row.
        """
        name = self._entity_name(item)
        if name:
            self.highlight.append(name)
        if isinstance(item, PythonClass):
            # classes aren't graph nodes — light up their subpackage + package instead
            self.highlight.append(item.subpackage)
            self.highlight.append(item.package)
        row = {"__entity__": name or repr(item), "__type__": type(item).__name__}
        for entity_field in fields(item):
            if entity_field.name != "name":
                row[entity_field.name] = self._jsonable(vars(item)[entity_field.name])
        return row

    @classmethod
    def _jsonable(cls, value: Any) -> Any:
        """
        A JSON-serializable rendering of one query result value.

        :param value: The raw query result value to render.
        """
        if isinstance(value, Point3):
            return position_label(value)
        if is_dataclass(value) and not isinstance(value, type):
            return cls._entity_name(value) or repr(value)
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, (str, int, bool)) or value is None:
            return value
        return repr(value)

    @staticmethod
    def _entity_name(value: Any) -> Optional[str]:
        """
        The entity's name, or None for a value that is not one of our entities.

        :param value: The query result value to name.
        """
        return str(value.name) if isinstance(value, NamedEntity) else None


@dataclass
class EqlSession:
    """
    Runs EQL queries against one recorded episode.

    The knowledge base is held here rather than fetched per query, so a session is
    pinned to the episode it was opened for.
    """

    knowledge_base: EpisodeKnowledgeBase
    """
    The recorded episode every query of this session ranges over.
    """

    @classmethod
    def of_active_scene(cls) -> "EqlSession":
        """
        A session against the scene bundle the server currently serves.
        """
        return cls.of_scene(None)

    @classmethod
    def of_scene(cls, scene: Optional[str]) -> "EqlSession":
        """
        A session against one named scene bundle.

        :param scene: Name of the scene to query, or None for the active one.
        """
        return cls(knowledge_base=EpisodeKnowledgeBase.of_scene(scene))

    def namespace(self) -> Dict[str, Any]:
        """
        A namespace for evaluating one EQL query (fresh variables each time).
        """
        namespace: Dict[str, Any] = eql_factory_namespace()
        namespace.update(
            Point3=Point3,
            Gripper=Gripper,
            Arm=Arm,
            Robot=Robot,
            BenchObject=BenchObject,
            ActionEpisode=ActionEpisode,
            JointMotion=JointMotion,
            Package=Package,
            SubPackage=SubPackage,
            PythonClass=PythonClass,
            objects=self.knowledge_base.objects,
            episodes=self.knowledge_base.episodes,
            arms=self.knowledge_base.arms,
            grippers=self.knowledge_base.grippers,
            joints=self.knowledge_base.joints,
            robots=[self.knowledge_base.robot],
            packages=self.knowledge_base.packages,
            subpackages=self.knowledge_base.subpackages,
            classes=self.knowledge_base.classes,
        )
        # ready-made query variables, one per entity type
        namespace["scene_object"] = eql_factories.variable(
            BenchObject, domain=self.knowledge_base.objects
        )
        namespace["episode"] = eql_factories.variable(
            ActionEpisode, domain=self.knowledge_base.episodes
        )
        namespace["arm"] = eql_factories.variable(Arm, domain=self.knowledge_base.arms)
        namespace["joint"] = eql_factories.variable(
            JointMotion, domain=self.knowledge_base.joints
        )
        namespace["robot"] = eql_factories.variable(
            Robot, domain=[self.knowledge_base.robot]
        )
        namespace["package"] = eql_factories.variable(
            Package, domain=self.knowledge_base.packages
        )
        namespace["subpackage"] = eql_factories.variable(
            SubPackage, domain=self.knowledge_base.subpackages
        )
        namespace["python_class"] = eql_factories.variable(
            PythonClass, domain=self.knowledge_base.classes
        )
        return namespace

    def run(self, code: str, limit: int = 200) -> RenderResult:
        """
        Execute an EQL query string and return its rendered result.

        The last expression of ``code`` is the query; preceding statements are executed
        as setup.

        :param code: The EQL query source.
        :param limit: Maximum number of result rows to return.
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
            result = eval(
                compile(ast.Expression(last.value), "<eql>", "eval"), namespace
            )
        else:
            exec(compile(tree, "<eql>", "exec"), namespace)
            result = namespace.get("result")

        if isinstance(result, Evaluable):
            result = result.evaluate()
        rendered = RowRenderer(limit=limit).rows_of(result)
        kind = (
            "rows"
            if rendered.rows and "__entity__" not in rendered.rows[0]
            else "entities"
        )
        return RenderResult(
            kind=kind,
            rows=rendered.rows,
            count=len(rendered.rows),
            more=rendered.more,
            highlight=sorted(set(rendered.highlight)),
        )
