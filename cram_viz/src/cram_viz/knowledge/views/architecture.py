"""
Drill-down views of the CRAM architecture: packages, subpackages and classes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from typing_extensions import Any, Dict, List, TYPE_CHECKING

from cram_viz.knowledge.architecture_entities import Package, PythonClass, SubPackage
from cram_viz.knowledge.enums import EdgeKind, NodeGroup
from cram_viz.knowledge.subgraph import (
    DetailEntry,
    GraphEdge,
    GraphNode,
    SubgraphAccumulator,
)

if TYPE_CHECKING:
    from cram_viz.knowledge.knowledge_base import EpisodeKnowledgeBase


@dataclass
class SubgraphViewPayload:
    """
    A drill-down view of one package, subpackage or class.
    """

    ok: bool
    """
    Always ``True`` — these views have no failure mode.
    """

    crumb: str
    """
    Breadcrumb label shown above the subgraph.
    """

    nodes: List[GraphNode]
    """
    Every node in this view.
    """

    edges: List[GraphEdge]
    """
    Every edge in this view.
    """

    details: Dict[str, DetailEntry]
    """
    Detail-panel entry per node id.
    """

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the frontend's graph panel expects.
        """
        return {
            "ok": self.ok,
            "crumb": self.crumb,
            "nodes": [node.to_payload() for node in self.nodes],
            "edges": [edge.to_payload() for edge in self.edges],
            "details": {
                node_id: asdict(entry) for node_id, entry in self.details.items()
            },
        }


def _class_id(python_class: PythonClass) -> str:
    """
    Graph node id of a scanned class (module-qualified).

    :param python_class: The scanned class to id.
    """
    return python_class.module + "." + python_class.name


def _class_lines(python_class: PythonClass, drill_hint: bool = True) -> List[str]:
    """
    Detail lines shown for a class node.

    :param python_class: The scanned class to describe.
    :param drill_hint: Whether to append the "double-click" drill-down hint.
    """
    lines = [
        "a PythonClass",
        "package: " + python_class.package,
        "module: " + python_class.module,
        "methods: %d" % python_class.methods,
    ]
    if python_class.bases:
        lines.append("bases: " + ", ".join(python_class.bases))
    if python_class.doc:
        lines.append(python_class.doc)
    if drill_hint:
        lines.append("double-click: inheritance view")
    return lines


def _add_classes(
    view: SubgraphAccumulator,
    parent_id: str,
    shown: List[PythonClass],
    total: int,
) -> List[str]:
    """
    Add class nodes plus their on-screen inheritance edges to a view.

    :param view: The subgraph accumulator to add nodes and edges to.
    :param parent_id: Id of the package/subpackage node the classes belong to.
    :param shown: The classes actually drawn (already capped).
    :param total: The total number of classes before capping, for the truncation note.
    :return: Extra detail lines for the parent (a truncation notice, if any).
    """
    name_to_id: Dict[str, str] = {}
    for python_class in shown:
        class_id = _class_id(python_class)
        view.add(
            class_id, python_class.name, NodeGroup.PYCLASS, _class_lines(python_class)
        )
        view.edges.append(GraphEdge(parent_id, class_id, EdgeKind.PROP, "defines"))
        name_to_id.setdefault(python_class.name, class_id)
    for python_class in shown:
        for base in python_class.bases:
            if base in name_to_id and name_to_id[base] != _class_id(python_class):
                view.edges.append(
                    GraphEdge(
                        _class_id(python_class),
                        name_to_id[base],
                        EdgeKind.TYPE,
                        "inherits",
                    )
                )
    if total > len(shown):
        return [
            "showing the %d largest of %d classes (by method count)"
            % (len(shown), total)
        ]
    return []


class ArchitectureViews:
    """
    Drill-down views of the CRAM architecture: packages, subpackages and classes.
    """

    #: at most this many classes are drawn in one drill-down view
    CLASS_CAP = 150

    #: at most this many subclasses are drawn in a class inheritance view
    SUBCLASS_CAP = 80

    @classmethod
    def package_view(
        cls, knowledge_base: EpisodeKnowledgeBase, package: Package
    ) -> SubgraphViewPayload:
        """
        Inside view of a package: its subpackages and top-level classes.

        :param knowledge_base: The knowledge base the package's entities are read from.
        :param package: The package to render.
        """
        view = SubgraphAccumulator()
        subpackages = [
            entry
            for entry in knowledge_base.subpackages
            if entry.package == package.name
        ]
        top_level = sorted(
            (
                entry
                for entry in knowledge_base.classes
                if entry.package == package.name and entry.subpackage == package.name
            ),
            key=lambda entry: -entry.methods,
        )
        view.add(
            package.name,
            package.name,
            NodeGroup.CONCEPT,
            [
                "a Package",
                package.description,
                "%d modules · %d classes" % (package.module_count, package.class_count),
            ],
        )
        for subpackage in subpackages:
            view.add(
                subpackage.name,
                subpackage.name.split(".", 1)[1],
                NodeGroup.KLASS,
                [
                    "a SubPackage of " + subpackage.package,
                    "%d modules · %d classes"
                    % (subpackage.module_count, subpackage.class_count),
                    "double-click to open",
                ],
            )
            view.edges.append(
                GraphEdge(package.name, subpackage.name, EdgeKind.PROP, "contains")
            )
        note = _add_classes(
            view, package.name, top_level[: cls.CLASS_CAP], len(top_level)
        )
        if note:
            view.details[package.name].lines += note
        return SubgraphViewPayload(
            True, package.name, view.nodes, view.edges, view.details
        )

    @classmethod
    def subpackage_view(
        cls, knowledge_base: EpisodeKnowledgeBase, subpackage: SubPackage
    ) -> SubgraphViewPayload:
        """
        Inside view of a subpackage: its classes with inheritance edges.

        :param knowledge_base: The knowledge base the subpackage's classes are read
            from.
        :param subpackage: The subpackage to render.
        """
        view = SubgraphAccumulator()
        classes = sorted(
            (
                entry
                for entry in knowledge_base.classes
                if entry.subpackage == subpackage.name
            ),
            key=lambda entry: -entry.methods,
        )
        view.add(
            subpackage.name,
            subpackage.name.split(".", 1)[1],
            NodeGroup.KLASS,
            [
                "a SubPackage of " + subpackage.package,
                "%d modules · %d classes"
                % (subpackage.module_count, subpackage.class_count),
            ],
        )
        note = _add_classes(
            view, subpackage.name, classes[: cls.CLASS_CAP], len(classes)
        )
        if note:
            view.details[subpackage.name].lines += note
        return SubgraphViewPayload(
            True, subpackage.name.split(".", 1)[1], view.nodes, view.edges, view.details
        )

    @classmethod
    def class_view(
        cls, knowledge_base: EpisodeKnowledgeBase, python_class: PythonClass
    ) -> SubgraphViewPayload:
        """
        Inheritance view of one class: bases above, repo subclasses below.

        :param knowledge_base: The knowledge base the class's bases/subclasses are read
            from.
        :param python_class: The class to render.
        """
        view = SubgraphAccumulator()
        class_id = _class_id(python_class)
        view.add(
            class_id,
            python_class.name,
            NodeGroup.PYCLASS,
            _class_lines(python_class, drill_hint=False),
        )
        # direct base classes: resolve inside the repo (same package preferred),
        # otherwise show them as external
        for base in python_class.bases:
            candidates = [
                entry for entry in knowledge_base.classes if entry.name == base
            ]
            pick = next(
                (
                    entry
                    for entry in candidates
                    if entry.package == python_class.package
                ),
                candidates[0] if candidates else None,
            )
            if pick:
                base_id = _class_id(pick)
                if base_id not in view.details:
                    view.add(base_id, pick.name, NodeGroup.PYCLASS, _class_lines(pick))
            else:
                base_id = "ext:" + base
                if base_id not in view.details:
                    view.add(
                        base_id,
                        base,
                        NodeGroup.UPPER,
                        ["external base class (outside the repo)"],
                    )
            view.edges.append(GraphEdge(class_id, base_id, EdgeKind.TYPE, "inherits"))
        # every subclass in the repo (matched by base name)
        subclasses = [
            entry
            for entry in knowledge_base.classes
            if python_class.name in entry.bases and _class_id(entry) != class_id
        ]
        for subclass in subclasses[: cls.SUBCLASS_CAP]:
            subclass_id = _class_id(subclass)
            if subclass_id not in view.details:
                view.add(
                    subclass_id,
                    subclass.name,
                    NodeGroup.PYCLASS,
                    _class_lines(subclass),
                )
            view.edges.append(
                GraphEdge(subclass_id, class_id, EdgeKind.TYPE, "inherits")
            )
        if len(subclasses) > cls.SUBCLASS_CAP:
            view.details[class_id].lines.append(
                "showing %d of %d subclasses" % (cls.SUBCLASS_CAP, len(subclasses))
            )
        return SubgraphViewPayload(
            True, python_class.name, view.nodes, view.edges, view.details
        )
