"""
Drill-down views of the CRAM architecture: packages, subpackages and classes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from typing_extensions import Any, ClassVar, Dict, List, TYPE_CHECKING

from cramera.knowledge.architecture_entities import Package, PythonClass, SubPackage
from cramera.knowledge.enums import EdgeKind, NodeGroup
from cramera.knowledge.subgraph import (
    DetailEntry,
    GraphEdge,
    GraphNode,
    GraphPanelPayload,
    SubgraphAccumulator,
)

if TYPE_CHECKING:
    from cramera.knowledge.knowledge_base import EpisodeKnowledgeBase


@dataclass(kw_only=True)
class SubgraphViewPayload(GraphPanelPayload):
    """
    A drill-down view of one package, subpackage or class.
    """

    breadcrumb: str
    """
    Breadcrumb label shown above the subgraph.
    """

    def panel_options(self) -> Dict[str, Any]:
        """
        The breadcrumb; a drill-down view sends nothing else of its own.
        """
        return {"breadcrumb": self.breadcrumb}

    MAXIMUM_CLASSES_SHOWN: ClassVar[int] = 150
    """
    At most this many classes are drawn in one drill-down view.
    """
    MAXIMUM_SUBCLASSES_SHOWN: ClassVar[int] = 80
    """
    At most this many subclasses are drawn in a class inheritance view.
    """

    @staticmethod
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
        if python_class.docstring_summary:
            lines.append(python_class.docstring_summary)
        if drill_hint:
            lines.append("double-click: inheritance view")
        return lines

    @classmethod
    def _add_classes(
        cls,
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
        :param total: The total number of classes before capping, for the truncation
            note.
        :return: Extra detail lines for the parent (a truncation notice, if any).
        """
        name_to_id: Dict[str, str] = {}
        for python_class in shown:
            class_id = python_class.qualified_name
            view.add(
                class_id,
                python_class.name,
                NodeGroup.PYTHON_CLASS,
                cls._class_lines(python_class),
            )
            view.add_edge(parent_id, class_id, EdgeKind.PROPERTY, "defines")
            name_to_id.setdefault(python_class.name, class_id)
        for python_class in shown:
            for base in python_class.bases:
                if (
                    base in name_to_id
                    and name_to_id[base] != python_class.qualified_name
                ):
                    view.add_edge(
                        python_class.qualified_name,
                        name_to_id[base],
                        EdgeKind.TYPE,
                        "inherits",
                    )
        if total > len(shown):
            return [
                "showing the %d largest of %d classes (by method count)"
                % (len(shown), total)
            ]
        return []

    @classmethod
    def for_package(
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
            NodeGroup.PACKAGE,
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
                NodeGroup.SUBPACKAGE,
                [
                    "a SubPackage of " + subpackage.package,
                    "%d modules · %d classes"
                    % (subpackage.module_count, subpackage.class_count),
                    "double-click to open",
                ],
            )
            view.add_edge(package.name, subpackage.name, EdgeKind.PROPERTY, "contains")
        note = cls._add_classes(
            view, package.name, top_level[: cls.MAXIMUM_CLASSES_SHOWN], len(top_level)
        )
        if note:
            view.details[package.name].lines += note
        return cls(
            breadcrumb=package.name,
            nodes=view.nodes,
            edges=view.edges,
            details=view.details,
        )

    @classmethod
    def for_subpackage(
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
            NodeGroup.SUBPACKAGE,
            [
                "a SubPackage of " + subpackage.package,
                "%d modules · %d classes"
                % (subpackage.module_count, subpackage.class_count),
            ],
        )
        note = cls._add_classes(
            view, subpackage.name, classes[: cls.MAXIMUM_CLASSES_SHOWN], len(classes)
        )
        if note:
            view.details[subpackage.name].lines += note
        return cls(
            breadcrumb=subpackage.name.split(".", 1)[1],
            nodes=view.nodes,
            edges=view.edges,
            details=view.details,
        )

    @classmethod
    def for_class(
        cls, knowledge_base: EpisodeKnowledgeBase, python_class: PythonClass
    ) -> SubgraphViewPayload:
        """
        Inheritance view of one class: bases above, repo subclasses below.

        :param knowledge_base: The knowledge base the class's bases/subclasses are read
            from.
        :param python_class: The class to render.
        """
        view = SubgraphAccumulator()
        class_id = python_class.qualified_name
        view.add(
            class_id,
            python_class.name,
            NodeGroup.PYTHON_CLASS,
            cls._class_lines(python_class, drill_hint=False),
        )
        # direct base classes: resolve inside the repo (same package preferred),
        # otherwise show them as external
        for base in python_class.bases:
            candidates = [
                entry for entry in knowledge_base.classes if entry.name == base
            ]
            resolved_base = next(
                (
                    entry
                    for entry in candidates
                    if entry.package == python_class.package
                ),
                candidates[0] if candidates else None,
            )
            if resolved_base:
                base_id = resolved_base.qualified_name
                if base_id not in view.details:
                    view.add(
                        base_id,
                        resolved_base.name,
                        NodeGroup.PYTHON_CLASS,
                        cls._class_lines(resolved_base),
                    )
            else:
                base_id = "external:" + base
                if base_id not in view.details:
                    view.add(
                        base_id,
                        base,
                        NodeGroup.EXTERNAL_CLASS,
                        ["external base class (outside the repo)"],
                    )
            view.add_edge(class_id, base_id, EdgeKind.TYPE, "inherits")
        # every subclass in the repo (matched by base name)
        subclasses = [
            entry
            for entry in knowledge_base.classes
            if python_class.name in entry.bases and entry.qualified_name != class_id
        ]
        for subclass in subclasses[: cls.MAXIMUM_SUBCLASSES_SHOWN]:
            subclass_id = subclass.qualified_name
            if subclass_id not in view.details:
                view.add(
                    subclass_id,
                    subclass.name,
                    NodeGroup.PYTHON_CLASS,
                    cls._class_lines(subclass),
                )
            view.add_edge(subclass_id, class_id, EdgeKind.TYPE, "inherits")
        if len(subclasses) > cls.MAXIMUM_SUBCLASSES_SHOWN:
            view.details[class_id].lines.append(
                "showing %d of %d subclasses"
                % (cls.MAXIMUM_SUBCLASSES_SHOWN, len(subclasses))
            )
        return cls(
            breadcrumb=python_class.name,
            nodes=view.nodes,
            edges=view.edges,
            details=view.details,
        )
