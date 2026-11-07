from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from types import ModuleType

from cramph.node import StatechartNode, TerminalNode

from .node_classes import NodeClassesOfModules

# %% rebuild audit


@dataclass
class NodeRebuildAudit(NodeClassesOfModules):
    """
    Finds the statechart node classes of some modules that hold other nodes among their
    constructor arguments but leave rebuilding them to
    :meth:`~cramph.node.StatechartNode.rebuilt_by`, which rebinds every argument as it
    is.

    A node ending the statechart is never part of the nodes a plan rebuilds, so it is
    not audited.
    """

    ignored_modules: list[ModuleType] = field(default_factory=list)
    """
    The modules among :attr:`modules` whose node classes are not audited.
    """

    def nodes_holding_nodes_they_do_not_rebuild(self) -> list[type[StatechartNode]]:
        """
        :return: Every audited node class with a constructor argument annotated with a
            node class that does not override
            :meth:`~cramph.node.StatechartNode.rebuilt_by`.
        """
        node_classes = self.node_classes()
        node_class_names = {node_class.__name__ for node_class in node_classes} | {
            StatechartNode.__name__
        }
        ignored = tuple(module.__name__ for module in self.ignored_modules)
        return [
            node_class
            for node_class in node_classes
            if not node_class.__module__.startswith(ignored)
            and not issubclass(node_class, TerminalNode)
            and node_class.rebuilt_by is StatechartNode.rebuilt_by
            and self._holds_nodes(node_class, node_class_names)
        ]

    @staticmethod
    def _holds_nodes(node_class: type, node_class_names: set[str]) -> bool:
        """
        :return: Whether a constructor argument of `node_class` is annotated with one of
            `node_class_names`.
        """
        return any(
            set(re.findall(r"\w+", str(dataclass_field.type))) & node_class_names
            for dataclass_field in fields(node_class)
            if dataclass_field.init
        )
