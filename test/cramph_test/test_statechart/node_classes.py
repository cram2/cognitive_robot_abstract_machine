from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, is_dataclass
from types import ModuleType

from typing_extensions import Iterator

from cramph.node import StatechartNode

# %% the node classes of some modules


@dataclass
class NodeClassesOfModules:
    """
    The dataclass statechart node classes some modules define.
    """

    modules: list[ModuleType]
    """
    The modules and packages whose node classes are found, each package with every
    module it contains.
    """

    def node_classes(self) -> list[type[StatechartNode]]:
        """
        :return: Every dataclass node class defined in :attr:`modules`, once.
        """
        self._import_every_module()
        module_names = tuple(module.__name__ for module in self.modules)
        return [
            node_class
            for node_class in dict.fromkeys(self._subclasses(StatechartNode))
            if node_class.__module__.startswith(module_names)
            and is_dataclass(node_class)
        ]

    def _import_every_module(self) -> None:
        """
        Imports every module of the packages in :attr:`modules`, so their node classes
        are defined.
        """
        for module in self.modules:
            package_path = module.__spec__.submodule_search_locations
            if package_path is None:
                continue
            for module_info in pkgutil.walk_packages(
                package_path, f"{module.__name__}."
            ):
                importlib.import_module(module_info.name)

    def _subclasses(self, node_class: type[StatechartNode]) -> Iterator[type]:
        """
        :return: Every direct and indirect subclass of `node_class`, once per path it
            inherits from `node_class` along.
        """
        for subclass in node_class.__subclasses__():
            yield subclass
            yield from self._subclasses(subclass)
