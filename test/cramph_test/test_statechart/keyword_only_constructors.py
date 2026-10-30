from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, fields, is_dataclass
from types import ModuleType

from typing_extensions import Iterator

from cramph.node import StatechartNode

# %% constructor audit


@dataclass
class PositionalConstructorParameters:
    """
    A statechart node class whose constructor still accepts positional arguments.
    """

    node_class: type[StatechartNode]
    """
    The node class whose constructor is audited.
    """

    parameter_names: list[str]
    """
    The constructor parameters of :attr:`node_class` that are not keyword-only.
    """


@dataclass
class KeywordOnlyConstructorAudit:
    """
    Finds the statechart node classes of some modules whose constructors accept
    positional arguments.
    """

    modules: list[ModuleType]
    """
    The modules and packages whose node classes are audited, each package with every
    module it contains.
    """

    def nodes_with_positional_parameters(self) -> list[PositionalConstructorParameters]:
        """
        :return: Every dataclass node class defined in :attr:`modules` with at least one
            constructor parameter that is not keyword-only.
        """
        self._import_every_module()
        module_names = tuple(module.__name__ for module in self.modules)
        return [
            PositionalConstructorParameters(node_class, parameter_names)
            for node_class in dict.fromkeys(self._subclasses(StatechartNode))
            if node_class.__module__.startswith(module_names)
            and is_dataclass(node_class)
            and (parameter_names := self._positional_parameter_names(node_class))
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

    @staticmethod
    def _positional_parameter_names(node_class: type) -> list[str]:
        """
        :return: The constructor parameters of `node_class` that are not keyword-only.
        """
        return [
            dataclass_field.name
            for dataclass_field in fields(node_class)
            if dataclass_field.init and not dataclass_field.kw_only
        ]
