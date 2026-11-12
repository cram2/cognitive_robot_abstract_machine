from __future__ import annotations

from dataclasses import dataclass, fields

from cramph.node import StatechartNode

from .node_classes import NodeClassesOfModules

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
class KeywordOnlyConstructorAudit(NodeClassesOfModules):
    """
    Finds the statechart node classes of some modules whose constructors accept
    positional arguments.
    """

    def nodes_with_positional_parameters(self) -> list[PositionalConstructorParameters]:
        """
        :return: Every dataclass node class defined in :attr:`modules` with at least one
            constructor parameter that is not keyword-only.
        """
        return [
            PositionalConstructorParameters(node_class, parameter_names)
            for node_class in self.node_classes()
            if (parameter_names := self._positional_parameter_names(node_class))
        ]

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
