from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.variable import InstantiatedVariable
from krrood.entity_query_language.predicate import Verbalizable
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.grammar.framework.planner import Planner


@dataclass(frozen=True)
class BindingPlan:
    """One field binding of an InstantiatedVariable (number decided up front)."""

    field_name: str
    """The Python attribute name on the consequent type (e.g. ``"container"``)."""

    is_plural: bool
    """``True`` when *field_name* is plural."""

    value: SymbolicExpression
    """The EQL expression providing the field's value."""


@dataclass(frozen=True)
class InstantiatedPlan:
    """Complete decomposition of an InstantiatedVariable (the plan)."""

    type_name: str
    """Display name of the instantiated type (e.g. ``"Drawer"``)."""

    bindings: List[BindingPlan]
    """Ordered field bindings."""


@dataclass
class InstantiatedPlanner(Planner[InstantiatedVariable, InstantiatedPlan]):
    """
    Decompose an instantiated variable (a *"a TypeName where the field of the TypeName is …"*
    form) into an ``InstantiatedPlan``.

    It records the type name and, per field binding, the field name, grammatical number, and
    value expression — no fragments, no context, no recursion.

    Reference: :cite:t:`reiter2000building` — content/structure determination (microplanning).

    >>> connection = variable(FixedConnection, [])
    >>> InstantiatedPlanner(inference(Drawer)(container=connection.parent, handle=connection.child)).plan().type_name
    'Drawer'
    """

    def plan(self) -> InstantiatedPlan:
        """:return: The plan: the type name and its field bindings.

        >>> connection = variable(FixedConnection, [])
        >>> drawer = inference(Drawer)(container=connection.parent, handle=connection.child)
        >>> InstantiatedPlanner(drawer).plan().type_name
        'Drawer'
        """
        return InstantiatedPlan(type_name=self._type_name(), bindings=self._bindings())

    def _type_name(self) -> str:
        """:return: The display name of the instantiated type (*"Drawer"*).

        >>> connection = variable(FixedConnection, [])
        >>> InstantiatedPlanner(inference(Drawer)(container=connection.parent, handle=connection.child))._type_name()
        'Drawer'
        """
        return getattr(self.node._type_, "__name__", str(self.node._type_))

    def _bindings(self) -> List[BindingPlan]:
        """:return: One :class:`BindingPlan` per field binding, in construction order.

        >>> connection = variable(FixedConnection, [])
        >>> bindings = InstantiatedPlanner(inference(Drawer)(container=connection.parent, handle=connection.child))._bindings()
        >>> [binding.field_name for binding in bindings]
        ['container', 'handle']
        """
        return [
            BindingPlan(
                field_name=field_name,
                is_plural=morphology.is_plural(field_name),
                value=child,
            )
            for field_name, child in self.node._child_vars_.items()
        ]

    @staticmethod
    def has_fragment(node: InstantiatedVariable) -> bool:
        """
        :param node: The instantiated variable.
        :return: ``True`` when *node*'s type implements ``Verbalizable`` and overrides
            ``_verbalization_fragment_`` with its own structured surface.

        >>> connection = variable(FixedConnection, [])
        >>> InstantiatedPlanner.has_fragment(inference(Drawer)(container=connection.parent, handle=connection.child))
        False
        """
        type_ = node._type_
        return (
            isinstance(type_, type)
            and issubclass(type_, Verbalizable)
            and type_._verbalization_fragment_.__func__
            is not Verbalizable._verbalization_fragment_.__func__
        )
