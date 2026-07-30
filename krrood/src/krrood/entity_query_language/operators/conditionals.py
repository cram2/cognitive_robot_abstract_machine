"""
Conditional EQL operator constructs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from krrood.entity_query_language.core.base_expressions import (
    Selectable,
    SymbolicExpression,
)


@dataclass(eq=False)
class CaseWhen(Selectable):
    """
    Represents a conditional expression: CASE WHEN condition THEN value ELSE else_value END.

    Supports both local Python evaluation and SQL translation via the EQL translator.

    .. code-block:: python

        action = variable(MoveAction, domain=[])
        query = an(set_of(
            min(case_when(action.polymorphic_type == 'PickUpActionDAO', action.database_id))
        ))
    """

    condition: SymbolicExpression
    """
    The condition to evaluate.
    """

    then_value: SymbolicExpression
    """
    The value returned if the condition is true.
    """

    else_value: Optional[SymbolicExpression] = None
    """
    The value returned if the condition is false.

    Defaults to None.
    """

    def __post_init__(self):
        self._type_ = getattr(self.then_value, "_type_", None)
        # Explicitly register children — _update_children_ converts non-SymbolicExpression
        # values (like plain Python ints/strings) to Literal nodes automatically
        if self.else_value is not None:
            self.condition, self.then_value, self.else_value = self._update_children_(
                self.condition, self.then_value, self.else_value
            )
        else:
            self.condition, self.then_value = self._update_children_(
                self.condition, self.then_value
            )
        super().__post_init__()

    def _replace_child_field_(self, old: Any, new: Any) -> None:
        """
        Replace a child expression node during EQL tree manipulation.
        """
        if self.condition is old:
            self.condition = new
        elif self.then_value is old:
            self.then_value = new
        elif self.else_value is old:
            self.else_value = new
        else:
            raise ValueError(
                f"Child {old} not found in CaseWhen — "
                f"expected one of: condition, then_value, else_value"
            )

    def _name_(self) -> str:
        """
        Return the symbolic name of this expression node.
        """
        return "case_when"

    def _evaluate__(self, sources: Any) -> Any:
        """
        Evaluate the condition locally in Python.

        :param sources: The variable bindings for evaluation
        :return: then_value if condition is true, else_value otherwise
        """
        cond_result = self.condition._evaluate__(sources)
        is_true = (
            bool(cond_result)
            if not isinstance(cond_result, list)
            else len(cond_result) > 0
        )

        if is_true:
            return self.then_value._evaluate__(sources)
        if self.else_value is not None:
            return self.else_value._evaluate__(sources)
        return None
