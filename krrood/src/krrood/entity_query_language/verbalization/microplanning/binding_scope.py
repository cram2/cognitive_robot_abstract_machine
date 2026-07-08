from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression
    from krrood.entity_query_language.verbalization.fragments.base import (
        VerbalizationFragment,
    )


@dataclass
class BindingScope:
    """
    The deferred-constraint frame stack and the field-reference override map used when
    verbalizing an instantiated variable.

    An instantiated variable such as ``inference(Drawer)(container=fc.parent)`` is rendered as
    *"a Drawer where the container of the Drawer is …"*: the field bindings are rendered first,
    any constraints on those fields are deferred into a frame, and the pre-rendered field
    fragments are registered as overrides so the deferred constraints reuse them instead of
    re-verbalizing.
    """

    constraint_frames: List[List[SymbolicExpression]] = field(default_factory=list)
    """Stack of deferred-expression frames.  Each frame belongs to one nesting
    level of InstantiatedVariable verbalization."""

    binding_overrides: Dict[uuid.UUID, VerbalizationFragment] = field(
        default_factory=dict
    )
    """Maps a child expression's ``_id_`` → a ``VerbalizationFragment`` that substitutes for
    it on subsequent encounters, so a pre-rendered field reference is reused rather
    than re-verbalized."""

    def push_constraint_frame(self) -> None:
        """Open a new constraint frame for the current InstantiatedVariable."""
        self.constraint_frames.append([])

    def pop_constraint_frame(self) -> List[SymbolicExpression]:
        """
        Close the current frame and return its deferred expressions (empty when none open).

        :return: Deferred expressions from the closed frame, in deferral order.
        """
        return self.constraint_frames.pop() if self.constraint_frames else []

    def defer_constraint(self, expression: SymbolicExpression) -> None:
        """
        Defer *expression* into the top constraint frame; a no-op when no frame is open.

        :param expression: EQL expression to defer.
        """
        if self.constraint_frames:
            self.constraint_frames[-1].append(expression)
