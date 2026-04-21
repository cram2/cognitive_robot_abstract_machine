"""Resolution layer — turn LLM output into concrete Python values.

- :mod:`llmr.resolution.grounder` maps an :class:`EntityDescriptionSchema`
  to a KRROOD :class:`Symbol` instance.
- :mod:`llmr.resolution.slot_resolution` dispatches a :class:`SlotValue`
  to a Python value by field kind (ENTITY/POSE/ENUM/PRIMITIVE/TYPE_REF),
  delegating entity-like kinds to the grounder.
"""

from llmr.resolution.grounder import EntityGrounder, GroundingResult
from llmr.resolution.slot_resolution import (
    coerce_enum,
    coerce_primitive,
    resolve_binding_value,
    resolve_entity_slot,
)

__all__ = [
    "EntityGrounder",
    "GroundingResult",
    "coerce_enum",
    "coerce_primitive",
    "resolve_binding_value",
    "resolve_entity_slot",
]
