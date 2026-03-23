"""Pipeline subpackage: NL instruction → typed pycram action.

Covers the full translation layer:
  - Slot filling (LLM classifies action type + extracts parameters)
  - Entity grounding (LLM entity description → world Body)
  - Action dispatch (typed schema → concrete pycram action)
"""

from llmr.pipeline.action_pipeline import ActionPipeline
from llmr.pipeline.action_dispatcher import ActionDispatcher, ActionHandler, WorldContext
from llmr.pipeline.entity_grounder import EntityGrounder, GroundingResult, ground_entity

__all__ = [
    "ActionPipeline",
    "ActionDispatcher",
    "ActionHandler",
    "WorldContext",
    "EntityGrounder",
    "GroundingResult",
    "ground_entity",
]
