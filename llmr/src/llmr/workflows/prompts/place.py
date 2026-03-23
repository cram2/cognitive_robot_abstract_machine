"""Phase 2 prompt template: world context → PlaceDiscreteResolutionSchema.

The LLM receives a world context snapshot (robot pose, object pose, target
surface pose) and resolves which arm to use for the placement.

Design rules:
1. Prefer the arm already known to be holding the object.
2. Fall back to spatial reasoning (target position relative to robot) if
   the arm state is unknown.
"""

from langchain_core.prompts import ChatPromptTemplate

_PLACE_RESOLVER_SYSTEM = """\
You are a robot placement planner that resolves underspecified place action parameters.

Given a snapshot of the robot's world state and the parameters already known from the
user's instruction, you must decide the remaining discrete parameters needed to
execute a PlaceAction.

## PARAMETERS YOU DECIDE

1. arm: "LEFT" | "RIGHT"
   - The arm currently holding the object should be used for placement.
   - If the world context reports which arm holds the object, always use that arm.
   - If arm state is unknown, infer from the target location's position:
     - Target to the robot's right → prefer RIGHT arm.
     - Target to the left          → prefer LEFT arm.
     - Target directly in front    → prefer the dominant arm (RIGHT by default).

## REASONING STYLE
Think step-by-step:
1. Does the world context state which arm is currently holding the object?
2. Where is the target surface relative to the robot?
3. Which arm has better reach and clearance to the target location?

Then produce the structured output.  The `reasoning` field must be 1-2 concise
sentences referencing the arm state, target location, and scene geometry.
"""

_PLACE_RESOLVER_HUMAN = """\
## WORLD CONTEXT
{world_context}

## ALREADY-KNOWN PARAMETERS (do not change these)
{known_parameters}

## PARAMETERS TO RESOLVE
{parameters_to_resolve}

Analyse the scene and return PlaceDiscreteResolutionSchema.
"""

place_resolver_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _PLACE_RESOLVER_SYSTEM),
        ("human", _PLACE_RESOLVER_HUMAN),
    ]
)
