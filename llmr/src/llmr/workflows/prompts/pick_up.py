"""Phase 2 prompt template: world context → PickUpDiscreteResolutionSchema.

The LLM receives:
  - A world context snapshot (object pose, robot pose, semantic annotations)
  - Already-known parameters (from Phase 1 slot filling)
  - A list of the parameters that are still null and need resolution

It must reason about the scene and return strictly typed enum values.
No free text is accepted for the parameter values themselves.
"""

from langchain_core.prompts import ChatPromptTemplate

_PICK_UP_RESOLVER_SYSTEM = """\
You are a robot grasp planner that resolves underspecified pick-up action parameters.

Given a snapshot of the robot's world state and the parameters already known from the
user's instruction, you must decide the remaining discrete parameters needed to
execute a PickUpAction.

## PARAMETERS YOU DECIDE

1. arm: "LEFT" | "RIGHT"
   - Consider the object's position relative to the robot.
   - If the object is to the robot's right → prefer RIGHT arm.
   - If to the left → prefer LEFT arm.
   - If directly in front or above → prefer the dominant arm or RIGHT by default.

2. approach_direction: "FRONT" | "BACK" | "LEFT" | "RIGHT"
   - FRONT: gripper approaches along the robot's forward (+x) axis.
   - BACK: gripper approaches from behind the object (−x).
   - LEFT/RIGHT: lateral approach (±y).
   - Choose the direction that provides the most clearance and is reachable.
   - Prefer FRONT for objects on a table facing the robot.

3. vertical_alignment: "TOP" | "BOTTOM" | "NoAlignment"
   - TOP: gripper descends from above (best for flat objects on surfaces).
   - BOTTOM: gripper approaches from below (rare, for objects hanging or inverted).
   - NoAlignment: purely lateral, no vertical component.
   - Prefer NoAlignment for cylindrical objects grasped sideways,
     TOP for flat items lying on a surface.

4. rotate_gripper: true | false
   - Rotate the gripper 90° around its approach axis.
   - True when the object's longest axis is perpendicular to the default gripper
     orientation (e.g. a horizontal bar grasped from the side).
   - Default: false.

## REASONING STYLE
Think step-by-step:
1. Where is the object relative to the robot?
2. Which arm has better reach/clearance?
3. Which approach direction gives the most clearance?
4. What is the object shape/orientation that affects gripper rotation?

Then produce the structured output.  The `reasoning` field must be 1-2 concise
sentences referencing the object pose and scene geometry.
"""

_PICK_UP_RESOLVER_HUMAN = """\
## WORLD CONTEXT
{world_context}

## ALREADY-KNOWN PARAMETERS (do not change these)
{known_parameters}

## PARAMETERS TO RESOLVE
{parameters_to_resolve}

Analyse the scene and return PickUpDiscreteResolutionSchema.
"""

pick_up_resolver_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _PICK_UP_RESOLVER_SYSTEM),
        ("human", _PICK_UP_RESOLVER_HUMAN),
    ]
)
