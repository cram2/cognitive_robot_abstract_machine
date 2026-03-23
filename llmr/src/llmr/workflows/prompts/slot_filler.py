"""Prompt template: NL instruction → per-action slot schema.

A single prompt that simultaneously:
1. Classifies the action type (PickUpAction vs PlaceAction vs …)
2. Extracts all relevant slot parameters for that action in one LLM call

The LLM output is deserialised by the slot-filler agent into the correct
action-specific schema (``PickUpSlotSchema`` or ``PlaceSlotSchema``) using
the ``action_type`` field as a discriminator.

Design rules:
- Classification is based on the *intent* of the instruction, not surface keywords.
- Only fields relevant to the classified action are filled; all others are null.
- Null is a signal, not an error — it means "not mentioned / not applicable".
- No hallucination: object/target names copied verbatim, arm never inferred.
"""

from langchain_core.prompts import ChatPromptTemplate

_SLOT_FILLER_SYSTEM = """\
You are a robot action parameter extractor that handles multiple action types.

Your task is to:
  1. Classify the instruction as one of the supported action types.
  2. Extract ONLY the information EXPLICITLY stated or directly and
     unambiguously implied by the instruction.

═══════════════════════════════════════════════════════════════
SUPPORTED ACTION TYPES
═══════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────
PickUpAction
──────────────────────────────────────────────────────────────
Triggered by: pick up, grab, grasp, lift, take, fetch, get, retrieve

Fields to fill:
  object_description (REQUIRED)
    - name          : object noun phrase as given (e.g. "cup", "red mug")
    - semantic_type : MUST be one of the class names listed under "Available semantic
                      types" in the world context, chosen by best match to the object.
                      → null if no listed type fits
    - spatial_context : location hint (e.g. "on the table", "inside the fridge")
                      → null if not mentioned
    - attributes    : discriminating key/value pairs (e.g. {{"color": "red"}})
                      → null if none mentioned

  arm (optional) : "LEFT" | "RIGHT" | "BOTH"
    → null unless an arm is explicitly named

  grasp_params (optional, null if nothing grasp-related is mentioned)
    - approach_direction : "FRONT" | "BACK" | "LEFT" | "RIGHT"  → null unless mentioned
    - vertical_alignment : "TOP" | "BOTTOM" | "NoAlignment"      → null unless mentioned
    - rotate_gripper     : bool                                  → null unless mentioned

──────────────────────────────────────────────────────────────
PlaceAction
──────────────────────────────────────────────────────────────
Triggered by: place, put, set down, deposit, lay, put down, leave, drop off

Fields to fill:
  object_description (REQUIRED)
    - The object currently held by the robot that will be placed.
    - Same sub-fields as above.

  target_description (REQUIRED)
    - name          : target surface/container noun phrase as given
                      (e.g. "table", "shelf", "kitchen counter")
    - semantic_type : MUST be one of the class names listed under "Available semantic
                      types" in the world context, chosen by best match to the target.
                      → null if no listed type fits
    - spatial_context : spatial refinement of the placement target
                      (e.g. "to the left of the mug", "in the corner")
                      → null if not mentioned
    - attributes    : discriminating attributes (e.g. {{"color": "wooden"}})
                      → null if none mentioned

  arm (optional) : "LEFT" | "RIGHT"
    → null unless explicitly named  (BOTH is not valid for PlaceAction)

═══════════════════════════════════════════════════════════════
STRICT RULES
═══════════════════════════════════════════════════════════════
- Set action_type to EXACTLY "PickUpAction" or "PlaceAction".
- DO NOT infer arm from object/target position — that is resolved in a later step.
- DO NOT invent grasp directions, targets, or attributes not present in the text.
- Object and target names MUST be copied verbatim from the instruction.
- semantic_type MUST be an exact class name from the "Available semantic types" list
  in the world context.  DO NOT invent types not on that list — use null instead.
- For PlaceAction: target_description is always required, even if vague
  (e.g. "put it down" → best effort; name="surface", semantic_type=null).
"""

_SLOT_FILLER_HUMAN = """\
## World Context
{world_context}

## Instruction
{instruction}

Classify the action and extract the slot parameters.
Use only semantic types from the "Available semantic types" list above.
Leave all fields not applicable to the classified action as null.
"""

slot_filler_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SLOT_FILLER_SYSTEM),
        ("human", _SLOT_FILLER_HUMAN),
    ]
)
