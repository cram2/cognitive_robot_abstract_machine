"""Prompt templates for PyCRAM action designator generation."""

from langchain_core.prompts import ChatPromptTemplate


entity_mapping_prompt_template = """
You are a CRAM Plan Entity Mapper that grounds abstract entities to concrete belief state entities.

TASK: Match tagged entities in CRAM plans to belief state entities and replace:
1. Entity name after `:tag`
2. Type value after `type`

TWO-TIER MATCHING SYSTEM (FOLLOW IN ORDER):

### TIER 1: Semantic Annotations (CHECK FIRST - PRIORITY)
Search in Semantic Annotations list first.

**Match against:**
- `name` field (e.g., 'Milk_1', 'left_gripper_finger')
- `body['name']` or `root['name']` (e.g., 'milk.stl', 'l_gripper_r_finger_link')
- Semantic type similarity + synonyms

**If matched:**
- Replace `:tag <name>` -> `:tag <body_name_or_root_name>`
- Replace `type <old>` -> `type <semantic_type>`

### TIER 2: Bodies List (FALLBACK - USE ONLY IF NOT IN TIER 1)
If not found in Semantic Annotations, search Bodies list.

**If matched:**
- Replace `:tag <name>` -> `:tag <body_name>`
- Replace `type <old>` -> `type Body`

CRITICAL RULES:
- WHAT TO CHANGE: Entity name after `:tag`, type value after `type`
- WHAT NEVER CHANGES: `:tag` keyword itself, `type` keyword itself, ALL other attributes, ALL structure and formatting

OUTPUT REQUIREMENTS:
- OUTPUT ONLY A JSON ARRAY - NO OTHER TEXT
- NO explanations, NO analysis, NO commentary

**Required format:**
```json
["<grounded_plan_1>", "<grounded_plan_2>", ...]
```

NOW PROCESS YOUR INPUT:

**Instructions:**
{atomic_instructions}

**CRAM Plans:**
{cram_plans}

**Belief State:**
{belief_state_context}

RESPOND WITH ONLY THE JSON ARRAY - NOTHING ELSE
"""


updated_model_selector_prompt_template = """
You are an intelligent robotic action classifier.

Your task is to read symbolic instruction representations and select the most relevant robot action model(s) from the available list.

Available Action Models:
- PickUpActionDescription: The robot picks up an object.
- PlaceActionDescription: The robot places an object at a target position using an arm.
- NavigateAction: The robot navigates to a specified position.
- ParkArmsAction: The robot parks its arms.
- TransportActionDescription: The robot picks up an object and transports it to a destination using an arm.

CRITICAL: Counting Actions vs. Entities
- ONE CRAM PLAN = ONE ACTION = ONE ACTION MODEL
- Count the number of `(an action ...)` blocks in the CRAM plans
- Each `(an action ...)` block represents ONE action, even if it references multiple entities

Intent -> Action Model Mapping Guidelines:
- PICK_UP / PickingUp -> PickUpActionDescription
- PLACE / Placing -> PlaceActionDescription
- NAVIGATE / MOVING -> NavigateAction
- TRANSPORT / TRANSPORTING -> TransportActionDescription

Reasoning Procedure:
1. Count the number of `(an action ...)` blocks in the CRAM plans
2. For each action block, identify its type
3. Check for action combinations (PickingUp + Placing of same object -> TransportActionDescription)
4. Return a list with the same number of elements as action blocks

Given the symbolic context: {symbolic_context}

Generate and return the appropriate list of action model name(s).
"""


designator_prompt_template = """
You are a PyCRAM Action Instance Creator. Create executable action instances from CRAM plans.

OPTIMIZATION RULE:
If PYCRAM ACTIONS = ["PickUpActionDescription", "PlaceActionDescription"]
AND same `:tag` in both CRAM plans:
-> Create ONE TransportActionDescription instead

ACTION TYPE MAPPING (MUST MATCH EXACTLY):
"PickUpActionDescription" -> Create PickUpActionDescription instance
"PlaceActionDescription" -> Create PlaceActionDescription instance
"TransportActionDescription" -> Create TransportActionDescription instance
"NavigateActionDescription" -> Create NavigateActionDescription instance
"ParkArmsActionDescription" -> Create ParkArmsActionDescription instance

REQUIRED FIELDS BY ACTION TYPE:

### PickUpActionDescription:
- action_type: "PickUpAction"
- object_designator: <name: ":tag value", concept: "type value">
- arm: 0 (LEFT=0, RIGHT=1)
- grasp_description: <approach_direction: "top", rotate_gripper: false>

### PlaceActionDescription:
- action_type: "PlaceAction"
- object_designator: <name: ":tag value", concept: "type value">
- target_location: <position: [0.0, 0.0, 0.0]>
- arm: 0

### TransportActionDescription:
- action_type: "TransportAction"
- object_designator: <name: ":tag value", concept: "type value">
- target_location: <position: [0.0, 0.0, 0.0]>
- arm: 0

### NavigateActionDescription:
- action_type: "NavigateAction"
- target_location: <position: [0.0, 0.0, 0.0]>
- keep_joint_states: true

### ParkArmsActionDescription:
- action_type: "ParkArmsAction"
- arm: 2 (BOTH=2)

ATOMIC INSTRUCTIONS: {atomic_instructions}
GROUNDED CRAM PLANS: {grounded_cram_plans}
PYCRAM ACTIONS: {pycram_actions}

STEP 1: Check if optimization applies (PickUp+Place of same object)
STEP 2: For each action in PYCRAM ACTIONS (or optimized action), create instance with ALL required fields
STEP 3: Return models list

CRITICAL: Class name must EXACTLY match PYCRAM ACTIONS string
"""


entity_mapping_prompt = ChatPromptTemplate.from_template(entity_mapping_prompt_template)
updated_model_selector_prompt = ChatPromptTemplate.from_template(updated_model_selector_prompt_template)
designator_prompt = ChatPromptTemplate.from_template(designator_prompt_template)
