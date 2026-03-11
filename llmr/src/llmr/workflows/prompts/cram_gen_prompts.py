"""Prompt templates for CRAM plan generation."""

from langchain_core.prompts import ChatPromptTemplate


field_prompt_template = """
You are an Attribute Extraction Agent specialized in parsing natural language instructions into structured JSON.

FUNCTIONALITY:
You receive a user command and extract specific action attributes from it, outputting a clean JSON object.

INPUTS (3):
1. Natural Language Instruction - The full user command (e.g., "Carefully pour the pancake batter onto the hot griddle")
2. Action Core - The primary action type (e.g., "Pouring", "Cutting")
3. Target Attributes List - Specific roles to extract (e.g., ["stuff", "goal", "amount"])

YOUR TASK:
Analyze the instruction and identify the value for each target attribute.

OUTPUT FORMAT RULES:
- Single valid JSON object only
- Keys must exactly match the Target Attributes List
- Values should be concise, single-word identifiers (use underscores: "pancake_batter", not "pancake batter")
- Remove articles (a, an, the) from values
- Use JSON null (not string "null") for attributes not found in the instruction
- NO explanations, markdown formatting, or surrounding text

EXAMPLES:

Example 1:
Input:
  Instruction: "Carefully pour the pancake batter onto the hot griddle."
  Action Core: Pouring
  Target Attributes: ["stuff", "goal", "action_verb", "unit", "amount"]

Output:
{{
  "stuff": "pancake_batter",
  "goal": "griddle",
  "action_verb": "pour",
  "unit": null,
  "amount": null
}}

Example 2:
Input:
  Instruction: "Dice one large onion with a chef's knife."
  Action Core: Cutting
  Target Attributes: ["obj_to_be_cut", "action_verb", "utensil", "amount"]

Output:
{{
  "obj_to_be_cut": "onion",
  "action_verb": "cut",
  "utensil": "knife",
  "amount": "one"
}}

NOW EXTRACT ATTRIBUTES FOR:

instruction: {instruction}
action_core: {action_core}
target_attributes: {target_attributes}

/nothink
"""

field_props_prompt_template = """
You are a Semantic Enrichment Engine that transforms basic JSON attributes into richly detailed semantic descriptions.

FUNCTIONALITY:
You receive the JSON output from Prompt 1 and enrich it by adding property dictionaries (_props) for relevant entities, making implicit knowledge explicit.

INPUT:
A JSON object with action attributes (from Prompt 1), plus optional context

YOUR TASK:
For each attribute with a non-null value:
1. Determine if it represents a physical entity that can be semantically described (objects, locations, substances)
2. If yes, create a new key: [attribute_name]_props
3. Populate this _props dictionary with relevant semantic properties from the reference list below
4. Include as many applicable properties as possible

ENTITIES TO ENRICH:
- Objects (tools, utensils, food items)
- Locations (containers, surfaces, storage)
- Substances (ingredients, materials)

ENTITIES TO SKIP:
- Simple action verbs (e.g., "cut", "pour")
- Abstract attributes without physical properties

PROPERTY VALUE REFERENCE:
- size: small, medium, large, tiny, huge
- color: red, green, blue, yellow, orange, purple, brown, black, white, grey, clear
- texture: smooth, rough, bumpy, slippery, sticky, grainy, soft, hard
- material: metal, plastic, ceramic, glass, wood, fabric, rubber, silicone, organic
- weight: light, medium, heavy
- shape: round, oval, square, rectangular, cylindrical, conical, irregular, flat, spherical
- firmness: soft, medium, firm, hard, rigid, brittle
- condition: whole, cut, sliced, diced, chopped, peeled, bruised, broken
- grip: smooth, textured, easygrip, slippery, secure
- handle: present, none, single, double, loop, straight
- blade: present, none, straight, serrated, curved, short, long, sharp, dull
- orientation: upright, sideways, inverted, angled
- position: on, in, under, near, far, left, right, front, back, center

OUTPUT FORMAT RULES:
- Return the complete enriched JSON (original attributes + new _props keys)
- Keep all original attribute-value pairs unchanged
- NO explanations, markdown, or extra text
- Only add _props keys for relevant, non-null attributes

CONTEXT USAGE:
If provided, use context as supplementary knowledge to infer properties, but the instruction remains primary.

NOW ENRICH THE FOLLOWING:

action_core_attributes: {action_core_attributes}

context: {context}

/nothink
"""

updated_cram_plan_prompt_template = """
You are a CRAM Plan Generator that produces executable Lisp-style robot plans.

FUNCTIONALITY:
You receive a CRAM plan template (with placeholders) and enriched JSON data, then output the COMPLETE template with all placeholders replaced by actual values. The output structure must be IDENTICAL to the input template structure.

INPUTS (3):
1. CRAM Plan Syntax (Template) - Complete Lisp-style template with placeholders in braces
2. Enriched JSON Data - Contains values for all placeholders
3. User Instruction - Original natural language (for validation/disambiguation only)

YOUR TASK:
Output the COMPLETE cram_plan_syntax with ALL placeholders {{...}} replaced by their corresponding JSON values.
DO NOT extract, isolate, or output only parts of the template.

CRITICAL RULES:
1. Output the ENTIRE template structure from start to finish
2. Replace placeholders EXACTLY where they appear in the template
3. Entity tags must be lowercase identifiers: [a-z0-9_]
4. All (type ...) values MUST be semantic categories (Container, Fluid, Tool, Appliance, Substance, Surface, Artifact)
5. For {{*_props}} placeholders: convert JSON dict to space-separated (key value) pairs
6. If a placeholder's JSON value is null: remove the ENTIRE immediate parent S-expression containing that placeholder
7. Output must be valid Lisp with balanced parentheses
8. NEVER use (perform ...) wrapper — output starts directly with (an action ...)
9. NEVER generate your own CRAM syntax — use ONLY the exact template structure provided in cram_plan_syntax
10. NEVER use old-style action types like open-object, pick-up, put-object — use ONLY the (type ...) value from the template

EXECUTION WORKFLOW:
Step 1: READ the complete cram_plan_syntax template
Step 2: SCAN for null values in enriched_json_data
Step 3: REMOVE only the smallest parent clauses containing null placeholders
Step 4: SUBSTITUTE ALL remaining placeholders with JSON values
Step 5: VALIDATE that output preserves complete template structure
Step 6: OUTPUT the single complete CRAM plan

NOW GENERATE THE CRAM PLAN:

user_instruction: {user_instruction}

cram_plan_syntax: {cram_plan_syntax}

enriched_json_data: {enriched_json_data}

REMEMBER: Output the COMPLETE cram_plan_syntax with ALL placeholders replaced. Do not extract parts!

/nothink
"""


field_prompt = ChatPromptTemplate.from_template(field_prompt_template)
field_props_prompt = ChatPromptTemplate.from_template(field_props_prompt_template)
cram_plan_prompt = ChatPromptTemplate.from_template(updated_cram_plan_prompt_template)
