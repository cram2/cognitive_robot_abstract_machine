"""Intent and entity parsing agent using an LLM with optional reflection."""

from __future__ import annotations

import uuid
from typing_extensions import Any, Dict

from ..llm_configuration import LLMFactory, default_llm
from ..prompts.intent_entity_prompts import DETAILED_PARSING_PROMPT
from ..models.intent_entity_models import (
    Instruction,
    InstructionList,
    IntentType,
    Metadata,
)

_INTENT_REQUIRED_ROLES: Dict[IntentType, list[str]] = {
    IntentType.POUR: ["patient", "destination_location"],
    IntentType.CUT: ["patient"],
    IntentType.PICK: ["patient"],
    IntentType.PLACE: ["patient", "destination_location"],
    IntentType.OPEN: ["patient"],
    IntentType.PULL: ["patient"],
}


class ReflectiveParser:
    """Parses natural language instructions into structured symbolic intents.

    Supports an optional reflection pass that identifies missing roles and,
    when *enable_reflection* is True, sends the feedback back to the LLM for
    a second refinement pass.
    """

    def __init__(self, model_name: str = "", enable_reflection: bool = False) -> None:
        self.llm = self._build_llm(model_name)
        self.structured_output_llm = self.llm.with_structured_output(
            InstructionList, method="json_schema"
        )
        self.enable_reflection = enable_reflection

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _build_llm(model_name: str) -> Any:
        """Select the appropriate LLM client based on *model_name*."""
        if "qwen3" in model_name:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model_name, temperature=0.1)
        if "gpt" in model_name:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        if "claude" in model_name:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model="claude-sonnet-4-5", temperature=0.1)
        # Default: Ollama with reasoning enabled
        from langchain_ollama import ChatOllama
        return ChatOllama(model="qwen3:14b", temperature=0.1, reasoning=True)

    @staticmethod
    def _assign_action_id(instruction: Instruction) -> Instruction:
        """Ensure every instruction has a stable, unique action_id."""
        if not instruction.action_id or instruction.action_id.startswith("A"):
            instruction.action_id = f"action_{uuid.uuid4().hex[:8]}"
        return instruction

    def _check_missing_roles(self, instruction: Instruction) -> list[str]:
        """Return role names required by this intent that are currently None."""
        required = _INTENT_REQUIRED_ROLES.get(instruction.intent, [])
        return [r for r in required if getattr(instruction.roles, r) is None]

    def _append_reflection_feedback(
        self, instruction: Instruction, feedback: list[str]
    ) -> Instruction:
        """Attach reflection feedback as a comment on the instruction metadata."""
        if not instruction.metadata:
            instruction.metadata = Metadata()
        existing = instruction.metadata.comments or ""
        instruction.metadata.comments = (
            existing + " | REFLECTION FEEDBACK: " + "; ".join(feedback)
        )
        return instruction

    # ── Core pipeline steps ────────────────────────────────────────────────────

    def parse_initial(self, instruction: str) -> InstructionList:
        """First-pass LLM parsing of *instruction*."""
        chain = DETAILED_PARSING_PROMPT | self.structured_output_llm
        try:
            result: InstructionList = chain.invoke({"original_instruction": instruction})
        except Exception as exc:
            raise RuntimeError(f"Parsing failed: {exc}") from exc

        validated = [self._assign_action_id(inst) for inst in result.instructions]
        return InstructionList(instructions=validated)

    def reflect(self, instructions: InstructionList) -> InstructionList:
        """Annotate instructions with missing-role and confidence feedback."""
        reflected: list[Instruction] = []
        for inst in instructions.instructions:
            feedback: list[str] = []

            missing = self._check_missing_roles(inst)
            if missing:
                feedback.append(f"Missing required roles: {missing}")

            if (
                inst.metadata
                and inst.metadata.confidence is not None
                and not (0.0 <= inst.metadata.confidence <= 1.0)
            ):
                feedback.append(
                    f"Confidence {inst.metadata.confidence} out of range [0.0, 1.0]"
                )

            if feedback:
                inst = self._append_reflection_feedback(inst, feedback)

            reflected.append(inst)
        return InstructionList(instructions=reflected)

    def reiterate(
        self, original_instruction: str, instructions: InstructionList
    ) -> InstructionList:
        """Use the LLM to refine instructions based on reflection feedback."""
        instructions_json = [inst.model_dump() for inst in instructions.instructions]
        feedback_prompt = (
            f"The following instructions were parsed from the input:\n{instructions_json}\n\n"
            "The reflection feedback points out issues or missing information.\n"
            "Please revise and return corrected instructions in the same JSON schema, "
            "fixing all issues mentioned.\n\n"
            f'Original Instruction: "{original_instruction}"'
        )
        result: InstructionList = self.structured_output_llm.invoke(feedback_prompt)
        return InstructionList(instructions=result.instructions)

    def parse(self, instruction: str) -> Dict[str, Any]:
        """Run the full parse -> reflect -> (optionally reiterate) pipeline.

        Returns:
            A dict with an "instructions" key (list of serialised instruction dicts)
            and an optional "error" key if parsing failed.
        """
        try:
            parsed = self.parse_initial(instruction)
            parsed = self.reflect(parsed)
            if self.enable_reflection:
                parsed = self.reiterate(instruction, parsed)
            return {"instructions": [inst.model_dump() for inst in parsed.instructions]}
        except Exception as exc:
            return {"instructions": [], "error": str(exc)}


if __name__ == "__main__":
    parser = ReflectiveParser()
    tests = ["pour water from the jug into the glass"]
    for test in tests:
        print("=" * 50)
        print(f"Instruction: {test}")
        result = parser.parse(test)
        print(result)
