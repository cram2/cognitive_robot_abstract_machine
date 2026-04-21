"""
Structured exceptions for llmr.

All exceptions inherit from :class:`krrood.utils.DataclassException`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import List, Any

from krrood.utils import DataclassException


@dataclass
class LLMProviderNotSupported(DataclassException):
    """Raised when an unknown or unsupported LLM provider is passed to ``make_llm()``."""

    provider: Any
    """The unsupported provider value that was passed."""

    valid_providers: List[str]
    """List of valid provider names."""

    def __post_init__(self):
        self.message = (
            f"Unknown LLM provider: {self.provider!r}. "
            f"Valid options: {self.valid_providers}."
        )
        super().__post_init__()


@dataclass
class LLMActionClassificationFailed(DataclassException):
    """Raised when the LLM cannot map a natural-language instruction to a known action class."""

    instruction: str
    """The NL instruction that could not be classified."""

    def __post_init__(self):
        self.message = (
            f"Could not classify an action type from instruction: {self.instruction!r}. "
            "Check that pycram action classes are importable and the LLM is reachable."
        )
        super().__post_init__()


@dataclass
class LLMSlotFillingFailed(DataclassException):
    """Raised when the LLM slot filler returns no output for a given action."""

    action_name: str
    """Name of the action class whose slots could not be filled."""

    def __post_init__(self):
        self.message = (
            f"LLM slot filler returned no output for {self.action_name!r}. "
            "Check LLM connectivity and the action's field schema."
        )
        super().__post_init__()


@dataclass
class LLMUnresolvedRequiredFields(DataclassException):
    """Raised when strict_required=True and one or more required action fields remain unresolved."""

    action_name: str
    """Name of the action class with unresolved fields."""

    unresolved_fields: List[str]
    """Names of the required fields that could not be resolved."""

    def __post_init__(self):
        fields_str = ", ".join(self.unresolved_fields)
        self.message = (
            f"LLMBackend could not resolve required field(s) for {self.action_name!r}: "
            f"{fields_str}."
        )
        super().__post_init__()


@dataclass
class LLMActionRegistryEmpty(DataclassException):
    """Raised when action discovery yields no registered PyCRAM actions."""

    def __post_init__(self):
        self.message = (
            "No PyCRAM actions found in registry. "
            "Check that action packages are properly imported and registered."
        )
        super().__post_init__()
