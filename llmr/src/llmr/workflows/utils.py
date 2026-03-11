"""Utility functions for the llmr workflow."""

import json
import re
from pathlib import Path


def read_json_from_file(filepath: str | Path) -> dict:
    """Read and parse JSON data from a file.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file content is not valid JSON.
    """
    path = Path(filepath)
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: '{filepath}'")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{filepath}': {exc}") from exc


def remove_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and markdown code fences from LLM output."""
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown code fences (```lisp, ```json, ``` etc.)
    text = re.sub(r"^```\w*\n?", "", text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\n?```$", "", text.strip(), flags=re.MULTILINE)
    return text.strip()


# Alias kept for backwards compatibility within the package
think_remover = remove_think_tags
