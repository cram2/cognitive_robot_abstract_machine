"""
The rules agents follow and the sections of ``AGENTS.md`` that list them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType

from typing_extensions import Any, Callable

from krrood.ormatic.utils import classproperty

# %% code named by rules


@dataclass(frozen=True)
class CodeReference:
    """
    A module, class or function a rule names, rendered as inline Markdown code, so the
    rule follows a rename of it.
    """

    target: ModuleType | type | Callable[..., Any]
    """
    The object the rule names.
    """

    public_module: ModuleType | None = None
    """
    The module the object is imported from, where that is not the module defining it.
    """

    @property
    def path(self) -> str:
        """
        The import path of the object, as inline code.
        """
        if isinstance(self.target, ModuleType):
            return f"`{self.target.__name__}`"
        module = self.public_module or self.target
        module_name = (
            module.__name__ if isinstance(module, ModuleType) else module.__module__
        )
        return f"`{module_name}.{self.target.__qualname__}`"

    @property
    def name(self) -> str:
        """
        The object's own name, as inline code.
        """
        return f"`{self.target.__name__}`"


# %% rules


@dataclass(frozen=True)
class AgentRule:
    """
    One rule agents follow, rendered as a Markdown list item.
    """

    statement: str
    """
    The rule itself.
    """

    details: tuple[str, ...] = ()
    """
    The points that spell the rule out, rendered as a nested list below it.
    """


# %% sections


class Section(AgentRule, Enum):
    """
    A section of ``AGENTS.md`` on one topic, whose members are the rules it lists.

    Its name is the topic followed by ``Rules``. A section nested under another
    overrides :attr:`parent` to return it.
    """

    @classproperty
    def parent(cls) -> type[Section] | None:
        """
        The section this one is nested under; none for a top-level section.
        """
        return None

    @classproperty
    def title(cls) -> str:
        """
        The text of the section's heading: the words of the topic in sentence case,
        keeping acronyms such as ``CI`` in capitals.
        """
        first_word, *other_words = re.findall(
            r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*", cls.__name__.removesuffix("Rules")
        )
        return " ".join(
            [
                first_word,
                *(word if word.isupper() else word.lower() for word in other_words),
            ]
        )

    @classproperty
    def subsections(cls) -> list[type[Section]]:
        """
        The sections nested directly under this one, in definition order.
        """
        return [
            section for section in Section.__subclasses__() if section.parent is cls
        ]

    @classproperty
    def depth(cls) -> int:
        """
        How deeply the section is nested, counting a top-level section as depth 1.
        """
        return 1 if cls.parent is None else cls.parent.depth + 1

    @classproperty
    def heading(cls) -> str:
        """
        The section's heading as a line of Markdown.
        """
        return f"{'#' * cls.depth} {cls.title}"

    @classproperty
    def source_path(cls) -> Path:
        """
        The path of the module declaring the section, relative to the directory its top-
        level package or module lies in.
        """
        return Path(*cls.__module__.split(".")).with_suffix(".py")
