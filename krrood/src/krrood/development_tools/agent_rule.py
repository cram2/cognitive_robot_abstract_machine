"""
The rules agents follow, the sections of ``AGENTS.md`` that declare them and the rule
sets those sections form.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import ModuleType

from typing_extensions import Any, Callable

from krrood.development_tools.exceptions import SectionWithoutParent
from krrood.development_tools.repository_files import RepositoryInstructionsFile
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
    A section of ``AGENTS.md``, whose members are the rules it lists. A member's value
    is the rule's statement, or its statement and details.
    """

    @classproperty
    def parent(cls) -> type[Section] | None:
        """
        The section this one is nested under; none for a rule set.

        :raises SectionWithoutParent: If the section names no section it is nested
            under.
        """
        raise SectionWithoutParent(cls)

    @classproperty
    def title(cls) -> str:
        """
        The text of the section's heading: the words of the class name in sentence case,
        keeping acronyms such as ``CI`` in capitals.
        """
        first_word, *other_words = re.findall(
            r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*", cls.__name__
        )
        return " ".join(
            [
                first_word,
                *(word if word.isupper() else word.lower() for word in other_words),
            ]
        )

    @classproperty
    def rules(cls) -> list[Section]:
        """
        The rules the section lists directly, in declaration order.
        """
        return list(cls)

    @classproperty
    def subsections(cls) -> list[type[Section]]:
        """
        The sections nested directly under this one, in definition order.
        """
        return [
            section for section in Section.__subclasses__() if section.parent is cls
        ]

    @classproperty
    def has_rules(cls) -> bool:
        """
        Whether the section lists any rule, directly or in a subsection.
        """
        return bool(cls.rules) or any(
            subsection.has_rules for subsection in cls.subsections
        )

    @classproperty
    def depth(cls) -> int:
        """
        How deeply the section is nested, counting a rule set as depth 1.
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
        The path of the module declaring the section, relative to the directory its
        top-level package or module lies in.
        """
        return Path(*cls.__module__.split(".")).with_suffix(".py")


# %% rule sets


class RuleSet(Section):
    """
    A set of rules a repository adopts as a whole: a top-level section of
    ``AGENTS.md``. The rule sets are the subclasses of this class.
    """

    @classproperty
    def parent(cls) -> None:
        return None

    @classproperty
    def available(cls) -> list[type[RuleSet]]:
        """
        Every rule set declared, in definition order.
        """
        return cls.__subclasses__()

    @classproperty
    def command_line_name(cls) -> str:
        """
        The name the command line chooses the rule set by.
        """
        return cls.__name__.removesuffix("Rules").lower()


class PythonRules(RuleSet):
    """
    The rule set for any Python repository.
    """


class KrroodRules(RuleSet):
    """
    The rule set for a repository built on krrood, naming the krrood API it uses.
    """


class ProjectRules(RuleSet):
    """
    The rule set of one repository only, whose sections the repository declares in its
    :attr:`~krrood.development_tools.repository_files.RepositoryInstructionsFile.PROJECT_RULES`
    or in modules that one imports.
    """

    @classmethod
    def import_from(cls, repository_root: Path) -> ModuleType | None:
        """
        Import the repository's rules module, once per process, which declares the
        repository's sections.

        :param repository_root: The directory holding the module.
        :return: The module; none if the repository has no such module.
        """
        path = repository_root / RepositoryInstructionsFile.PROJECT_RULES
        if not path.is_file():
            return None
        module_name = path.stem
        if module_name in sys.modules:
            return sys.modules[module_name]
        specification = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        specification.loader.exec_module(module)
        return module
