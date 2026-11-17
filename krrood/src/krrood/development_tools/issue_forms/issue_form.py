"""
GitHub issue forms declared as classes, each written into a YAML file of the repository's
``.github/ISSUE_TEMPLATE`` directory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from krrood.development_tools.agents.rule import ClassNamedDeclaration
from krrood.patterns.classproperty import classproperty

# %% text areas


class CodeLanguage(StrEnum):
    """
    The languages GitHub highlights the text of a text area as code in, by the names
    the form file uses.
    """

    PYTHON = "python"
    """
    Python source code.
    """

    SHELL = "shell"
    """
    Shell sessions and the output of commands.
    """


@dataclass(kw_only=True, frozen=True)
class TextArea:
    """
    A field of an issue form that the reporter fills with text of several lines,
    rendered under its label as a section of the issue.
    """

    identifier: str
    """
    The identifier of the field, unique within its form, which links to the form can
    fill it by.
    """

    label: str
    """
    The heading of the field, in the form and in the issue.
    """

    description: str | None = None
    """
    The explanation below the label of what the field asks for; none for a field its
    label explains.
    """

    code_language: CodeLanguage | None = None
    """
    The language the text is highlighted as code in; none for prose.
    """

    required: bool = False
    """
    Whether the form refuses an issue with the field left empty.
    """


# %% issue forms


class IssueForm(ClassNamedDeclaration, ABC):
    """
    An issue form, declared as a class named after it followed by ``IssueForm``, whose
    text areas are written in the order :attr:`text_areas` lists them.
    """

    @classproperty
    def name_suffix(cls) -> str:
        """
        ``IssueForm``, as :meth:`ClassNamedDeclaration.name_suffix` asks for.
        """
        return "IssueForm"

    @classproperty
    def path(cls) -> Path:
        """
        Where the form's file lies, relative to the repository root: named by the words
        of its topic in lower case, joined by hyphens.
        """
        file_name = "-".join(word.lower() for word in cls.topic_words)
        return Path(".github") / "ISSUE_TEMPLATE" / f"{file_name}.yml"

    @classproperty
    @abstractmethod
    def description(cls) -> str:
        """
        What the form is for, which GitHub shows next to its title when a reporter picks
        a form.
        """

    @classproperty
    def issue_type(cls) -> str | None:
        """
        The issue type, one the repository's organisation defines, that GitHub sets on
        every issue opened with the form; none to set no type.
        """
        return None

    @classproperty
    @abstractmethod
    def text_areas(cls) -> list[TextArea]:
        """
        The fields of the form.
        """
