"""
GitHub issue forms declared as classes, each written into a YAML file of the repository's
``.github/ISSUE_TEMPLATE`` directory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.development_tools.agents.rule import ClassNamedDeclaration
from krrood.development_tools.issue_forms.exceptions import UnknownTextArea
from krrood.patterns.classproperty import classproperty

# %% issue types


class IssueType(StrEnum):
    """
    The issue types an issue form can set, by the names GitHub organisations give them.
    """

    BUG = "Bug"
    """
    Code that behaves differently from its documentation or intent.
    """

    FEATURE = "Feature"
    """
    Something the code cannot do yet.
    """

    TASK = "Task"
    """
    Work that changes no behaviour.
    """


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

    @classproperty
    def unanswered_placeholder(cls) -> str:
        """
        What GitHub writes into an issue in place of the answer to a field left empty.
        """
        return "_No response_"

    @property
    def heading(self) -> str:
        """
        The line GitHub writes above the answer to the field in an issue's body.
        """
        return f"### {self.label}"

    def read_answer(self, *, section: str) -> str | None:
        """
        :param section: The lines of an issue's body between :attr:`heading` and the
            next field's heading.
        :return: The answer the section holds, code without its fence; none for a field
            left empty.
        """
        answer = section.strip()
        if answer == self.unanswered_placeholder:
            return None
        lines = answer.splitlines()
        is_fenced = (
            len(lines) >= 2 and lines[0].startswith("```") and lines[-1] == "```"
        )
        if self.code_language is None or not is_fenced:
            return answer
        return "\n".join(lines[1:-1])


# %% issue forms


class IssueForm(ClassNamedDeclaration, ABC):
    """
    An issue form, declared as a class named after it followed by ``IssueForm``, whose
    fields are its classproperties returning a :class:`TextArea`.
    """

    @classproperty
    def name_suffix(cls) -> str:
        """
        ``IssueForm``, as :meth:`ClassNamedDeclaration.name_suffix` asks for.
        """
        return "IssueForm"

    @classproperty
    def name(cls) -> str:
        """
        The name the form is looked up by: the words of its topic in lower case, joined
        by hyphens.
        """
        return "-".join(word.lower() for word in cls.topic_words)

    @classproperty
    def path(cls) -> Path:
        """
        Where the form's file lies, relative to the repository root, named by
        :attr:`name`.
        """
        return Path(".github") / "ISSUE_TEMPLATE" / f"{cls.name}.yml"

    @classproperty
    @abstractmethod
    def description(cls) -> str:
        """
        What the form is for, which GitHub shows next to its title when a reporter picks
        a form.
        """

    @classproperty
    def issue_type(cls) -> IssueType | None:
        """
        The issue type GitHub sets on every issue opened with the form, which the
        repository's organisation has to define; none to set no type.
        """
        return None

    @classproperty
    def text_areas(cls) -> list[TextArea]:
        """
        The fields of the form: every classproperty of it returning a
        :class:`TextArea`, in the order the form and its base forms define them.
        """
        text_area_properties = {
            name: value
            for form in reversed(cls.__mro__)
            if issubclass(form, IssueForm)
            for name, value in vars(form).items()
            if isinstance(value, classproperty)
            and get_type_hints_of_object(value.fget).get("return") is TextArea
        }
        return [value.fget(cls) for value in text_area_properties.values()]

    @classmethod
    def find_text_area(cls, *, identifier: str) -> TextArea:
        """
        :param identifier: The identifier of one of the form's text areas.
        :return: The text area with that identifier.
        :raises UnknownTextArea: If no text area of the form has it.
        """
        for text_area in cls.text_areas:
            if text_area.identifier == identifier:
                return text_area
        raise UnknownTextArea(
            form_name=cls.name,
            identifier=identifier,
            known_identifiers=[text_area.identifier for text_area in cls.text_areas],
        )

    @classmethod
    def find_answer(cls, *, issue_body: str, text_area: TextArea) -> str | None:
        """
        :param issue_body: The body of an issue opened with the form.
        :param text_area: One of the form's text areas.
        :return: The issue's answer to the text area; none if the field was left empty
            or its heading is no longer in the body.
        """
        lines = issue_body.splitlines()
        if text_area.heading not in lines:
            return None
        start = lines.index(text_area.heading) + 1
        headings = {area.heading for area in cls.text_areas}
        end = next(
            (index for index in range(start, len(lines)) if lines[index] in headings),
            len(lines),
        )
        return text_area.read_answer(section="\n".join(lines[start:end]))
