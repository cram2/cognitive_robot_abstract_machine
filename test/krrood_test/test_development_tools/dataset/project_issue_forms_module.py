"""
A repository's own issue forms module, standing in for the one a repository declares.
"""

from __future__ import annotations

from krrood.development_tools.issue_forms.issue_form import (
    CodeLanguage,
    IssueForm,
    TextArea,
)
from krrood.patterns.classproperty import classproperty

# %% issue forms


class CrashReportIssueForm(IssueForm):
    """
    A form with an issue type and a text area of every kind.
    """

    @classproperty
    def description(cls) -> str:
        return "Report a crash."

    @classproperty
    def issue_type(cls) -> str:
        """
        A crash report is a ``Bug``, overriding :meth:`IssueForm.issue_type`.
        """
        return "Bug"

    @classproperty
    def what_crashed(cls) -> TextArea:
        """
        A required prose field.
        """
        return TextArea(
            identifier="what-crashed",
            label="What crashed",
            description="The program and what it was doing: say it plainly.",
            required=True,
        )

    @classproperty
    def script(cls) -> TextArea:
        """
        An optional field holding Python code.
        """
        return TextArea(
            identifier="script",
            label="Script",
            code_language=CodeLanguage.PYTHON,
        )

    @classproperty
    def text_areas(cls) -> list[TextArea]:
        return [cls.what_crashed, cls.script]


class AnnouncementIssueForm(IssueForm):
    """
    A form without an issue type, listed before the first although defined after it.
    """

    @classproperty
    def description(cls) -> str:
        return "Announce something."

    @classproperty
    def text_areas(cls) -> list[TextArea]:
        return [TextArea(identifier="news", label="News")]
