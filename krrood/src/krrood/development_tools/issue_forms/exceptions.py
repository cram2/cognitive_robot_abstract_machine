"""
Exceptions raised when looking up issue forms and their fields.
"""

from __future__ import annotations

from dataclasses import dataclass

from krrood.exceptions import DataclassException


@dataclass(kw_only=True)
class UnknownIssueForm(DataclassException):
    """
    Raised when no issue form krrood ships or the repository declares has a name.
    """

    name: str
    """
    The name looked for.
    """

    known_names: list[str]
    """
    The names of every form there is.
    """

    def error_message(self) -> str:
        return f"No issue form is named `{self.name}`."

    def suggest_correction(self) -> str:
        return f"Name one of the issue forms: {', '.join(self.known_names)}."


@dataclass(kw_only=True)
class UnknownTextArea(DataclassException):
    """
    Raised when no text area of an issue form has an identifier.
    """

    form_name: str
    """
    The name of the form looked in.
    """

    identifier: str
    """
    The identifier looked for.
    """

    known_identifiers: list[str]
    """
    The identifiers of every text area of the form.
    """

    def error_message(self) -> str:
        return f"The issue form `{self.form_name}` has no field `{self.identifier}`."

    def suggest_correction(self) -> str:
        return f"Name one of its fields: {', '.join(self.known_identifiers)}."
