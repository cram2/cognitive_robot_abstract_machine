from __future__ import annotations

from dataclasses import dataclass, field
from http import HTTPStatus

from krrood.exceptions import DataclassException


@dataclass
class PartFindingServerUnknownError(DataclassException):
    """
    Raised when a part finding client is given no server to ask.
    """

    variable: str = field(kw_only=True)
    """
    The environment variable that names the server by default.
    """

    def error_message(self) -> str:
        return f"No part finding server was given, and {self.variable} is not set."

    def suggest_correction(self) -> str:
        return (
            f"set {self.variable} to the address of the institute's part finding "
            "server, or pass base_url."
        )


@dataclass
class PartFindingServerError(DataclassException):
    """
    Raised when a part finding server does not answer a query with the parts.
    """

    url: str = field(kw_only=True)
    """
    The address that was requested.
    """

    status_code: int = field(kw_only=True)
    """
    The status the server answered with.
    """

    def error_message(self) -> str:
        return f"The part finding server answered {self.status_code} for '{self.url}'."

    def suggest_correction(self) -> str:
        if self.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE:
            return "simplify the shape, or start the server with a higher face limit."
        if self.status_code == HTTPStatus.BAD_REQUEST:
            return (
                "ask only for part-whole relationships of the annotation type, and make "
                "sure client and server run the same version of semantic_digital_twin."
            )
        return "check that the part finding server is running and reachable."
