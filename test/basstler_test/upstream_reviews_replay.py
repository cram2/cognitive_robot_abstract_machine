"""
Replaying the recorded GraphQL responses ``test_upstream_reviews.py`` reads.

The reader is exercised against responses captured from a real pull request rather than
against GitHub, so the suite needs no network access and no upstream pull request of its
own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from basstler.upstream_reviews import GraphQLClient, RepositoryJSON

RECORDED_RESPONSE_DIRECTORY = (
    Path(__file__).parent / "dataset" / "upstream-review-responses"
)
"""
Where the recorded GraphQL responses live.
"""


class RecordedResponse(StrEnum):
    """
    The recorded responses the tests replay, named by their filename stem.
    """

    PULL_REQUEST_PAGE_ONE = "pull_request_page_one"
    PULL_REQUEST_PAGE_TWO = "pull_request_page_two"
    BRANCH_PULL_REQUESTS = "branch_pull_requests"
    BRANCH_PULL_REQUESTS_FOREIGN_OWNER = "branch_pull_requests_foreign_owner"
    BRANCH_PULL_REQUESTS_NONE = "branch_pull_requests_none"

    def load(self) -> dict[str, Any]:
        """:return: The recorded ``data`` object this response holds."""
        return json.loads((RECORDED_RESPONSE_DIRECTORY / f"{self}.json").read_text())

    def recorded(self) -> RepositoryJSON:
        """
        Read this response through the same model the production code uses.

        Lets a test reach recorded values as typed attributes rather than indexing the
        raw data at the call site.

        :return: The recorded repository.
        """
        return RepositoryJSON.from_json(self.load())


@dataclass(frozen=True)
class RecordedCall:
    """
    One query the reader executed, kept so a test can assert on it.
    """

    query: str
    """
    The GraphQL document that was sent.
    """

    variables: dict[str, Any]
    """
    The variables it was sent with.
    """


@dataclass
class ReplayingClient(GraphQLClient):
    """
    A client that returns queued responses instead of calling GitHub.

    Records every call it was given, so a test can assert the exact request the reader
    made.
    """

    responses: list[dict[str, Any]]
    """
    The responses still to be returned, in order.
    """

    calls: list[RecordedCall] = field(default_factory=list)
    """
    Every call the reader executed, oldest first.
    """

    def execute(self, query: str, variables: dict[str, Any]) -> dict[str, Any]:
        """
        Return the next queued data.

        :param query: The GraphQL document, recorded for assertions.
        :param variables: The GraphQL variables, recorded for assertions.
        :return: The next queued data.
        """
        self.calls.append(RecordedCall(query, variables))
        return self.responses.pop(0)
