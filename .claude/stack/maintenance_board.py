"""
Reading fetched pull requests into the board the stack is derived from.

A fetch that drops a field is not partially correct, so every field the board is derived
from is declared with how to read it and whether it may be absent, and a record omitting
a required one is rejected rather than defaulted.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from maintenance_constants import SESSION_LINK_PATTERN
from stack import BOARD_PATH, PullRequest

PullRequestRecord = Mapping[str, Any]
"""
One pull request as the REST API answers it, before any field is read.
"""

# %% the fields a board is read from


class PullRequestFieldShape(StrEnum):
    """
    How one pull-request field's value has to be read.

    The API answers some fields with a nested object where a plain value would do, so
    reading is per-field rather than uniform.
    """

    VALUE = "value"
    """
    Taken as it comes.
    """

    BRANCH_REFERENCE = "branch-reference"
    """
    A branch, given either plainly or as an object carrying a ``ref``.
    """

    LABEL_NAMES = "label-names"
    """
    A list of labels, each given either plainly or as an object carrying a ``name``.
    """


@dataclass(frozen=True)
class PullRequestFieldSpecification:
    """
    What one pull-request field is called, how to read it, and whether it may be absent.
    """

    key: str
    """
    The key the API answers under.
    """

    shape: PullRequestFieldShape = PullRequestFieldShape.VALUE
    """
    How its value has to be read.
    """

    required: bool = False
    """
    Whether a record omitting it is rejected rather than read.
    """


class PullRequestField(PullRequestFieldSpecification, Enum):
    """
    Every pull-request field this executor reads, and how to read it.

    Each member *is* a specification, so nothing outside this enum knows that ``head``
    arrives nested while ``draft`` does not, or which fields a board cannot be derived
    without.

    A member is written as the specification it carries, and :meth:`__init__` unpacks it
    onto the member itself - so ``PullRequestField.HEAD.key`` reads directly and the
    member is a :class:`PullRequestFieldSpecification` in its own right.
    """

    def __init__(self, specification: PullRequestFieldSpecification) -> None:
        """
        Carry the specification's values on the member itself.

        Without this the mixin would receive the whole specification as its first
        argument - silently, landing the instance in :attr:`key` - since an enum passes
        a member's value straight to the type it mixes in.

        :param specification: What this field is called and how to read it.
        """
        for field in dataclasses.fields(PullRequestFieldSpecification):
            object.__setattr__(self, field.name, getattr(specification, field.name))

    NUMBER = PullRequestFieldSpecification(key="number", required=True)
    """
    The pull request's number.
    """
    HEAD = PullRequestFieldSpecification(
        key="head", shape=PullRequestFieldShape.BRANCH_REFERENCE, required=True
    )
    """The branch the pull request would merge - the stack node it names."""
    BASE = PullRequestFieldSpecification(
        key="base", shape=PullRequestFieldShape.BRANCH_REFERENCE, required=True
    )
    """The branch it would merge into - its parent in the stack."""
    DRAFT = PullRequestFieldSpecification(key="draft", required=True)
    """
    Whether its author has yet reviewed it themselves.
    """
    LABELS = PullRequestFieldSpecification(
        key="labels", shape=PullRequestFieldShape.LABEL_NAMES, required=True
    )
    """
    The labels it carries, which the workflow reads as state.
    """
    BODY = PullRequestFieldSpecification(key="body")
    """
    Its description, read for the session link and the promotion prefill.
    """
    TITLE = PullRequestFieldSpecification(key="title")
    """
    Its title, which prefills the upstream pull request.
    """
    MERGEABLE_STATE = PullRequestFieldSpecification(key="mergeable_state")
    """
    GitHub's own verdict on whether it currently conflicts with its base.
    """

    def read(self, record: PullRequestRecord, number: int | None = None) -> Any:
        """
        Read this field out of a fetched pull request.

        :param record: The fetched pull request.
        :param number: The pull request being read, named in any rejection.
        :return: The field's value, read according to its shape.
        :raises MissingPullRequestFieldError: If a required field is absent, or its
            value carries no name where its shape says one belongs.
        """
        value = record.get(self.key)
        if value is None:
            if self.required:
                raise MissingPullRequestFieldError(self, number)
            return None
        match self.shape:
            case PullRequestFieldShape.BRANCH_REFERENCE:
                return self._branch_reference(value, number)
            case PullRequestFieldShape.LABEL_NAMES:
                return [
                    label if isinstance(label, str) else str(label["name"])
                    for label in value
                ]
            case _:
                return value

    def _branch_reference(self, value: Any, number: int | None) -> str:
        """:param value: The field's value, plain or nested.
        :param number: The pull request being read, named in any rejection.
        :return: The branch it names.
        :raises MissingPullRequestFieldError: If it names none."""
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping) and value.get("ref"):
            return str(value["ref"])
        raise MissingPullRequestFieldError(self, number)


@dataclass
class MissingPullRequestFieldError(ValueError):
    """
    Raised when a fetched pull request omits a field the board is derived from.

    A fetch that drops a field is not partially correct: absent and legitimately empty
    are different facts, and defaulting one to the other is what makes bad board data
    indistinguishable from good.
    """

    field_name: PullRequestField
    """
    The field that was absent.
    """

    pull_request_number: int | None
    """
    The pull request it was absent from, or ``None`` when the number itself is.
    """

    def __str__(self) -> str:
        """:return: Which field is missing, and from where."""
        subject = (
            f"pull request {self.pull_request_number}"
            if self.pull_request_number is not None
            else "a fetched pull request"
        )
        return (
            f"{subject} has no '{self.field_name}'; the board cannot be derived from a "
            f"fetch that omits it"
        )


def get_session_link_in(body: str | None) -> str | None:
    """
    Read the session link out of a pull request description.

    :param body: The description to search, which may be absent.
    :return: The first session link, or ``None`` if the description names none.
    """
    if not body:
        return None
    found = SESSION_LINK_PATTERN.search(body)
    return found.group(0) if found else None


# %% the export itself


@dataclass(frozen=True)
class BoardExport:
    """
    The fork's open pull requests, in the shape the derived stack is read from.
    """

    pull_requests: tuple[PullRequest, ...]
    """
    The exported pull requests.
    """

    @classmethod
    def from_api_records(cls, records: Iterable[PullRequestRecord]) -> BoardExport:
        """
        Build the export from what the REST API returned.

        :param records: The fetched pull requests.
        :return: The export.
        :raises MissingPullRequestFieldError: If any record omits a required field.
        """
        return cls(tuple(cls._pull_request(record) for record in records))

    @staticmethod
    def _pull_request(record: PullRequestRecord) -> PullRequest:
        """
        Read one fetched pull request into a board entry.

        :param record: The fetched pull request.
        :return: The board entry.
        :raises MissingPullRequestFieldError: If a required field is absent.
        """
        number = int(PullRequestField.NUMBER.read(record))
        return PullRequest(
            number=number,
            head=PullRequestField.HEAD.read(record, number),
            base=PullRequestField.BASE.read(record, number),
            draft=bool(PullRequestField.DRAFT.read(record, number)),
            labels=PullRequestField.LABELS.read(record, number),
            ci=record.get("ci"),
            session=get_session_link_in(PullRequestField.BODY.read(record, number)),
        )

    def as_json(self) -> str:
        """:return: The export, in the document :func:`stack.load_board` parses."""
        return json.dumps(
            {"pull_requests": [asdict(entry) for entry in self.pull_requests]},
            indent=2,
        )

    def write(self, path: Path = BOARD_PATH) -> Path:
        """
        Write the export where the derived stack is read from.

        :param path: Where to write it.
        :return: The path written to.
        """
        path.write_text(self.as_json() + "\n")
        return path
