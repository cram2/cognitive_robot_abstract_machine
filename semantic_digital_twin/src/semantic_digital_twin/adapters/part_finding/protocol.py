"""
What a part finding server and its clients say to each other over http.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import StrEnum

from semantic_digital_twin.part_finding.locating import LocatedPart


class PartFindingRoute(StrEnum):
    """
    The paths a part finding server answers.
    """

    PARTS = "/parts"
    """
    Takes a gzip-compressed
    :class:`~semantic_digital_twin.part_finding.locating.PartQuery` and answers
    with a :class:`PartFindingAnswer`.
    """

    STATUS = "/status"
    """
    Answers with a :class:`PartFindingServerStatus`.
    """


class EnvironmentVariable(StrEnum):
    """
    The environment variables clients of a part finding server read.
    """

    PART_FINDING_SERVER_URL = "SEMANTIC_DIGITAL_TWIN_PART_FINDING_SERVER_URL"
    """
    The address of the server clients ask by default.
    """


class HeaderName(StrEnum):
    """
    The http headers the server and its clients read or write.
    """

    CONTENT_LENGTH = "Content-Length"
    """
    The size of the body, in bytes.
    """

    CONTENT_ENCODING = "Content-Encoding"
    """
    How the body is compressed.
    """

    CONTENT_TYPE = "Content-Type"
    """
    The format of the body.
    """


class ContentType(StrEnum):
    """
    The content types the server and its clients send.
    """

    JSON = "application/json"
    """
    A body in json.
    """


class ContentEncoding(StrEnum):
    """
    The encodings a query may be sent in.
    """

    GZIP = "gzip"
    """
    A body compressed with gzip.
    """


@dataclass
class PartFindingAnswer:
    """
    What the server answers a query with.
    """

    parts: list[LocatedPart]
    """
    The parts found.
    """

    locator: str
    """
    The settings of the locator that found them, as json.
    """

    duration: timedelta
    """
    How long finding them took, or reading them from the cache.
    """

    from_cache: bool
    """
    Whether the parts were found for the same query before.
    """


@dataclass
class PartFindingServerStatus:
    """
    What the server answers a status request with.
    """

    locator: str
    """
    The settings of the locator the server finds parts with, as json.
    """

    waiting_queries: int
    """
    How many queries are waiting for the locator, the one it is working on included.
    """
