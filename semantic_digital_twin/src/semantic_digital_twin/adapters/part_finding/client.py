"""
Asking a part finding server, such as the institute's internal one, where the parts of a
shape are.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from datetime import timedelta
from http import HTTPStatus

import requests
from typing_extensions import Optional

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.adapters.part_finding.exceptions import (
    PartFindingServerError,
    PartFindingServerUnknownError,
)
from semantic_digital_twin.adapters.part_finding.protocol import (
    ContentEncoding,
    ContentType,
    EnvironmentVariable,
    HeaderName,
    PartFindingAnswer,
    PartFindingRoute,
    PartFindingServerStatus,
)
from semantic_digital_twin.part_finding.locating import (
    LocatedPart,
    PartLocator,
    PartQuery,
)


@dataclass
class RemotePartLocator(PartLocator):
    """
    Locates parts by asking a part finding server.
    """

    base_url: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            EnvironmentVariable.PART_FINDING_SERVER_URL
        )
    )
    """
    The address of the server; by default the value of the environment variable
    ``SEMANTIC_DIGITAL_TWIN_PART_FINDING_SERVER_URL``.
    """

    timeout: timedelta = timedelta(seconds=30)
    """
    How long to wait for an answer, including the time the query waits behind others;
    the server takes about 5 seconds per object.
    """

    session: requests.Session = field(default_factory=requests.Session)
    """
    The connection the queries are sent on.
    """

    def __post_init__(self) -> None:
        if self.base_url is None:
            raise PartFindingServerUnknownError(
                variable=EnvironmentVariable.PART_FINDING_SERVER_URL
            )

    def locate(self, query: PartQuery) -> list[LocatedPart]:
        return self.answer(query).parts

    def answer(self, query: PartQuery) -> PartFindingAnswer:
        """
        :param query: The shape and the parts wanted.
        :return: The server's whole answer, with the parts and how it found them.
        """
        url = self._url(PartFindingRoute.PARTS)
        response = self.session.post(
            url,
            data=gzip.compress(json.dumps(to_json(query)).encode()),
            headers={
                HeaderName.CONTENT_ENCODING: ContentEncoding.GZIP,
                HeaderName.CONTENT_TYPE: ContentType.JSON,
            },
            timeout=self.timeout.total_seconds(),
        )
        if response.status_code != HTTPStatus.OK:
            raise PartFindingServerError(url=url, status_code=response.status_code)
        return from_json(response.json())

    def status(self) -> PartFindingServerStatus:
        """
        :return: Which locator the server runs and how busy it is.
        """
        url = self._url(PartFindingRoute.STATUS)
        response = self.session.get(url, timeout=self.timeout.total_seconds())
        if response.status_code != HTTPStatus.OK:
            raise PartFindingServerError(url=url, status_code=response.status_code)
        return from_json(response.json())

    def _url(self, route: PartFindingRoute) -> str:
        """
        :param route: A path the server answers.
        :return: Its address.
        """
        return f"{self.base_url.rstrip('/')}{route}"
