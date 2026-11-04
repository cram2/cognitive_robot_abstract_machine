"""
A server that finds the parts of shapes for clients that have no GPU, such as the one on
the institute's internal server. Start it with ``semantic-digital-twin-part-finding-
server``, see :mod:`~semantic_digital_twin.adapters.part_finding.command_line`.

The server answers one query at a time and keeps every answer, so the same question
about the same shape is only ever worked out once.

.. warning:: The server checks no credentials; run it only inside a trusted network.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from typing_extensions import Any

from krrood.adapters.json_serializer import from_json, to_json
from semantic_digital_twin.adapters.part_finding.protocol import (
    ContentEncoding,
    ContentType,
    HeaderName,
    PartFindingAnswer,
    PartFindingRoute,
    PartFindingServerStatus,
)
from semantic_digital_twin.part_finding.relationships import PartRelationshipInspector
from semantic_digital_twin.part_finding.locating import (
    LocatedPart,
    PartQuery,
)
from semantic_digital_twin.part_finding.vision_language import (
    VisionLanguagePartLocator,
)
from semantic_digital_twin.semantic_annotations.mixins import PartWholeRelationship
from semantic_digital_twin.utils import create_cache_dir

logger = logging.getLogger(__name__)

# %% keeping the answers


@dataclass
class PartFindingCache:
    """
    The parts found for every query so far, one file per query and locator.
    """

    directory: Path = field(default_factory=lambda: create_cache_dir("part_finding"))
    """
    Where the answers are kept.
    """

    def key(self, query: PartQuery, locator: str) -> str:
        """
        :param query: A query.
        :param locator: The settings of the locator that answers it.
        :return: A name that is the same exactly for the same query to the same locator.
        """
        content = json.dumps(to_json(query), sort_keys=True) + locator
        return hashlib.sha256(content.encode()).hexdigest()

    def read(self, key: str) -> list[LocatedPart] | None:
        """
        :param key: The query's key.
        :return: The parts found for it before; ``None`` if it was not answered yet.
        """
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        return [from_json(part) for part in json.loads(path.read_text())]

    def write(self, key: str, parts: list[LocatedPart]) -> None:
        """
        Keep the parts found for a query.

        The file appears only once it is complete.
        :param key: The query's key.
        :param parts: The parts found for it.
        """
        path = self.directory / f"{key}.json"
        partial_path = path.with_name(f"{path.name}.partial")
        partial_path.write_text(json.dumps([to_json(part) for part in parts]))
        os.replace(partial_path, path)


# %% the server


@dataclass
class PartFindingServer:
    """
    Answers part queries over http with a locator that runs on this machine.

    The server listens as soon as it is created; :meth:`serve_forever` or :meth:`start`
    makes it answer.
    """

    locator: VisionLanguagePartLocator
    """
    Finds the parts.
    """

    host: str = "0.0.0.0"
    """
    The address the server listens on.
    """

    port: int = 8790
    """
    The port the server listens on; 0 picks a free one.
    """

    cache: PartFindingCache = field(default_factory=PartFindingCache)
    """
    Keeps the answers.
    """

    maximum_faces: int = 1_000_000
    """
    The most faces a query's shape may have.
    """

    maximum_request_bytes: int = 512 * 2**20
    """
    The largest query the server reads, compressed.
    """

    waiting_queries: int = field(default=0, init=False)
    """
    How many queries are waiting for the locator, the one it is working on included.
    """

    _locator_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    """
    Lets one query at a time use the locator.
    """

    _counter_lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    """
    Guards :attr:`waiting_queries`.
    """

    http_server: ThreadingHTTPServer = field(init=False)
    """
    Listens for and answers the requests.
    """

    def __post_init__(self) -> None:
        self.http_server = ThreadingHTTPServer(
            (self.host, self.port), partial(PartFindingRequestHandler, self)
        )

    @property
    def url(self) -> str:
        """
        :return: The address clients on this machine reach the server at.
        """
        return f"http://localhost:{self.http_server.server_address[1]}"

    @property
    def locator_settings(self) -> str:
        """
        :return: The locator's settings as json, such as its models and the prompt, which
            the cache keys its answers by. A model is named by its enum member, so clear
            the cache after pinning a member to a new revision.
        """
        return json.dumps(to_json(self.locator), sort_keys=True)

    def serve_forever(self) -> None:
        """
        Load the locator's models and answer requests until the process is stopped.
        """
        self.locator.load()
        logger.info("Finding parts at %s with %s", self.url, self.locator_settings)
        self.http_server.serve_forever()

    def start(self) -> threading.Thread:
        """
        Answer requests in the background until :meth:`stop` is called.

        :return: The thread answering them.
        """
        thread = threading.Thread(target=self.http_server.serve_forever, daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        """
        Stop answering requests and release the port.
        """
        self.http_server.shutdown()
        self.http_server.server_close()

    def answer(self, query: PartQuery) -> PartFindingAnswer:
        """
        :param query: A query.
        :return: The parts found for it, from the cache if it was asked before.
        """
        start = time.perf_counter()
        settings = self.locator_settings
        key = self.cache.key(query, settings)
        cached = self.cache.read(key)
        if cached is not None:
            return PartFindingAnswer(
                parts=cached,
                locator=settings,
                duration=timedelta(seconds=time.perf_counter() - start),
                from_cache=True,
            )
        with self._counter_lock:
            self.waiting_queries += 1
        try:
            with self._locator_lock:
                # A query for the same shape may have been answered while this one waited.
                parts = self.cache.read(key)
                from_cache = parts is not None
                if not from_cache:
                    parts = self.locator.locate(query)
                    self.cache.write(key, parts)
        finally:
            with self._counter_lock:
                self.waiting_queries -= 1
        seconds = time.perf_counter() - start
        logger.info(
            "%s: %d faces, %d parts, %.1f s",
            key[:12],
            len(query.faces),
            len(parts),
            seconds,
        )
        return PartFindingAnswer(
            parts=parts,
            locator=settings,
            duration=timedelta(seconds=seconds),
            from_cache=from_cache,
        )

    def status(self) -> PartFindingServerStatus:
        """
        :return: Which locator the server runs and how busy it is.
        """
        return PartFindingServerStatus(
            locator=self.locator_settings, waiting_queries=self.waiting_queries
        )

    def is_answerable(self, query: Any) -> bool:
        """
        :param query: What a client sent.
        :return: Whether it is a query about parts of a part-whole relationship type
            that this server can find.
        """
        if not isinstance(query, PartQuery):
            return False
        if not (
            isinstance(query.whole_type, type)
            and issubclass(query.whole_type, PartWholeRelationship)
        ):
            return False
        known = {
            relationship.name
            for relationship in PartRelationshipInspector.all_of_type(query.whole_type)
        }
        return set(query.relationship_names) <= known


# %% answering a request


@dataclass(eq=False)
class PartFindingRequestHandler(BaseHTTPRequestHandler):
    """
    Answers one http request to a :class:`PartFindingServer`.

    The http server creates one for each request, which is answered as soon as it is
    created.
    """

    part_finding_server: PartFindingServer
    """
    The server the request is answered for.
    """

    request: socket.socket
    """
    The connection the request came in on.
    """

    client_address: tuple[str, int]
    """
    The address of the client.
    """

    server: ThreadingHTTPServer
    """
    The http server that received the request.
    """

    def __post_init__(self) -> None:
        super().__init__(self.request, self.client_address, self.server)

    def do_GET(self) -> None:
        """
        Answer a status request.
        """
        if self.path != PartFindingRoute.STATUS:
            self._send_empty(HTTPStatus.NOT_FOUND)
            return
        self._send_json(to_json(self.part_finding_server.status()))

    def do_POST(self) -> None:
        """
        Answer a part query.
        """
        server = self.part_finding_server
        if self.path != PartFindingRoute.PARTS:
            self._send_empty(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get(HeaderName.CONTENT_LENGTH, 0))
        if length > server.maximum_request_bytes:
            self._send_empty(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        body = self.rfile.read(length)
        if self.headers.get(HeaderName.CONTENT_ENCODING) == ContentEncoding.GZIP:
            body = gzip.decompress(body)
        query = from_json(json.loads(body))
        if not server.is_answerable(query):
            self._send_empty(HTTPStatus.BAD_REQUEST)
            return
        if len(query.faces) > server.maximum_faces:
            self._send_empty(HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        self._send_json(to_json(server.answer(query)))

    def log_message(self, message_format: str, *arguments: Any) -> None:
        logger.debug(message_format, *arguments)

    def _send_json(self, content: Any) -> None:
        """
        :param content: The json to answer with, with status 200.
        """
        body = json.dumps(content).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header(HeaderName.CONTENT_TYPE, ContentType.JSON)
        self.send_header(HeaderName.CONTENT_LENGTH, str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: HTTPStatus) -> None:
        """
        :param status: The status to answer with, without a body.
        """
        self.send_response(status)
        self.send_header(HeaderName.CONTENT_LENGTH, "0")
        self.end_headers()
