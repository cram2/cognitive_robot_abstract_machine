"""
Serving the live event dashboard: the page, the events so far, and the events as they
come.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from importlib import resources

from flask import Flask, Response, jsonify, render_template_string, stream_with_context
from typing_extensions import TYPE_CHECKING, Iterator, Optional, Self
from werkzeug.serving import BaseWSGIServer, make_server

from segmind.event_feed import EventFeed, EventRow, FeedField

if TYPE_CHECKING:
    from segmind.live_segmenter import LiveSegmenter

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
"""
The interface the dashboard is served on unless told otherwise.
"""

DEFAULT_PORT = 5000
"""
The port the dashboard is served on unless told otherwise.
"""

HEARTBEAT_INTERVAL_SECONDS = 15.0
"""
How long the event stream may stay silent before a comment line is sent, so a quiet
stretch does not read as a dropped connection.
"""


class DashboardRoute(StrEnum):
    """
    The paths the dashboard answers on.
    """

    PAGE = "/"
    """
    The page itself.
    """

    EVENTS = "/events"
    """
    Every event so far, as one JSON list.
    """

    EVENT_STREAM = "/events/stream"
    """
    Every event so far and then each new one, as server-sent events.
    """


class DashboardFile(StrEnum):
    """
    The files of the dashboard package the server reads.
    """

    PAGE = "live_events.html"
    """
    The page's template.
    """


class MediaType(StrEnum):
    """
    The media types the dashboard answers with besides JSON and HTML.
    """

    EVENT_STREAM = "text/event-stream"
    """
    A stream of server-sent events.
    """


class StreamMessage(StrEnum):
    """
    The fixed messages of the event stream.
    """

    HEARTBEAT = ": heartbeat\n\n"
    """
    A comment line keeping a quiet connection open.
    """


@dataclass(frozen=True)
class DashboardAddress:
    """
    Where the dashboard is served.
    """

    host: str = DEFAULT_HOST
    """
    The interface to bind to.
    """

    port: int = DEFAULT_PORT
    """
    The port to bind to; 0 lets the operating system pick a free one.
    """


@dataclass
class LiveEventDashboard:
    """
    A page listing the events of an :class:`~segmind.event_feed.EventFeed` as they are
    detected, served on a thread of its own between :meth:`start` and :meth:`stop`.
    """

    feed: EventFeed = field(default_factory=EventFeed)
    """
    The events shown.
    """

    address: DashboardAddress = field(default_factory=DashboardAddress)
    """
    Where the page is served.
    """

    app: Flask = field(init=False, repr=False)
    """
    The application answering the dashboard's routes.
    """

    _server: Optional[BaseWSGIServer] = field(init=False, default=None, repr=False)
    """
    The running server, between :meth:`start` and :meth:`stop`.
    """

    _thread: Optional[threading.Thread] = field(init=False, default=None, repr=False)
    """
    The thread serving requests, between :meth:`start` and :meth:`stop`.
    """

    def __post_init__(self) -> None:
        self.app = Flask(__name__)
        self.app.add_url_rule(DashboardRoute.PAGE, "page", self._page)
        self.app.add_url_rule(DashboardRoute.EVENTS, "events", self._events)
        self.app.add_url_rule(
            DashboardRoute.EVENT_STREAM, "event_stream", self._event_stream
        )

    @classmethod
    def watching(
        cls, segmenter: LiveSegmenter, address: Optional[DashboardAddress] = None
    ) -> Self:
        """
        A dashboard showing what ``segmenter`` detects from now on and before.

        :param segmenter: The watched run whose events are shown.
        :param address: Where the page is served; the default address when None.
        """
        dashboard = cls() if address is None else cls(address=address)
        segmenter.listeners.append(dashboard.feed)
        return dashboard

    @property
    def port(self) -> int:
        """
        The port the running dashboard is bound to.
        """
        return self._server.server_port

    def start(self) -> None:
        """
        Start serving the page on a thread of its own.
        """
        self._server = make_server(
            self.address.host, self.address.port, self.app, threaded=True
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info(
            "SegMind live event dashboard: http://%s:%d", self.address.host, self.port
        )

    def stop(self) -> None:
        """
        Stop serving, waiting for the serving loop to end.
        """
        if self._server is None:
            return
        self._server.shutdown()
        self._thread.join()
        self._server.server_close()
        self._server = None
        self._thread = None

    def _page(self) -> str:
        page = resources.files(__package__).joinpath(DashboardFile.PAGE).read_text()
        return render_template_string(
            page, fields=FeedField, event_stream_route=DashboardRoute.EVENT_STREAM
        )

    def _events(self) -> Response:
        return jsonify([EventRow.of(event).to_json() for event in self.feed.snapshot()])

    def _event_stream(self) -> Response:
        return Response(
            stream_with_context(self._stream_messages()),
            mimetype=MediaType.EVENT_STREAM,
        )

    def _stream_messages(self) -> Iterator[str]:
        """
        Every event so far and then each new one as a server-sent event, with a
        heartbeat whenever none arrives for :data:`HEARTBEAT_INTERVAL_SECONDS`.
        """
        subscription = self.feed.subscribe()
        while True:
            events = subscription.take(timeout=HEARTBEAT_INTERVAL_SECONDS)
            if not events:
                yield StreamMessage.HEARTBEAT
                continue
            for event in events:
                yield f"data: {json.dumps(EventRow.of(event).to_json())}\n\n"
