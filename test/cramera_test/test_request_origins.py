"""
Only local browser pages may query or control capture on local servers.
"""

from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from .test_live_http import bridge, server
from .test_server import server as viewer_server
from cramera.request_origin import RequestOrigin


# %% HTTP origin boundary


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:99999",
        "http://[::1",
        "file://localhost",
        "http://user@localhost",
        "http://localhost/path",
        "http://localhost?query=1",
        "http://localhost#fragment",
        "http://192.0.2.1",
    ],
)
def test_malformed_or_nonlocal_origins_are_rejected(origin):
    """
    Malformed origins never inherit the permission of a local hostname.
    """
    assert RequestOrigin(origin).is_allowed() is False


def request_with_origin(url: str, origin: str, method: str = "POST"):
    """
    Request an endpoint with the origin header a browser supplies.
    """
    request = Request(
        url,
        method=method,
        data=b"{}" if method == "POST" else None,
        headers={"Origin": origin, "Content-Type": "application/json"},
    )
    try:
        return urlopen(request, timeout=10)
    except HTTPError as error:
        return error


@pytest.mark.parametrize(
    "origin", ["https://example.org", "null", "http://localhost.example.org"]
)
def test_live_server_rejects_untrusted_origins(server, origin):
    """
    A website cannot submit executable queries to the live process.
    """
    with request_with_origin(server + "/eql", origin) as response:
        assert response.status == 403
        assert response.headers["Access-Control-Allow-Origin"] is None


@pytest.mark.parametrize(
    "origin", ["https://example.org", "null", "http://localhost.example.org"]
)
def test_static_server_rejects_untrusted_origins(viewer_server, origin):
    """
    A website cannot submit executable queries to the viewer process.
    """
    with request_with_origin(viewer_server + "/api/eql", origin) as response:
        assert response.status == 403


def test_preflight_rejects_an_untrusted_origin(server):
    """
    Browser preflight never grants a website access to capture endpoints.
    """
    with request_with_origin(
        server + "/recording/stop", "https://example.org", "OPTIONS"
    ) as response:
        assert response.status == 403
        assert response.headers["Access-Control-Allow-Origin"] is None


@pytest.mark.parametrize(
    "origin", ["http://localhost:8711", "http://127.0.0.1:8711", "http://[::1]:8711"]
)
def test_live_server_reflects_only_an_allowed_origin(server, origin):
    """
    Local viewer pages retain their explicit cross-origin access.
    """
    with request_with_origin(server + "/state", origin, "GET") as response:
        assert response.status == 200
        assert response.headers["Access-Control-Allow-Origin"] == origin
