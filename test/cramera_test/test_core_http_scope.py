"""
The core viewer exposes inspection and recording without robot editing.
"""

import pytest

from .test_live_http import bridge, server, post
from .test_server import server as viewer_server, post as viewer_post


# %% passive live endpoints


@pytest.mark.parametrize(
    "route", ["/move", "/joint", "/constraint", "/teleop", "/teleop/stop"]
)
def test_live_editing_routes_are_absent(server, route):
    """
    Requests cannot drive or modify the observed world.
    """
    status, payload = post(server + route)

    assert status == 404
    assert payload["ok"] is False


# %% static server scope


@pytest.mark.parametrize(
    "route",
    [
        "/api/plan/save",
        "/api/plan/scaffold",
        "/api/plan/scaffold/stop",
        "/api/models/load",
    ],
)
def test_builder_and_model_workbench_routes_are_absent(viewer_server, route):
    """
    The static viewer accepts no generated programs or model edits.
    """
    status, payload = viewer_post(viewer_server + route)

    assert status == 404
    assert payload["ok"] is False
