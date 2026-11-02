"""
Asking a part finding server for the parts of a shape, with a locator that needs no
model standing in for the real one.
"""

from __future__ import annotations

from http import HTTPStatus

import numpy as np
import pytest
import requests
import trimesh

from semantic_digital_twin.adapters.part_finding.client import RemotePartLocator
from semantic_digital_twin.adapters.part_finding.protocol import EnvironmentVariable
from semantic_digital_twin.datastructures.camera_resolution import CameraResolution
from semantic_digital_twin.adapters.part_finding.exceptions import (
    PartFindingServerError,
    PartFindingServerUnknownError,
)
from semantic_digital_twin.adapters.part_finding.server import (
    PartFindingCache,
    PartFindingServer,
)
from semantic_digital_twin.part_finding.locating import PartQuery
from semantic_digital_twin.part_finding.vision_language import (
    FilledBoxSegmenter,
    KnownPartBoxDetector,
    VisionLanguagePartLocator,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import Mug
from semantic_digital_twin.spatial_computations.mesh_rendering import CameraRing


@pytest.fixture
def mug_query() -> PartQuery:
    """
    :return: A query for the handle of a mug whose last twelve faces are its handle.
    """
    cup = trimesh.creation.annulus(r_min=0.035, r_max=0.04, height=0.1)
    handle = trimesh.creation.box(extents=[0.02, 0.01, 0.06])
    handle.apply_translation([0.07, 0.0, 0.0])
    return PartQuery.from_mesh(trimesh.util.concatenate([cup, handle]), Mug, ["handle"])


@pytest.fixture
def server(mug_query, tmp_path):
    """
    :return: A server on a free local port whose locator boxes the mug's handle.
    """
    handle_faces = np.arange(len(mug_query.faces) - 12, len(mug_query.faces))
    server = PartFindingServer(
        locator=VisionLanguagePartLocator(
            detector=KnownPartBoxDetector(part_faces={"handle": [handle_faces]}),
            segmenter=FilledBoxSegmenter(),
            cameras=CameraRing(resolution=CameraResolution(width=128, height=128)),
        ),
        host="127.0.0.1",
        port=0,
        cache=PartFindingCache(directory=tmp_path),
    )
    server.start()
    yield server
    server.stop()


def test_a_remote_query_is_answered_like_a_local_one(server, mug_query):
    [remote] = RemotePartLocator(base_url=server.url).locate(mug_query)
    [local] = server.locator.locate(mug_query)

    assert remote.relationship_name == local.relationship_name
    assert np.array_equal(remote.faces, local.faces)
    assert remote.confidence == local.confidence


def test_a_repeated_query_is_answered_from_the_cache(server, mug_query):
    client = RemotePartLocator(base_url=server.url)

    first = client.answer(mug_query)
    second = client.answer(mug_query)

    assert not first.from_cache
    assert second.from_cache
    assert np.array_equal(second.parts[0].faces, first.parts[0].faces)
    assert second.locator == server.locator_settings


def test_a_query_the_locator_fails_on_is_no_longer_waiting(
    server, mug_query, monkeypatch
):
    def fail(query: PartQuery):
        raise RuntimeError("the locator failed")

    monkeypatch.setattr(server.locator, "locate", fail)

    with pytest.raises(RuntimeError):
        server.answer(mug_query)

    assert server.status().waiting_queries == 0


def test_a_shape_with_too_many_faces_is_refused(server, mug_query):
    server.maximum_faces = len(mug_query.faces) - 1

    with pytest.raises(PartFindingServerError) as raised:
        RemotePartLocator(base_url=server.url).locate(mug_query)

    assert raised.value.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE


def test_a_query_for_a_relationship_the_annotation_does_not_have_is_refused(
    server, mug_query
):
    mug_query.relationship_names = ["spout"]

    with pytest.raises(PartFindingServerError) as raised:
        RemotePartLocator(base_url=server.url).locate(mug_query)

    assert raised.value.status_code == HTTPStatus.BAD_REQUEST


def test_the_status_names_the_locator(server):
    status = RemotePartLocator(base_url=server.url).status()

    assert status.locator == server.locator_settings
    assert status.waiting_queries == 0


def test_an_unknown_path_is_not_found(server):
    assert requests.get(f"{server.url}/nothing").status_code == HTTPStatus.NOT_FOUND


def test_a_client_without_a_server_to_ask_fails(monkeypatch):
    monkeypatch.delenv(EnvironmentVariable.PART_FINDING_SERVER_URL, raising=False)

    with pytest.raises(PartFindingServerUnknownError):
        RemotePartLocator()


def test_a_client_asks_the_server_the_environment_names(monkeypatch, server):
    monkeypatch.setenv(EnvironmentVariable.PART_FINDING_SERVER_URL, server.url)

    assert RemotePartLocator().status().locator == server.locator_settings
