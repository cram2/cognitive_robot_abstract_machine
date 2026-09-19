"""
The rendered Montessori scene, and the pipeline that reads it, shared by every test
module that needs detections rather than one component in isolation.
"""

from __future__ import annotations

import pytest

from experiments.montessori.perception.detections import MontessoriScene
from experiments.montessori.perception.orthophoto import WorkspaceRegion
from experiments.montessori.perception.pipeline import MontessoriPerceptionPipeline
from experiments.montessori.semantics import MontessoriShapeCategory

from .montessori_scene_renderer import MontessoriSceneRenderer, PlacedPiece


@pytest.fixture
def renderer() -> MontessoriSceneRenderer:
    return MontessoriSceneRenderer()


@pytest.fixture
def placed_pieces() -> list[PlacedPiece]:
    return [
        PlacedPiece(MontessoriShapeCategory.CUBE, x=0.58, y=0.15),
        PlacedPiece(MontessoriShapeCategory.CYLINDER, x=0.58, y=0.25),
        PlacedPiece(MontessoriShapeCategory.TRIANGULAR_PRISM, x=0.58, y=0.35),
    ]


@pytest.fixture
def pipeline(renderer: MontessoriSceneRenderer) -> MontessoriPerceptionPipeline:
    return MontessoriPerceptionPipeline(
        region=WorkspaceRegion(
            minimum_x=0.35, maximum_x=1.35, minimum_y=-0.45, maximum_y=0.75
        ),
        table_height=renderer.table_height,
        board_height=renderer.board_height,
    )


@pytest.fixture
def scene(
    pipeline: MontessoriPerceptionPipeline,
    renderer: MontessoriSceneRenderer,
    placed_pieces: list[PlacedPiece],
) -> MontessoriScene:
    return pipeline.detect(renderer.render(placed_pieces))
