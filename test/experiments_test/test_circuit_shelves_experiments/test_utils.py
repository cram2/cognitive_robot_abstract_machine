from __future__ import annotations

from pathlib import Path

from experiments.shelf_generation_experiments.utils import build_source_id_to_path


def _scene_with_object(scenes_root: Path, scene_name: str, source_id: str) -> Path:
    """
    Create a scene directory under *scenes_root* holding a PLY/texture pair for
    *source_id*.

    File contents are irrelevant to :func:`build_source_id_to_path`, which only checks
    for the files' existence, so both are written empty.

    :return: The created scene directory.
    """
    objects_directory = scenes_root / scene_name / "objects"
    objects_directory.mkdir(parents=True, exist_ok=True)
    (objects_directory / f"{source_id}.ply").touch()
    (objects_directory / f"{source_id}_texture.png").touch()
    return objects_directory.parent


# %% build_source_id_to_path
def test_a_source_id_resolves_to_its_own_scene_directory(tmp_path: Path) -> None:
    """
    The ordinary case: one source_id, cached under one scene, resolves to that scene's
    own directory.
    """
    scene_directory = _scene_with_object(tmp_path, "scene_1", "book_src")

    assert build_source_id_to_path(tmp_path) == {"book_src": scene_directory}


def test_a_source_id_shared_by_two_scenes_is_dropped(tmp_path: Path) -> None:
    """
    Two independently generated scenes can produce the same short, hash-like source_id
    for two unrelated meshes (a birthday-paradox collision in the id space, confirmed
    against the live sage10k corpus: ~30 of 564,896 distinct source_ids collide this
    way). Picking one of the two scene directories would silently correct objects in
    the other scene against the wrong mesh's bounding box, so an ambiguous source_id is
    left out of the mapping entirely rather than arbitrarily resolved.
    """
    _scene_with_object(tmp_path, "scene_1", "collided_src")
    _scene_with_object(tmp_path, "scene_2", "collided_src")

    assert build_source_id_to_path(tmp_path) == {}


def test_an_unambiguous_source_id_survives_alongside_a_collision(
    tmp_path: Path,
) -> None:
    """
    Dropping a colliding source_id must not affect any other, unrelated source_id
    discovered in the same scan.
    """
    scene_directory = _scene_with_object(tmp_path, "scene_1", "unique_src")
    _scene_with_object(tmp_path, "scene_1", "collided_src")
    _scene_with_object(tmp_path, "scene_2", "collided_src")

    assert build_source_id_to_path(tmp_path) == {"unique_src": scene_directory}
