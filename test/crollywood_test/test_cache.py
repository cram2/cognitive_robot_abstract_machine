"""
Tests for :mod:`crollywood.cache`: where the kept pictures and readings lie.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crollywood.cache import EnvironmentVariable, SceneCache
from crollywood.frame import Resolution

# %% where the cache lies


def test_the_cache_lies_where_the_environment_says(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(EnvironmentVariable.CACHE_ROOT, str(tmp_path))
    assert SceneCache("scene").root == tmp_path


def test_the_cache_lies_at_the_fallback_when_the_environment_is_silent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(EnvironmentVariable.CACHE_ROOT, raising=False)
    assert SceneCache.configured_root(fallback=tmp_path) == tmp_path


# %% what is kept


def test_a_kept_picture_reads_back_the_same(tmp_path: Path) -> None:
    cache = SceneCache("scene", root=tmp_path)
    picture = Resolution(width=8, height=4).blank(grey=90)
    cache.keep_picture("still", picture)
    assert (cache.picture("still").pixels == picture.pixels).all()


def test_a_picture_never_kept_reads_back_as_none(tmp_path: Path) -> None:
    assert SceneCache("scene", root=tmp_path).picture("never kept") is None
