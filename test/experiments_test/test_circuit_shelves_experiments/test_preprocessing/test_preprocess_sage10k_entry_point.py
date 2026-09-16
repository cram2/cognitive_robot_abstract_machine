from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from experiments.orm.ormatic_interface import PreprocessedObjectDAO
from krrood.ormatic.utils import create_engine

from .test_preprocessing import _populated_sqlite_engine, _sage10k_object

# %% the documented entry point, run as its own process


def test_entry_point_writes_objects_across_its_own_worker_processes(
    tmp_path: Path,
) -> None:
    """
    Running the preprocessing entry point as its own process -- the way the real
    pipeline is launched -- must reach its spawned worker processes without losing track
    of which class an object belongs to.

    A worker process re-imports whatever module the entry point itself ran as; a class
    the entry point module defines, such as
    :class:`~experiments.scene_generation_experiments.preprocessing.preprocess_sage10k.PreprocessedObject`,
    then exists under two different identities somewhere in the run -- its own and the
    one the generated DAO interface points at. Regression test for
    :func:`~krrood.ormatic.data_access_objects.helper.get_dao_class` resolving those
    back to one DAO regardless.
    """
    objects = [
        _sage10k_object(object_id="book_1", room_id="room_1", source_id="book_1_src")
    ]
    _populated_sqlite_engine(tmp_path, objects)
    raw_uri = f"sqlite:///{tmp_path}/raw.db"
    processed_uri = f"sqlite:///{tmp_path}/processed.db"

    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "experiments.scene_generation_experiments.preprocessing.preprocess_sage10k",
        ],
        env={
            **os.environ,
            "SAGE10k_DATABASE_URI": raw_uri,
            "SAGE10K_PROCESSED_DATABASE_URI": processed_uri,
        },
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    with Session(create_engine(processed_uri)) as session:
        stored_ids = set(session.execute(select(PreprocessedObjectDAO.id)).scalars())
    assert stored_ids == {"book_1"}
