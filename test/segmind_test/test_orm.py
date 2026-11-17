from __future__ import annotations

from sqlalchemy import create_engine, inspect

import cramph.orm.ormatic_interface
import semantic_digital_twin.orm.ormatic_interface
from segmind.orm.ormatic_interface import AbstractDetectorDAO, Base

# %% database tables


def test_every_table_of_segmind_can_be_created_in_a_database() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    assert set(inspect(engine).get_table_names()) == set(Base.metadata.tables)
    engine.dispose()


# %% tables of the interfaces segmind builds on


def test_a_detector_table_builds_on_the_table_of_cramph_s_node() -> None:
    """
    A detector is a cramph statechart node, so its table inherits the table cramph's
    interface maps the node's class to.
    """
    assert inspect(AbstractDetectorDAO).inherits.class_ is (
        cramph.orm.ormatic_interface.EndedByOwnerDAO
    )


def test_a_detector_refers_to_its_tracked_body_through_semantic_digital_twin_s_table() -> (
    None
):
    """
    The body a detector tracks is stored in the table of
    :mod:`semantic_digital_twin`'s interface, not copied into segmind's.
    """
    tracked_object = inspect(AbstractDetectorDAO).relationships["tracked_object"]
    assert tracked_object.mapper.class_ is (
        semantic_digital_twin.orm.ormatic_interface.BodyDAO
    )
