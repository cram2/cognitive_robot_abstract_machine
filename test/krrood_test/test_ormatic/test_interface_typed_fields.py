"""
A field typed with an interface is stored through the interface's mapping hierarchy, so
each value comes back as the implementation it was stored as.
"""

from __future__ import annotations

from sqlalchemy import inspect, select

from krrood.ormatic.data_access_objects.helper import to_dao
from ..dataset.interface_typed_fields import (
    HolderOfLabelledPositions,
    LabelledPosition2D,
    LabelledPosition3D,
)
from ..dataset.ormatic_interface import (
    HolderOfLabelledPositionsDAO,
    LabelledPosition2DMappingDAO,
    LabelledPosition3DMappingDAO,
    LabelledPositionMappingDAO,
)

# %% generated schema


def test_interface_typed_fields_relate_to_the_base_mapping():
    relationships = inspect(HolderOfLabelledPositionsDAO).relationships
    route_association = relationships["route"].mapper

    assert relationships["position"].mapper.class_ is LabelledPositionMappingDAO
    assert (
        route_association.relationships["target"].mapper.class_
        is LabelledPositionMappingDAO
    )


def test_the_mappings_of_each_dimension_extend_the_base_mapping():
    assert issubclass(LabelledPosition2DMappingDAO, LabelledPositionMappingDAO)
    assert issubclass(LabelledPosition3DMappingDAO, LabelledPositionMappingDAO)


# %% round trip


def test_interface_typed_fields_round_trip_to_their_implementations(session, database):
    holder = HolderOfLabelledPositions(
        position=LabelledPosition3D(x=1.0, y=2.0, z=3.0, origin_name="map"),
        route=[
            LabelledPosition2D(x=4.0, y=5.0, origin_name="floor"),
            LabelledPosition3D(x=6.0, y=7.0, z=8.0, origin_name="map"),
        ],
    )

    session.add(to_dao(holder))
    session.commit()
    session.expunge_all()

    reconstructed = (
        session.scalars(select(HolderOfLabelledPositionsDAO)).one().from_dao()
    )

    assert reconstructed == holder
