"""
Interfaces that fields are typed with, but that the mapped package does not declare,
keep their inheritance in the generated interface.
"""

from __future__ import annotations

from krrood.ormatic.ormatic import ORMatic
from ..dataset.interfaces_outside_the_mapped_package import mapped
from ..dataset.interfaces_outside_the_mapped_package.regions import Disc, Region

# %% inheritance through an interface without a table


def test_an_interface_inheriting_through_an_unmapped_interface_follows_its_parent() -> (
    None
):
    """
    An interface deriving from another only through an interface without a table
    inherits the other's table, which is generated before its own.
    """
    ormatic = ORMatic.from_package(
        packages=[mapped],
        ormatic_interface_dependencies=[],
        ignored_classes=set(),
        type_mappings={},
        ignore_krrood_test_classes=False,
    )
    order_of_tables = [
        wrapped_class.clazz for wrapped_class in ormatic.wrapped_tables.keys()
    ]
    disc_table = ormatic.tables_by_class[Disc]

    assert disc_table.parent_table is ormatic.tables_by_class[Region]
    assert order_of_tables.index(Region) < order_of_tables.index(Disc)
