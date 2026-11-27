"""
The table every stored object has a row in, which every generated interface shares.
"""

from __future__ import annotations

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from krrood.ormatic.base import Base
from krrood.ormatic.data_access_objects.stored_object_row import StoredObjectRow

# %% the table of every stored object


class StoredObjectDAO(Base, StoredObjectRow):
    """
    The row every stored object has, whatever its class and whichever interface maps it.
    """

    __tablename__ = "StoredObjectDAO"

    database_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    """
    The id of the object, which its rows in all other tables share.
    """

    polymorphic_type: Mapped[str] = mapped_column(String(255), nullable=False)
    """
    The table name of the DAO class the object is loaded as.
    """
