from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """The single declarative base every generated ormatic interface shares."""

    type_mappings = {}
