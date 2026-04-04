import os
from functools import lru_cache

from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from krrood.ormatic.utils import create_engine
from semantic_digital_twin.orm.exceptions import DatabaseNotAvailableError


def persistent_database_available() -> bool:
    """
    Check if a persistent database is available for connection.

    This function validates if the environment variable `semantic_digital_twin_DATABASE_URI`
    is set and attempts to establish a connection with the database using the
    provided URI. If the connection is successful, it returns True, indicating the
    database is available. Otherwise, it returns False.

    :return: Indicates whether the persistent database is accessible
    """
    semantic_digital_twin_database_uri = os.environ.get(
        "SEMANTIC_DIGITAL_TWIN_DATABASE_URI"
    )
    if semantic_digital_twin_database_uri is None:
        return False

    try:
        engine = create_engine(semantic_digital_twin_database_uri)
        with engine.connect():
            ...
    except OperationalError as e:
        return False
    return True


@lru_cache
def semantic_digital_twin_sessionmaker():
    """
    Creates a session maker for the semantic digital twin database.
    This requires the environment variable `SEMANTIC_DIGITAL_TWIN_DATABASE_URI` to be set to a reachable database.
    """
    uri = os.environ.get("SEMANTIC_DIGITAL_TWIN_DATABASE_URI")
    if uri is None:
        raise DatabaseNotAvailableError()
    engine = create_engine(uri)
    return sessionmaker(bind=engine)
