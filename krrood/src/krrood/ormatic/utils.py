from __future__ import annotations

import datetime
import inspect
import json
import types
from contextlib import suppress
from enum import Enum
from types import ModuleType

import sqlalchemy
from sqlalchemy import (
    Engine,
    text,
    MetaData,
    create_engine as create_sqlalchemy_engine,
    URL,
)
from sqlalchemy.orm import DeclarativeBase
from typing_extensions import (
    TypeVar,
    _SpecialForm,
    Type,
    List,
    Iterable,
    Union,
    Tuple,
    Dict,
    Any,
)

from .dao import AlternativeMapping, DataAccessObject
from ..adapters.json_serializer import to_json, from_json


class classproperty:
    """
    A decorator that allows a class method to be accessed as a property.
    """

    def __init__(self, fget):
        self.fget = fget

    def __get__(self, instance, owner):
        return self.fget(owner)


def classes_of_module(module: types.ModuleType) -> List[Type]:
    """
    Get all classes of a given module.

    :param module: The module to inspect.
    :return: All classes of the given module.
    """

    result = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            result.append(obj)
    return result


T = TypeVar("T")

leaf_types = (int, float, str, Enum, datetime.datetime, bool)


def _drop_fk_constraints(engine: Engine, tables: Iterable[str]) -> None:
    """
    Drops foreign key constraints for the specified tables in the given engine.

    This function removes all foreign key constraints for the specified list
    of tables using the provided database engine. It supports multiple
    SQL dialects, including MySQL, PostgreSQL, SQLite, and others.

    :param engine: The SQLAlchemy Engine instance used to interact with
        the database.
    :param tables: An iterable of table names whose foreign key constraints
        need to be dropped.
    """
    insp = sqlalchemy.inspect(engine)
    dialect = engine.dialect.name.lower()

    with engine.begin() as conn:
        for table in tables:
            for fk in insp.get_foreign_keys(table):
                name = fk.get("name")
                if not name:  # unnamed FKs (e.g. SQLite)
                    continue

                if dialect.startswith("mysql"):
                    stmt = text(f"ALTER TABLE `{table}` DROP FOREIGN KEY `{name}`")
                else:  # PostgreSQL, SQLite, MSSQL, …
                    stmt = text(f'ALTER TABLE "{table}" DROP CONSTRAINT "{name}"')

                with suppress(Exception):
                    conn.execute(stmt)


def drop_database(engine: Engine) -> None:
    """
     Drops all tables in the given database engine. This function removes foreign key
     constraints and tables in reverse dependency order to ensure that proper
     dropping of objects occurs without conflict. For MySQL/MariaDB, foreign key
    checks are disabled temporarily during the process.

     This method differs from sqlalchemy `MetaData.drop_all <https://docs.sqlalchemy.org/en/20/core/metadata.html#sqlalchemy.schema.MetaData.drop_all>`_ such that databases containing cyclic
     backreferences are also droppable.

     :param engine: The SQLAlchemy Engine instance connected to the target database
         where tables will be dropped.
     :type engine: Engine
     :return: None
    """
    metadata = MetaData()
    metadata.reflect(bind=engine)

    if not metadata.tables:
        return

    # 1. Drop FK constraints that would otherwise block table deletion.
    _drop_fk_constraints(engine, metadata.tables.keys())

    # 2. On MySQL / MariaDB it is still safest to disable FK checks entirely
    #    while the DROP TABLE statements run; other back-ends don’t need this.
    disable_fk_checks = engine.dialect.name.lower().startswith("mysql")

    with engine.begin() as conn:
        if disable_fk_checks:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))

        # Drop in reverse dependency order (children first → parents last).
        for table in reversed(metadata.sorted_tables):
            table.drop(bind=conn, checkfirst=True)

        if disable_fk_checks:
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))


class InheritanceStrategy(Enum):
    JOINED = "joined"
    SINGLE = "single"


def is_direct_subclass(cls: Type, *bases: Type) -> bool:
    """
    :param cls: The class to check.
    :param bases: The base classes to check against.

    :return: Whether 'cls' is directly derived from any of the given base classes or is the same class.
    """
    return cls in bases or (set(cls.__bases__) & set(bases))


def get_classes_of_ormatic_interface(
    interface: ModuleType,
) -> Tuple[List[Type], List[Type[AlternativeMapping]], Dict]:
    """
    Get all classes and alternative mappings of an existing ormatic interface.

    :param interface: The ormatic interface to extract the information from.
    :return: A list of classes and a list of alternative mappings used in the interface.
    """
    classes = []
    alternative_mappings = []
    classes_of_ormatic_interface = classes_of_module(interface)
    type_mappings = {}

    for cls in filter(
        lambda x: issubclass(x, DataAccessObject), classes_of_ormatic_interface
    ):
        original_class = cls.original_class()

        if issubclass(original_class, AlternativeMapping):
            alternative_mappings.append(original_class)
            classes.append(original_class.original_class())
        else:
            classes.append(original_class)

    # get the type mappings from the direct subclass of declarative base
    for cls in filter(
        lambda x: is_direct_subclass(x, DeclarativeBase), classes_of_ormatic_interface
    ):
        type_mappings.update(cls.type_mappings)

    return classes, alternative_mappings, type_mappings


def create_engine(url: Union[str, URL], **kwargs: Any) -> Engine:
    """
    Check https://docs.sqlalchemy.org/en/20/core/engines.html#sqlalchemy.create_engine for more information.

    :param url: The database URL.
    :return: An SQLAlchemy engine that uses the JSON (de)serializer from KRROOD.
    """
    return create_sqlalchemy_engine(
        url,
        json_serializer=lambda x: json.dumps(to_json(x)),
        json_deserializer=lambda x: from_json(json.loads(x)),
        **kwargs,
    )
