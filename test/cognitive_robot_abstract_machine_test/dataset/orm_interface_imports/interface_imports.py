"""
A module importing generated ORM interfaces when it is loaded, inside a function and
under ``TYPE_CHECKING``, next to an import of a module the generator does not write.
"""

from __future__ import annotations

from typing_extensions import TYPE_CHECKING

import base.orm.ormatic_interface  # reported
from base.orm import mapping
from base.orm import ormatic_interface  # reported
from base.orm.ormatic_interface import PersonDAO  # reported

from ..dataset.ormatic_interface import *  # reported

if TYPE_CHECKING:
    from base.orm.ormatic_interface import PlaceDAO


def load_person() -> PersonDAO:
    """
    Imports the interface only when it is called.
    """
    from base.orm.ormatic_interface import PersonDAO

    return PersonDAO(mapping=mapping)
