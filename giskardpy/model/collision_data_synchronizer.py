from __future__ import annotations

import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from itertools import product, combinations_with_replacement
from typing import (
    List,
    Dict,
    Optional,
    Tuple,
    Iterable,
    Set,
    DefaultDict,
    Callable,
    TYPE_CHECKING,
)

import numpy as np
from line_profiler import profile
from lxml import etree

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import UnknownGroupException, UnknownLinkException
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.collision_data_synchronizer import CollisionDataSynchronizer
from giskardpy.model.collision_detector import CollisionDetector
from giskardpy.model.collision_matrix_manager import CollisionMatrixManager
from giskardpy.qp.free_variable import FreeVariable
from semantic_digital_twin.world_description.connections import ActiveConnection
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types.derivatives import Derivatives
from semantic_digital_twin.spatial_types.symbol_manager import symbol_manager
from semantic_digital_twin.utils import copy_lru_cache
from semantic_digital_twin.world_description.world_entity import Body, Connection


class CollisionDataSynchronizer:
    """
    Handles collision data lifecycle and caching.
    """

    closest_points: Collisions = field(default_factory=Collisions)
