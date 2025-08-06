from __future__ import annotations

import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from itertools import product, combinations_with_replacement
from typing import List, Dict, Optional, Tuple, Iterable, Set, DefaultDict, Callable, TYPE_CHECKING

import numpy as np
from line_profiler import profile
from lxml import etree

import semantic_world.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import UnknownGroupException, UnknownLinkException
from giskardpy.god_map import god_map
from giskardpy.middleware import get_middleware
from giskardpy.model.collision_data_synchronizer import CollisionDataSynchronizer
from giskardpy.model.collision_detector import CollisionDetector
from giskardpy.model.collision_matrix_manager import CollisionMatrixManager
from giskardpy.qp.free_variable import FreeVariable
from semantic_world.connections import ActiveConnection
from semantic_world.robots import AbstractRobot
from semantic_world.spatial_types.derivatives import Derivatives
from semantic_world.spatial_types.symbol_manager import symbol_manager
from semantic_world.utils import copy_lru_cache
from semantic_world.world_entity import Body, Connection


class CollisionDataSynchronizer:
    """
    Handles collision data lifecycle and caching.
    """

    closest_points: Collisions = field(default_factory=Collisions)

    @profile
    def get_external_collision_data(self) -> np.ndarray:
        offset = 0
        for link_name, max_idx in self.external_monitored_links.items():
            collisions = self.closest_points.get_external_collisions(link_name)

            for idx in range(max_idx + 1):
                np.copyto(self.external_collision_data[offset:offset + 8], collisions[idx].external_data)
                offset += 8

            self.external_collision_data[offset] = self.closest_points.get_number_of_external_collisions(link_name)
            offset += 1

        return self.external_collision_data

    @profile
    def get_self_collision_data(self) -> np.ndarray:

        offset = 0
        for (link_a, link_b), max_idx in self.self_monitored_links.items():
            collisions = self.closest_points.get_self_collisions(link_a, link_b)

            for idx in range(max_idx + 1):
                np.copyto(self.self_collision_data[offset:offset + 10], collisions[idx].self_data)
                offset += 10

            self.self_collision_data[offset] = self.closest_points.get_number_of_self_collisions(link_a, link_b)
            offset += 1

        return self.self_collision_data