from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict, Tuple

import numpy as np

from giskardpy.middleware import get_middleware
from giskardpy.utils.utils import create_path
from semantic_digital_twin.adapters.ros.world_synchronizer import (
    ModelSynchronizer,
    StateSynchronizer,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

if TYPE_CHECKING:
    from semantic_digital_twin.world_description.degree_of_freedom import (
        DegreeOfFreedom,
    )
    from semantic_digital_twin.world import World
    from giskardpy.model.trajectory import Trajectory
    from giskardpy.qp.qp_controller import QPController
    from giskardpy.debug_expression_manager import DebugExpressionManager
    from giskardpy.model.collision_world_syncer import (
        Collisions,
    )
    import semantic_digital_twin.spatial_types.spatial_types as cas


class GodMap:
    # %% important objects
    world: World

    # %% managers
    debug_expression_manager: DebugExpressionManager
    model_synchronizer: ModelSynchronizer
    state_synchronizer: StateSynchronizer

    # %% controller datatypes
    time: float  # real/planning time in s
    time_symbol: cas.FloatVariable
    control_cycle_counter: int
    trajectory: Trajectory
    qp_solver_solution: np.ndarray
    added_collision_checks: Dict[Tuple[PrefixedName, PrefixedName], float]
    motion_start_time: float
    hack: float
    free_variables: List[DegreeOfFreedom]

    # %% other
    tmp_folder: str

    def __getattr__(self, item):
        # automatically initialize certain attributes
        if item == "world":
            from semantic_digital_twin.world import World

            self.world = World()
        elif item == "debug_expression_manager":
            from giskardpy.debug_expression_manager import DebugExpressionManager

            self.debug_expression_manager = DebugExpressionManager()
        elif item == "time_symbol":
            self.time_symbol = symbol_manager.register_symbol_provider(
                "time", lambda: self.time
            )
        return super().__getattribute__(item)

    def to_tmp_path(self, file_name: str) -> str:
        path = god_map.tmp_folder
        return get_middleware().resolve_iri(f"{path}{file_name}")

    def write_to_tmp(self, file_name: str, file_str: str) -> str:
        """
        Writes a URDF string into a temporary file on disc. Used to deliver URDFs to PyBullet that only loads file.
        :param file_name: Name of the temporary file without any path information, e.g. 'pr2.urdfs'
        :param file_str: URDF as an XML string that shall be written to disc.
        :return: Complete path to where the urdfs was written, e.g. '/tmp/pr2.urdfs'
        """
        new_path = self.to_tmp_path(file_name)
        create_path(new_path)
        with open(new_path, "w") as f:
            f.write(file_str)
        return new_path

    def load_from_tmp(self, file_name: str):
        new_path = self.to_tmp_path(file_name)
        create_path(new_path)
        with open(new_path, "r") as f:
            loaded_file = f.read()
        return loaded_file


god_map = GodMap()
