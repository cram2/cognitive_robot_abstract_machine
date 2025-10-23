import numpy as np

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.tasks.task import Task
from giskardpy.utils.decorators import validated_dataclass
from semantic_digital_twin.spatial_types.symbol_manager import symbol_manager
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


@validated_dataclass
class DebugGoal(Task):
    def __post_init__(self):
        q = cas.Quaternion(reference_frame=god_map.world.root)
        god_map.debug_expression_manager.add_debug_expression("q", q)

        p = cas.Point3(1, 0, 0, reference_frame=god_map.world.root)
        god_map.debug_expression_manager.add_debug_expression("p", p)

        pose = cas.TransformationMatrix.from_xyz_rpy(
            y=1, reference_frame=god_map.world.root
        )
        god_map.debug_expression_manager.add_debug_expression("pose", pose)

        v = cas.Vector3(1, 0, 0, reference_frame=god_map.world.root)
        god_map.debug_expression_manager.add_debug_expression("v", v)

        r = cas.Quaternion(reference_frame=god_map.world.root).to_rotation_matrix()
        god_map.debug_expression_manager.add_debug_expression("r", r)

        e1 = cas.Expression(np.eye(3))
        god_map.debug_expression_manager.add_debug_expression("e1", e1)

        e2 = cas.Expression(np.array([1, 2, 3]))
        god_map.debug_expression_manager.add_debug_expression("e2", e2)

        t = god_map.time_symbol
        god_map.debug_expression_manager.add_debug_expression("t", t)

        god_map.debug_expression_manager.add_debug_expression("f", 23)


@validated_dataclass
class CannotResolveSymbol(Task):
    joint_name: str

    def __post_init__(self):
        self.data = {}
        s = symbol_manager.register_symbol_provider("s", provider=lambda: self.data[2])
        joint: ActiveConnection1DOF = god_map.world.get_connection_by_name(
            self.joint_name
        )
        joint_position = joint.dof.symbols.position
        self.add_equality_constraint(
            reference_velocity=1,
            equality_bound=1,
            weight=1,
            task_expression=s * joint_position,
        )
