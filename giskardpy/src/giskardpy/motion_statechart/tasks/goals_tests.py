from dataclasses import dataclass

import numpy as np

import krrood.symbolic_math.symbolic_math as sm
from giskardpy.motion_statechart.graph_node import Task
from semantic_digital_twin.spatial_types import (
    Point3,
    Quaternion,
    Vector3,
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF


@dataclass
class DebugGoal(Task):
    def __post_init__(self):
        q = Quaternion(reference_frame=context.world.root)
        context.add_debug_expression("q", q)

        p = Point3(1, 0, 0, reference_frame=context.world.root)
        context.add_debug_expression("p", p)

        pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            y=1, reference_frame=context.world.root
        )
        context.add_debug_expression("pose", pose)

        v = Vector3(1, 0, 0, reference_frame=context.world.root)
        context.add_debug_expression("v", v)

        r = Quaternion(reference_frame=context.world.root).to_rotation_matrix()
        context.add_debug_expression("r", r)

        e1 = sm.Matrix.eye(3)
        context.add_debug_expression("e1", e1)

        e2 = sm.Vector(np.array([1, 2, 3]))
        context.add_debug_expression("e2", e2)

        t = context.time_symbol
        context.add_debug_expression("t", t)

        context.add_debug_expression("f", 23)


@dataclass
class CannotResolveSymbol(Task):
    joint_name: str

    def __post_init__(self):
        self.data = {}
        s = symbol_manager.register_symbol_provider("s", provider=lambda: self.data[2])
        joint: ActiveConnection1DOF = context.world.get_connection_by_name(
            self.joint_name
        )
        joint_position = joint.dof.variables.position
        self.add_equality_constraint(
            reference_velocity=1,
            equality_bound=1,
            weight=1,
            task_expression=s * joint_position,
        )
