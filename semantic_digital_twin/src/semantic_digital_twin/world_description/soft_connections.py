import casadi as cs
from semantic_digital_twin.world_description.world_entity import Connection
from semantic_digital_twin.spatial_types.spatial_types import HomogeneousTransformationMatrix

class SoftPCCConnection(Connection):
    """
    A continuum connection for Piecewise Constant Curvature.
    """
    def __init__(self, parent, child, kappa_dof, phi_dof, length: float, name=None):
        self.kappa_dof = kappa_dof
        self.phi_dof = phi_dof
        self.segment_length = length
        
        # Build the PCC Symbolic Matrix (CasADi SX)
        k = kappa_dof.variables.position.casadi_sx
        p = phi_dof.variables.position.casadi_sx
        L = length
        theta = k * L
        
        it_is_straight = cs.fabs(k) < 1e-8
        
        # Position formulas
        px = cs.if_else(it_is_straight, 0, (cs.cos(p) * (1 - cs.cos(theta))) / (k + 1e-12))
        py = cs.if_else(it_is_straight, 0, (cs.sin(p) * (1 - cs.cos(theta))) / (k + 1e-12))
        pz = cs.if_else(it_is_straight, L, cs.sin(theta) / (k + 1e-12))
        
        # Rotation Matrix components
        c_p = cs.cos(p); s_p = cs.sin(p); c_t = cs.cos(theta); s_t = cs.sin(theta)
        r11 = c_p**2 * (c_t - 1) + 1; r12 = s_p * c_p * (c_t - 1); r13 = c_p * s_t
        r21 = s_p * c_p * (c_t - 1);   r22 = c_p**2 * (1 - c_t) + c_t; r23 = s_p * s_t
        r31 = -c_p * s_t;              r32 = -s_p * s_t;            r33 = c_t

        pcc_matrix = cs.vertcat(
            cs.horzcat(r11, r12, r13, px),
            cs.horzcat(r21, r22, r23, py),
            cs.horzcat(r31, r32, r33, pz),
            cs.horzcat(0,   0,   0,   1)
        )

        # Set required internal attributes manually
        self.parent = parent
        self.child = child
        self.name = name
        self.parent_T_connection_expression = HomogeneousTransformationMatrix()
        self.connection_T_child_expression = HomogeneousTransformationMatrix(pcc_matrix)
        
        self._active_dofs = [kappa_dof, phi_dof]
        
        self._kinematics = HomogeneousTransformationMatrix()

    @property
    def active_dofs(self):
        return self._active_dofs

    def __post_init__(self):
        # Prevent any parent validation logic from running
        pass