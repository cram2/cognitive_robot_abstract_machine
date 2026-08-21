import pytest
import numpy as np
from semantic_digital_twin.world import World
from semantic_digital_twin.datastructures.soft_trunk import SoftTrunk, SoftTrunkSection
from semantic_digital_twin.spatial_computations.ik_solver import InverseKinematicsSolver
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.soft_connections import (
    CosseratRodConnection,
    PiecewiseConstantCurvatureConnection,
)
from semantic_digital_twin.world_description.world_entity import Body


class TestSoftTrunk:

    def test_piecewise_constant_curvature_construction(self):
        """
        Tests if the PCC robot is built and its DOFs are accessible via properties.
        """
        world = World()
        # Create a uniform robot
        sections = [SoftTrunkSection(length=0.3, radius=0.02, resolution=5)] * 3

        trunk = SoftTrunk.build_piecewise_constant_curvature(world, sections)

        # Verify robot parts
        assert len(trunk.arms) == 1
        assert trunk.arms[0].end_effector is not None

        # Verify property-based DOF access
        assert len(trunk.kappa_dofs) == 3
        assert len(trunk.phi_dofs) == 3
        # Check the helper property
        assert len(trunk.piecewise_constant_curvature_sections) == 3

    def test_cosserat_rod_construction(self):
        """
        Tests if the Cosserat robot correctly initializes its 4 strain DOFs per section.
        """
        world = World()
        sections = [SoftTrunkSection(length=0.5, radius=0.02, resolution=10)] * 2

        trunk = SoftTrunk.build_cosserat(world, sections)

        # Verify 4 DOFs per section
        assert len(trunk.extension_dofs) == 2
        assert len(trunk.torsion_dofs) == 2
        assert len(trunk.bending_x_dofs) == 2
        assert len(trunk.bending_y_dofs) == 2

        # Verify extension is initialized to 1.0
        assert world.state[trunk.extension_dofs[0].id].position == 1.0

    def test_piecewise_constant_curvature_kinematics(self):
        """
        Validates the geometric accuracy of the PCC arc math.
        """
        world = World()
        # 1 section, 1.0 meters
        sections = [SoftTrunkSection(length=1.0, radius=0.02, resolution=10)]
        trunk = SoftTrunk.build_piecewise_constant_curvature(world, sections)

        # Set kappa for a 90 degree bend (r = 2/pi)
        world.state[trunk.kappa_dofs[0].id].position = np.pi / 2
        world.notify_state_change()

        # Get FK from root to tip of the arm
        fk = world.compute_forward_kinematics_np(world.root, trunk.arms[0].tip)

        # For 90 deg bend: x = radius, z = radius. radius = 1/kappa = 2/pi
        expected_val = 2 / np.pi
        np.testing.assert_allclose(fk[0, 3], expected_val, atol=1e-5)
        np.testing.assert_allclose(fk[2, 3], expected_val, atol=1e-5)

    def test_cosserat_rod_extension(self):
        """
        Validates that the Cosserat model scales correctly with extension.
        """
        world = World()
        sections = [SoftTrunkSection(length=1.0, radius=0.02, resolution=10)]
        trunk = SoftTrunk.build_cosserat(world, sections)

        # Stretch to 1.5m
        world.state[trunk.extension_dofs[0].id].position = 1.5
        world.notify_state_change()

        fk = world.compute_forward_kinematics_np(world.root, trunk.arms[0].tip)
        np.testing.assert_allclose(fk[2, 3], 1.5, atol=1e-5)

    def test_soft_trunk_ik_reachability(self):
        """
        Verifies compatibility with the framework's IK solver after refactoring.
        """
        world = World()
        sections = [SoftTrunkSection(length=0.3, radius=0.02, resolution=10)] * 3
        trunk = SoftTrunk.build_piecewise_constant_curvature(world, sections)

        # Reachable target
        target = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.2, z=0.8, reference_frame=world.root
        )

        ik_solver = InverseKinematicsSolver(world=world)

        # Solve
        ik_results = ik_solver.solve(
            root=world.root,
            tip=trunk.arms[0].tip,
            target=target,
            max_iterations=200,
            dt=0.1,
        )

        # Apply
        for dof, pos in ik_results.items():
            world.state[dof.id].position = pos
        world.notify_state_change()

        # Validate distance
        fk = world.compute_forward_kinematics_np(world.root, trunk.arms[0].tip)
        dist_error = np.linalg.norm(fk[:3, 3] - target.to_position().to_np()[:3])
        assert dist_error < 0.03

    def test_soft_trunk_semantic_annotation(self):
        """
        Ensures SoftTrunk is correctly registered as a SemanticAnnotation.
        """
        world = World()
        sections = [SoftTrunkSection(0.5, 0.02, 5)]
        SoftTrunk.build_piecewise_constant_curvature(world, sections)

        # Find by type in the world
        annotations = world.get_semantic_annotations_by_type(SoftTrunk)
        assert len(annotations) == 1
        # Check internal arm registration
        assert len(annotations[0].arms) == 1


# %% create_with_dofs honours the shared connection interface


class TestSoftConnectionFactories:
    """
    The soft connections are materialized through the same ``create_with_dofs`` entry
    point as every other connection family, so a caller that does not know the concrete
    type statically can place them.
    """

    def test_piecewise_constant_curvature_factory_applies_placement(self):
        world = World.create_with_root_body(PrefixedName("root"))
        limits = DegreeOfFreedomLimits(
            lower=DerivativeMap(position=-10.0, velocity=-10.0),
            upper=DerivativeMap(position=10.0, velocity=10.0),
        )
        kappa = DegreeOfFreedom(name=PrefixedName("kappa"), limits=limits)
        phi = DegreeOfFreedom(name=PrefixedName("phi"), limits=limits)
        tip = Body(name=PrefixedName("tip"))

        with world.modify_world():
            world.add_degree_of_freedom(kappa)
            world.add_degree_of_freedom(phi)
            world.add_body(tip)
            connection = PiecewiseConstantCurvatureConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=tip,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.4
                ),
                kappa_dof_id=kappa.id,
                phi_dof_id=phi.id,
                segment_length=1.0,
            )
            world.add_connection(connection)

        # A straight segment of length 1 offset by x=0.4 puts the tip at (0.4, 0, 1).
        root_T_tip = world.compute_forward_kinematics_np(world.root, tip)
        np.testing.assert_allclose(root_T_tip[:3, 3], [0.4, 0.0, 1.0], atol=1e-5)
        assert connection.active_dofs == [kappa, phi]

    def test_cosserat_rod_factory_applies_placement(self):
        world = World.create_with_root_body(PrefixedName("root"))
        limits = DegreeOfFreedomLimits(
            lower=DerivativeMap(position=-10.0, velocity=-10.0),
            upper=DerivativeMap(position=10.0, velocity=10.0),
        )
        strain_dofs = {
            name: DegreeOfFreedom(name=PrefixedName(name), limits=limits)
            for name in ("bending_x", "bending_y", "torsion", "extension")
        }
        tip = Body(name=PrefixedName("tip"))

        with world.modify_world():
            for dof in strain_dofs.values():
                world.add_degree_of_freedom(dof)
            world.add_body(tip)
            connection = CosseratRodConnection.create_with_dofs(
                world=world,
                parent=world.root,
                child=tip,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.4
                ),
                bending_x_dof_id=strain_dofs["bending_x"].id,
                bending_y_dof_id=strain_dofs["bending_y"].id,
                torsion_dof_id=strain_dofs["torsion"].id,
                extension_dof_id=strain_dofs["extension"].id,
                segment_length=1.0,
            )
            world.add_connection(connection)

        world.state[strain_dofs["extension"].id].position = 1.0
        world.notify_state_change()

        root_T_tip = world.compute_forward_kinematics_np(world.root, tip)
        np.testing.assert_allclose(root_T_tip[:3, 3], [0.4, 0.0, 1.0], atol=1e-5)
