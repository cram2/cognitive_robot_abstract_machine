import pytest
import numpy as np
from semantic_digital_twin.world import World
from semantic_digital_twin.robots.soft_trunk import SoftTrunk, SoftTrunkSection
from semantic_digital_twin.spatial_computations.ik_solver import InverseKinematicsSolver
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


class TestSoftTrunk:

    def test_piecewise_constant_curvature_construction(self):
        """Tests if the PCC robot is built and its DOFs are accessible via properties."""
        world = World()
        # Create a uniform robot
        sections = [SoftTrunkSection(length=0.3, radius=0.02, resolution=5)] * 3

        trunk = SoftTrunk.build_piecewise_constant_curvature(world, sections)

        # Verify robot parts
        assert len(trunk.manipulator_chains) == 1
        assert trunk.manipulator_chains[0].manipulator is not None

        # Verify property-based DOF access
        assert len(trunk.kappa_dofs) == 3
        assert len(trunk.phi_dofs) == 3
        assert len(trunk.pcc_sections) == 3  # Check the (kappa, phi) zipped property

    def test_cosserat_rod_construction(self):
        """Tests if the Cosserat robot correctly initializes its 4 strain DOFs per section."""
        world = World()
        sections = [SoftTrunkSection(length=0.5, radius=0.02, resolution=10)] * 2

        trunk = SoftTrunk.build_cosserat(world, sections)

        assert len(trunk.extension_dofs) == 2
        assert len(trunk.torsion_dofs) == 2
        assert len(trunk.bending_x_dofs) == 2

        # Extension should be initialized to 1.0 (rest length)
        assert world.state[trunk.extension_dofs[0].id].position == 1.0

    def test_piecewise_constant_curvature_kinematics(self):
        """Validates the geometric accuracy of the PCC arc math."""
        world = World()
        # 1 section, 1.0 meters
        sections = [SoftTrunkSection(length=1.0, radius=0.02, resolution=1)]
        trunk = SoftTrunk.build_piecewise_constant_curvature(world, sections)

        # Set kappa for a 90 degree bend
        world.state[trunk.kappa_dofs[0].id].position = np.pi / 2
        world.notify_state_change()

        # Compute FK relative to root
        fk = world.compute_forward_kinematics_np(
            world.root, trunk.manipulator_chains[0].tip
        )

        # For 90 deg bend: x = radius, z = radius. radius = 1/kappa = 2/pi
        expected_val = 2 / np.pi
        np.testing.assert_allclose(fk[0, 3], expected_val, atol=1e-5)
        np.testing.assert_allclose(fk[2, 3], expected_val, atol=1e-5)

    def test_cosserat_rod_extension(self):
        """Validates that the Cosserat model correctly handles stretching."""
        world = World()
        sections = [SoftTrunkSection(length=1.0, radius=0.02, resolution=10)]
        trunk = SoftTrunk.build_cosserat(world, sections)

        # Stretch to 1.5m
        world.state[trunk.extension_dofs[0].id].position = 1.5
        world.notify_state_change()

        fk = world.compute_forward_kinematics_np(
            world.root, trunk.manipulator_chains[0].tip
        )
        np.testing.assert_allclose(fk[2, 3], 1.5, atol=1e-5)

    def test_soft_trunk_ik_reachability(self):
        """Verifies compatibility with the framework's built-in Inverse Kinematics solver."""
        world = World()
        sections = [SoftTrunkSection(length=0.3, radius=0.02, resolution=10)] * 3
        trunk = SoftTrunk.build_piecewise_constant_curvature(world, sections)

        # Define a reachable target pose
        target_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.2, z=0.8, reference_frame=world.root
        )

        solver = InverseKinematicsSolver(world=world)

        # Solve
        ik_results = solver.solve(
            root=world.root,
            tip=trunk.manipulator_chains[0].tip,
            target=target_pose,
            max_iterations=200,
            dt=0.1,
        )

        # Apply results
        for dof, position in ik_results.items():
            world.state[dof.id].position = position
        world.notify_state_change()

        # Check final distance error
        fk = world.compute_forward_kinematics_np(
            world.root, trunk.manipulator_chains[0].tip
        )
        dist_error = np.linalg.norm(fk[:3, 3] - target_pose.to_position().to_np()[:3])
        assert dist_error < 0.03

    def test_soft_trunk_semantic_annotation(self):
        """Ensures the robot is searchable in the semantic world model."""
        world = World()
        sections = [SoftTrunkSection(0.5, 0.02, 5)]
        SoftTrunk.build_piecewise_constant_curvature(world, sections)

        # Find by type
        annotated_robots = world.get_semantic_annotations_by_type(SoftTrunk)
        assert len(annotated_robots) == 1
        assert annotated_robots[0].manipulator_chains[0].name.name == "arm"
