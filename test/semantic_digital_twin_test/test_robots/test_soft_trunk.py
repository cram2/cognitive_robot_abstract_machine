import pytest
import numpy as np
from semantic_digital_twin.world import World
from semantic_digital_twin.robots.soft_trunk import SoftTrunk


class TestSoftTrunk:
    def test_piecewise_constant_curvature_construction(self):
        """
        Test that a SoftTrunk can be constructed with the PCC model
        and that it registers correctly in the world.
        """
        world = World()
        # Create a robot with 1 section and 5 segments
        trunk, kappas, phis = SoftTrunk.build_piecewise_constant_curvature(
            world, num_sections=1, segments_per_section=5
        )
        # Check that the manipulator chain was properly registered
        assert len(trunk.manipulator_chains) == 1
        assert trunk.manipulator_chains[0].tip is not None

    def test_piecewise_constant_curvature_kinematics(self):
        """
        Test the forward kinematics of a single PCC segment against expected values.
        Creates a 1-meter segment and bends it exactly 90 degrees.
        Checks that the tip is at (2/pi, 0, 2/pi) in the base frame,
        which is the expected position for a quarter circle arc of radius 1.
        """
        world = World()
        # Build 1 section, 1 segment, length 1.0
        trunk, kappas, phis = SoftTrunk.build_piecewise_constant_curvature(
            world, num_sections=1, segments_per_section=1, total_length=1.0
        )

        # Set kappa for a 90 degree bend.
        # Plane phi is 0 by default (bending in X-Z plane)
        world.state[kappas[0].id].position = np.pi / 2
        world.notify_state_change()

        # Get the transform of the tip relative to the root body
        tip_body = trunk.manipulator_chains[0].tip
        fk = world.compute_forward_kinematics_np(world.root, tip_body)

        expected_val = 2 / np.pi

        # fk[0, 3] is X translation, fk[2, 3] is Z translation
        np.testing.assert_allclose(fk[0, 3], expected_val, atol=1e-5)
        np.testing.assert_allclose(fk[2, 3], expected_val, atol=1e-5)

    def test_cosserat_rod_extension(self):
        """
        Test the extension of a Cosserat rod segment by modifying the vz DOF.
        Validates the Numerical Integration (RK4) and stretching capability.
        Creates a 1-meter rod and extends it to 1.5 meters.
        Checks that the tip is at (0, 0, 1.5) in the base frame when there is no bending (kappa = 0).
        """
        world = World()
        # Build 1 section, 10 integration steps, length 1.0
        trunk, ux, uy, uz, vz = SoftTrunk.build_cosserat(
            world, num_sections=1, segments_per_section=10, total_length=1.0
        )

        # Set extension to 1.5 (50% stretch)
        world.state[vz[0].id].position = 1.5
        world.notify_state_change()

        # Get tip transform
        tip_body = trunk.manipulator_chains[0].tip
        fk = world.compute_forward_kinematics_np(world.root, tip_body)

        # Tip should be exactly at z=1.5
        np.testing.assert_allclose(fk[2, 3], 1.5, atol=1e-5)

    def test_soft_trunk_semantic_annotation(self):
        """
        Test that the SoftTrunk and its manipulator chain are properly annotated in the world.
        """
        world = World()
        trunk, kappas, phis = SoftTrunk.build_piecewise_constant_curvature(world)

        # Verify the robot is found in semantic annotations
        annotated = world.get_semantic_annotations_by_type(SoftTrunk)
        assert len(annotated) == 1
        # Verify the manipulator chain is present
        assert len(annotated[0].manipulator_chains) == 1
        assert annotated[0].manipulator_chains[0].name.name == "arm"
