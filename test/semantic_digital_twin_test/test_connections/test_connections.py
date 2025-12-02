from numpy.testing import assert_allclose
from semantic_digital_twin.world_description.connection_properties import JointDynamics


class TestJointProperty:
    def test_default_values(self):
        joint_dynamics = JointDynamics()
        assert_allclose(joint_dynamics.armature, 0.0)
        assert_allclose(joint_dynamics.dry_friction, 0.0)
        assert_allclose(joint_dynamics.damping, 0.0)

    def test_custom_values(self):
        armature = 1.5
        dry_friction = 0.2
        damping = 0.05
        joint_dynamics = JointDynamics(
            armature=armature, dry_friction=dry_friction, damping=damping
        )
        assert_allclose(joint_dynamics.armature, armature)
        assert_allclose(joint_dynamics.dry_friction, dry_friction)
        assert_allclose(joint_dynamics.damping, damping)

        joint_prop_dict = joint_dynamics.__dict__
        assert_allclose(joint_prop_dict["armature"], armature)
        assert_allclose(joint_prop_dict["dry_friction"], dry_friction)
        assert_allclose(joint_prop_dict["damping"], damping)
