import pytest
import numpy as np
import trimesh
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

from pycram.datastructures.grasp_scoring import GraspScorer, load_successful_grasps_from_dataset
from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Quaternion

import uuid
from semantic_digital_twin.world_description.world_entity import Body
from pycram.datastructures.grasp import GraspPose
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.ormatic.utils import create_engine
from sqlalchemy.orm import Session
from pycram.orm.ormatic_interface import Base as PycramBase


scorer = GraspScorer()

@pytest.fixture
def object_mesh():
    """Creates a simple 10cm cubic box positioned above the ground (z=0.05)."""
    mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    mesh.apply_translation([0, 0, 0.05])
    return mesh

@pytest.fixture
def object_tree(object_mesh):
    """Creates a CGAL AABB tree for the dummy object mesh."""
    cgal_triangles = scorer._trimesh_to_cgal_triangles(object_mesh)
    return AABB_tree_Triangle_3_soup(cgal_triangles)

@pytest.fixture
def gripper_mesh():
    """Creates a simple parallel-jaw gripper out of two box-shaped fingers."""
    finger1 = trimesh.creation.box(extents=[0.02, 0.02, 0.1])
    finger1.apply_translation([0, 0.06, 0])
    
    finger2 = trimesh.creation.box(extents=[0.02, 0.02, 0.1])
    finger2.apply_translation([0, -0.06, 0])
    
    return trimesh.util.concatenate([finger1, finger2])


def test_trimesh_to_cgal_triangles(object_mesh):
    """Validates the Trimesh to CGAL triangle conversion."""
    cgal_triangles = scorer._trimesh_to_cgal_triangles(object_mesh)
    
    # A box should have exactly 12 triangular faces
    assert len(cgal_triangles) == len(object_mesh.faces)
    assert len(cgal_triangles) == 12

def test_calculate_grasp_score_collision(gripper_mesh, object_mesh, object_tree):
    """Tests if the GraspScorer properly detects collisions and penalizes them."""
    # Translate gripper such that one of the fingers intersects with the object.
    # The object occupies y in [-0.05, 0.05]. Moving the gripper by 0.02 in y
    # puts finger2 at y=-0.04, causing an intersection.
    grasp_pose = Pose(Point3(0.0, 0.02, 0.05), Quaternion(1.0, 0.0, 0.0, 0.0))
    
    score = scorer.calculate_grasp_score(grasp_pose, gripper_mesh, object_mesh, object_tree)
    assert score == pytest.approx(scorer.penalty_collision)

def test_calculate_grasp_score_clearance(gripper_mesh, object_mesh, object_tree):
    """Tests if the GraspScorer detects when the gripper dives below the ground plane."""
    # Submerge the gripper below the ground plane z=0
    grasp_pose = Pose(Point3(0.0, 0.0, -0.5), Quaternion(1.0, 0.0, 0.0, 0.0))
    
    score = scorer.calculate_grasp_score(grasp_pose, gripper_mesh, object_mesh, object_tree)
    assert score == pytest.approx(scorer.penalty_clearance)

def test_calculate_grasp_score_good_grasp(gripper_mesh, object_mesh, object_tree):
    """Tests the stability analysis on a completely valid grasp without collisions."""
    # Position the gripper perfectly around the object. 
    # Fingers are at y=+-0.06, which clears the object (y ends at +-0.05).
    grasp_pose = Pose(Point3(0.0, 0.0, 0.05), Quaternion(1.0, 0.0, 0.0, 0.0))
    
    score = scorer.calculate_grasp_score(grasp_pose, gripper_mesh, object_mesh, object_tree)
    # Since the internal rays from y=+-0.06 pointing inwards will intersect the 
    # object and give normal and distance scores, the score should be positive.
    assert score > 0.0

def test_rank_grasps(gripper_mesh, object_mesh):
    """Tests whether grasping poses are correctly ranked by score."""
    pose_good = Pose(Point3(0.0, 0.0, 0.05), Quaternion(1.0, 0.0, 0.0, 0.0))
    pose_collision = Pose(Point3(0.0, 0.02, 0.05), Quaternion(1.0, 0.0, 0.0, 0.0))
    pose_clearance = Pose(Point3(0.0, 0.0, -0.5), Quaternion(1.0, 0.0, 0.0, 0.0))
    
    grasps = [pose_clearance, pose_collision, pose_good]
    
    # Ranks grasps descending by score
    ranked = scorer.rank_grasps(grasps, gripper_mesh, object_mesh)
    
    assert len(ranked) == 3
    # The structurally sound pose should be the champion and rank first
    # Floating-Point Safe Assertion: Convert cleanly back to raw matrices and evaluate tolerances
    assert np.allclose(
        ranked[0].pose.to_homogeneous_matrix().to_np(), 
        pose_good.to_homogeneous_matrix().to_np()
    )
    assert ranked[0].score > 0.0
    
    # Verify UUID id logic
    assert isinstance(ranked[0].id, str)
    assert len(ranked[0].id) == 36 # length of a standard UUID4 representation
    
    # Check that heavily penalized grasps rank at the bottom
    assert ranked[-1].score <= max(scorer.penalty_collision, scorer.penalty_clearance)

def test_load_successful_grasps_from_dataset(tmp_path):
    """Tests the ORMatic dataset loader for grasp scoring."""

    db_path = tmp_path / "test_dataset.sqlite"
    db_uri = f"sqlite+pysqlite:///{db_path}"
    
    engine = create_engine(db_uri)
    PycramBase.metadata.create_all(engine)

    body = Body(name="test_body")
    test_uuid = body.id
    
    grasp = GraspPose(
        position=Point3(1.0, 2.0, 3.0),
        orientation=Quaternion(1.0, 0.0, 0.0, 0.0),
        reference_frame=body
    )

    with Session(engine) as session:
        session.add(to_dao(body))
        session.add(to_dao(grasp))
        session.commit()

    # Test loading the existing grasp
    grasps = load_successful_grasps_from_dataset(str(db_path), "fake_gripper", str(test_uuid))
    
    assert len(grasps) == 1
    assert grasps[0].to_position().x == 1.0
    assert grasps[0].to_position().y == 2.0
    assert grasps[0].to_position().z == 3.0

    # Test with non-existing UUID
    empty_grasps = load_successful_grasps_from_dataset(str(db_path), "fake_gripper", str(uuid.uuid4()))
    assert len(empty_grasps) == 0
