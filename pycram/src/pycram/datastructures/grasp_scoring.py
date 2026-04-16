import os
import trimesh
import numpy as np
import uuid
from dataclasses import dataclass, field
from typing import List, Optional
from CGAL.CGAL_Kernel import Point_3, Triangle_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

from semantic_digital_twin.spatial_types.spatial_types import Pose, Point3, Quaternion

@dataclass
class ScoredGrasp:
    """
    Represents a grasp candidate that has been evaluated and scored.
    """
    pose: Pose
    """A 4x4 transformation matrix representing the grasp pose."""
    
    score: float
    """The calculated quality score for this grasp."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """A unique identifier for the grasp."""

@dataclass
class GraspScorer:
    """Evaluates and ranks grasp poses using geometric checks and heuristics."""
    
    w_normal: float = 15.0
    """Weight assigned to well-aligned opposing normals."""
    
    w_distance: float = 5.0
    """Weight assigned to the magnitude of grip distance between contacts."""
    
    w_clearance: float = 10.0
    """Weight for maintaining safe clearance above the ground plane."""
    
    penalty_collision: float = -1000.0
    """Penalty applied when the gripper collides with the object mesh."""
    
    penalty_clearance: float = -1000.0
    """Penalty applied when the gripper hits or dips below the ground plane."""
    
    penalty_unstable: float = -500.0
    """Penalty applied when an unstable (e.g. fewer than 2) contact points are found."""
    
    score_partial_contact: float = 5.0
    """Constant score applied when only a single contact point is identified."""
    
    ground_plane_z: float = 0.0
    """The predefined absolute Z-axis bounds considered as the ground."""

    def _trimesh_to_cgal_triangles(self, mesh: trimesh.Trimesh) -> List[Triangle_3]:
        """Converts a Trimesh object into a list of CGAL Triangle_3 objects."""
        triangles = []
        for face in mesh.faces:
            p1_coords, p2_coords, p3_coords = mesh.vertices[face]
            p1 = Point_3(p1_coords[0], p1_coords[1], p1_coords[2])
            p2 = Point_3(p2_coords[0], p2_coords[1], p2_coords[2])
            p3 = Point_3(p3_coords[0], p3_coords[1], p3_coords[2])
            triangles.append(Triangle_3(p1, p2, p3))
        return triangles

    def calculate_grasp_score(
            self,
            grasp_pose: Pose,
            gripper_mesh: trimesh.Trimesh,
            object_mesh: trimesh.Trimesh,
            object_tree: AABB_tree_Triangle_3_soup
    ) -> float:
        """
        Calculates a quality score for a given grasp pose using geometric heuristics.
        Applies penalties for collisions and clearance, and bonuses for stability.
        
        :param grasp_pose: A semantic_digital_twin Pose representing the gripper pose.
        :param gripper_mesh: The 3D mesh of the gripper.
        :param object_mesh: The 3D mesh of the target object.
        :param object_tree: A CGAL AABB tree of the object for fast collision checking.
        :return: The calculated float score for the grasp.
        """
        total_score = 0.0
        grasp_pose_matrix = grasp_pose.to_homogeneous_matrix().to_np()
        
        gripper_at_pose = gripper_mesh.copy()
        gripper_at_pose.apply_transform(grasp_pose_matrix)

        # --- 1. Collision Check ---
        gripper_cgal_triangles = self._trimesh_to_cgal_triangles(gripper_at_pose)
        if any(object_tree.do_intersect(tri) for tri in gripper_cgal_triangles):
            total_score += self.penalty_collision

        # --- 2. Clearance Check ---
        min_gripper_z = gripper_at_pose.bounds[0][2]
        if min_gripper_z < self.ground_plane_z:
            total_score += self.penalty_clearance

        # If score is already heavily penalized, no need to check stability
        if total_score < -1:
            return total_score

        # --- 3. Stability Analysis (Contact Points & Normals) ---
        ray_origins_local = np.array([[0.0, 0.06, 0.0], [0.0, -0.06, 0.0]])
        ray_directions_local = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])

        ray_origins_world = trimesh.transform_points(ray_origins_local, grasp_pose_matrix)
        ray_directions_world = trimesh.transform_points(ray_directions_local, grasp_pose_matrix, translate=False)

        locations, index_ray, index_tri = object_mesh.ray.intersects_location(
            ray_origins=ray_origins_world, ray_directions=ray_directions_world
        )

        # Grade the contact instead of pass/fail
        if len(locations) == 2:
            # IDEAL CASE: Two contacts found, calculate a full, detailed score.
            contact_p1, contact_p2 = locations
            normal_p1 = object_mesh.face_normals[index_tri[0]]
            normal_p2 = object_mesh.face_normals[index_tri[1]]

            normal_score = max(0.0, -np.dot(normal_p1, normal_p2))
            distance_score = np.linalg.norm(contact_p1 - contact_p2)
            clearance_score = min_gripper_z

            positive_score = (self.w_normal * normal_score) + (self.w_distance * distance_score) + (self.w_clearance * clearance_score)
            total_score += positive_score

        elif len(locations) == 1:
            # GOOD ENOUGH CASE: One contact found. Give a small, fixed bonus.
            total_score += self.score_partial_contact
        else:
            # WORST CASE: A complete miss. Apply the instability penalty.
            total_score += self.penalty_unstable

        return total_score

    def rank_grasps(
            self,
            grasp_poses: List[Pose],
            gripper_mesh: trimesh.Trimesh,
            object_mesh: trimesh.Trimesh
    ) -> List[ScoredGrasp]:
        """
        Evaluates a list of grasp poses and returns a sorted list of ScoredGrasp objects 
        (best grasps first).
        
        :param grasp_poses: A list of Pose objects representing candidate poses.
        :param gripper_mesh: The 3D mesh of the gripper.
        :param object_mesh: The 3D mesh of the target object.
        :return: A list of ScoredGrasp objects, sorted in descending order by score.
        """
        object_cgal_triangles = self._trimesh_to_cgal_triangles(object_mesh)
        tree_object = AABB_tree_Triangle_3_soup(object_cgal_triangles)

        ranked_grasps = []
        for i, grasp_pose in enumerate(grasp_poses):
            score = self.calculate_grasp_score(
                grasp_pose=grasp_pose,
                gripper_mesh=gripper_mesh,
                object_mesh=object_mesh,
                object_tree=tree_object
            )
            ranked_grasps.append(ScoredGrasp(pose=grasp_pose, score=score))

        # Sort the list of scored grasps primarily by score in descending order
        ranked_grasps.sort(key=lambda x: x.score, reverse=True)
        return ranked_grasps

def load_successful_grasps_from_dataset(dataset_path: str, gripper_name: str, object_uuid: str) -> List[Pose]:
    """
    Helper to read dataset and return a list of successful grasp poses.
    
    :param dataset_path: The root directory path of the dataset.
    :param gripper_name: The name of the gripper used in the dataset.
    :param object_uuid: The unique identifier for the target object.
    :return: A list of Pose objects representing successful grasp poses. Returns an empty list if no grasps are found or an error occurs.
    """
    from dataset import GraspWebDatasetReader
    webdataset_reader = GraspWebDatasetReader(os.path.join(dataset_path, gripper_name))
    try:
        grasp_data = webdataset_reader.read_grasps_by_uuid(object_uuid)
        if grasp_data is None: 
            return []
        grasp_poses_matrices = np.array(grasp_data["grasps"]["transforms"])
        grasp_mask = np.array(grasp_data["grasps"]["object_in_gripper"])
        
        import transforms3d
        grasps = []
        for matrix in grasp_poses_matrices[grasp_mask]:
            pos = matrix[:3, 3]
            quat = transforms3d.quaternions.mat2quat(matrix[:3, :3]) # [w, x, y, z]
            grasps.append(Pose(Point3(*pos), Quaternion(*quat)))
            
        return grasps
    except Exception as e:
        print(f"Error reading grasps for {object_uuid}: {e}")
        return []
