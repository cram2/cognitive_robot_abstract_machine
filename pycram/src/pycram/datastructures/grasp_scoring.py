import os
import trimesh
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from CGAL.CGAL_Kernel import Point_3, Triangle_3
from CGAL.CGAL_AABB_tree import AABB_tree_Triangle_3_soup

@dataclass
class ScoredGrasp:
    """Represents a grasp candidate that has been evaluated and scored."""
    id: int
    pose: np.ndarray
    score: float

class GraspScorer:
    """Evaluates and ranks grasp poses using geometric checks and heuristics."""

    def __init__(
            self,
            w_normal: float = 15.0,
            w_distance: float = 5.0,
            w_clearance: float = 10.0,
            penalty_collision: float = -1000.0,
            penalty_clearance: float = -1000.0,
            penalty_unstable: float = -500.0,
            score_partial_contact: float = 5.0,
            ground_plane_z: float = 0.0
    ):
        """
        Initializes the grasp scorer with weights and penalties.
        """
        self.w_normal = w_normal
        self.w_distance = w_distance
        self.w_clearance = w_clearance
        self.penalty_collision = penalty_collision
        self.penalty_clearance = penalty_clearance
        self.penalty_unstable = penalty_unstable
        self.score_partial_contact = score_partial_contact
        self.ground_plane_z = ground_plane_z

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
            grasp_pose: np.ndarray,
            gripper_mesh: trimesh.Trimesh,
            object_mesh: trimesh.Trimesh,
            object_tree: AABB_tree_Triangle_3_soup
    ) -> float:
        """
        Calculates a quality score for a given grasp pose using geometric heuristics.
        Applies penalties for collisions and clearance, and bonuses for stability.
        """
        total_score = 0.0
        gripper_at_pose = gripper_mesh.copy()
        gripper_at_pose.apply_transform(grasp_pose)

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

        ray_origins_world = trimesh.transform_points(ray_origins_local, grasp_pose)
        ray_directions_world = trimesh.transform_points(ray_directions_local, grasp_pose, translate=False)

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
            grasp_poses: List[np.ndarray],
            gripper_mesh: trimesh.Trimesh,
            object_mesh: trimesh.Trimesh
    ) -> List[ScoredGrasp]:
        """
        Evaluates a list of grasp poses and returns a sorted list of ScoredGrasp objects 
        (best grasps first).
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
            ranked_grasps.append(ScoredGrasp(id=i, pose=grasp_pose, score=score))

        # Sort the list of scored grasps primarily by score in descending order
        ranked_grasps.sort(key=lambda x: x.score, reverse=True)
        return ranked_grasps

def load_successful_grasps_from_dataset(dataset_path: str, gripper_name: str, object_uuid: str) -> List[np.ndarray]:
    """Helper to read dataset and return list of successful grasp poses."""
    from dataset import GraspWebDatasetReader
    webdataset_reader = GraspWebDatasetReader(os.path.join(dataset_path, gripper_name))
    try:
        grasp_data = webdataset_reader.read_grasps_by_uuid(object_uuid)
        if grasp_data is None: 
            return []
        grasp_poses = np.array(grasp_data["grasps"]["transforms"])
        grasp_mask = np.array(grasp_data["grasps"]["object_in_gripper"])
        return [grasp for grasp in grasp_poses[grasp_mask]]
    except Exception as e:
        print(f"Error reading grasps for {object_uuid}: {e}")
        return []
