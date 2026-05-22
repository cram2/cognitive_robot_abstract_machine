from py_trees.common import Status

import robokudo.world as rk_world
from robokudo.annotators.world_descriptor_bootstrap import (
    WorldDescriptorBootstrapAnnotator,
)
from robokudo.world_descriptor import PredefinedObject
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Region


class TestWorldDescriptorBootstrapAnnotator:
    def test_augment_world_keeps_existing_frame_references(self):
        rk_world.init_world_with_entity_tracker()
        rk_world.setup_world_for_camera_frame(world_frame="map", camera_frame="camera")

        runtime_world = rk_world.world_instance()
        camera_body = runtime_world.get_body_by_name("camera")
        map_body = runtime_world.get_body_by_name("map")
        camera_body_id = camera_body.id
        map_body_id = map_body.id

        transform = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=0.1,
            pos_y=0.2,
            pos_z=0.3,
            quat_x=0.0,
            quat_y=0.0,
            quat_z=0.0,
            quat_w=1.0,
            child_frame=camera_body,
            reference_frame=map_body,
        )

        annotator = WorldDescriptorBootstrapAnnotator()
        result = annotator.update()

        assert result is Status.SUCCESS
        assert runtime_world.get_body_by_name("camera").id == camera_body_id
        assert runtime_world.get_body_by_name("map").id == map_body_id
        assert transform.child_frame.id == camera_body_id
        assert transform.reference_frame.id == map_body_id

        predefined_objects = runtime_world.get_semantic_annotations_by_type(
            PredefinedObject
        )
        predefined_object_names = {
            annotation.body.name.name
            for annotation in predefined_objects
            if annotation.body is not None
        }
        assert {"cereal", "milk"}.issubset(predefined_object_names)

        region_names = {
            str(region.name)
            for region in runtime_world.get_kinematic_structure_entity_by_type(Region)
        }
        assert "kitchen_island" in region_names

    def test_update_is_idempotent_for_same_world_instance(self):
        rk_world.init_world_with_entity_tracker()
        annotator = WorldDescriptorBootstrapAnnotator()

        first = annotator.update()
        second = annotator.update()
        runtime_world = rk_world.world_instance()

        assert first is Status.SUCCESS
        assert second is Status.SUCCESS
        assert len(runtime_world.get_bodies_by_name("cereal")) == 1
        assert len(runtime_world.get_bodies_by_name("milk")) == 1
        assert (
            len(
                runtime_world.get_kinematic_structure_entities_by_name("kitchen_island")
            )
            == 1
        )
