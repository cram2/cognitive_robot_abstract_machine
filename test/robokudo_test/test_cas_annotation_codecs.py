import numpy as np

from robokudo.io import cas_annotation_codecs
from robokudo.types.annotation import Shape, Sphere, Cylinder
from semantic_digital_twin.world_description.geometry import (
    Box as SemDTBox,
    Cylinder as SemDTCylinder,
    Sphere as SemDTSphere,
)


def test_deserialize_annotations_none_returns_empty_list():
    assert cas_annotation_codecs.deserialize_annotations(None) == []


def test_deserialize_annotations_forwards_kwargs_for_lists(monkeypatch):
    calls: list[tuple[dict, dict]] = []

    def fake_from_json(item, **kwargs):
        calls.append((item, kwargs))
        return {"item": item, "kwargs": kwargs}

    monkeypatch.setattr(cas_annotation_codecs, "from_json", fake_from_json)

    payload = [{"id": 1}, {"id": 2}]
    result = cas_annotation_codecs.deserialize_annotations(payload, tracker="active")

    assert len(result) == 2
    assert [call[0] for call in calls] == payload
    assert all(call[1] == {"tracker": "active"} for call in calls)


def test_deserialize_annotations_forwards_kwargs_for_single_item(monkeypatch):
    calls: list[tuple[dict, dict]] = []

    def fake_from_json(item, **kwargs):
        calls.append((item, kwargs))
        return {"item": item, "kwargs": kwargs}

    monkeypatch.setattr(cas_annotation_codecs, "from_json", fake_from_json)

    payload = {"id": 3}
    result = cas_annotation_codecs.deserialize_annotations(payload, tracker="active")

    assert len(result) == 1
    assert calls == [(payload, {"tracker": "active"})]
    assert result[0]["item"] == payload


def test_numpy_scalar_json_serializer_roundtrip():
    scalar = np.int16(42)
    encoded = cas_annotation_codecs.NumpyScalarJSONSerializer.to_json(scalar)
    restored = cas_annotation_codecs.NumpyScalarJSONSerializer.from_json(
        encoded, np.generic
    )

    assert encoded["dtype"] == "int16"
    assert restored == scalar.item()


def test_shape_annotation_serialization_roundtrip_preserves_metadata():
    shape = Shape(source="shape_annotator", inliers=[1, 2, 3], geometry=SemDTBox())
    shape.geometry.scale.x = 0.31

    serialized = cas_annotation_codecs.serialize_annotations([shape])
    restored = cas_annotation_codecs.deserialize_annotations(serialized)

    assert len(restored) == 1
    restored_shape = restored[0]
    assert isinstance(restored_shape, Shape)
    assert restored_shape.source == "shape_annotator"
    assert restored_shape.inliers == [1, 2, 3]
    assert isinstance(restored_shape.geometry, SemDTBox)
    assert restored_shape.geometry.scale.x == 0.31


def test_sphere_annotation_serialization_roundtrip_preserves_radius():
    sphere = Sphere(
        source="sphere_annotator",
        inliers=[9, 8],
        geometry=SemDTSphere(radius=0.27),
    )

    serialized = cas_annotation_codecs.serialize_annotations([sphere])
    restored = cas_annotation_codecs.deserialize_annotations(serialized)

    assert len(restored) == 1
    restored_sphere = restored[0]
    assert isinstance(restored_sphere, Sphere)
    assert restored_sphere.source == "sphere_annotator"
    assert restored_sphere.inliers == [9, 8]
    assert isinstance(restored_sphere.geometry, SemDTSphere)
    assert restored_sphere.geometry.radius == 0.27


def test_cylinder_annotation_serialization_roundtrip_preserves_dimensions():
    cylinder = Cylinder(
        source="cylinder_annotator",
        inliers=[3, 4],
        geometry=SemDTCylinder(width=0.2, height=0.6),
    )

    serialized = cas_annotation_codecs.serialize_annotations([cylinder])
    restored = cas_annotation_codecs.deserialize_annotations(serialized)

    assert len(restored) == 1
    restored_cylinder = restored[0]
    assert isinstance(restored_cylinder, Cylinder)
    assert restored_cylinder.source == "cylinder_annotator"
    assert restored_cylinder.inliers == [3, 4]
    assert isinstance(restored_cylinder.geometry, SemDTCylinder)
    assert restored_cylinder.geometry.width == 0.2
    assert restored_cylinder.geometry.height == 0.6
