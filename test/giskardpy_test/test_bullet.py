import giskardpy_bullet_bindings as bpb
import numpy as np


def create_sphere_shape(diameter: float) -> bpb.SphereShape:
    out = bpb.SphereShape(0.5 * diameter)
    out.margin = 0.001
    return out


def test_boxes():
    kw = bpb.KineverseWorld()
    body1 = create_sphere_shape(1)
    body2 = create_sphere_shape(1)
    body1 = kw.add_collision_object(
        body1,
        transform=bpb.Transform().identity(),
    )
    body2 = kw.add_collision_object(
        body2,
        transform=bpb.Transform().identity(),
    )
    assert kw.num_collision_objects() == 2

    poses = np.array(
        [
            [1, 0, 0, 2],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )

    bpb.batch_set_transforms([body1, body2], poses)
    distance = kw.get_distance(body1, body2)
    assert len(distance) == 1
    assert np.isclose(distance[0].distance, 1)
    assert np.allclose(distance[0].point_a, [-0.5, 0, 0, 1])
    assert np.allclose(distance[0].point_b, [0.5, 0, 0, 1])
    assert np.allclose(distance[0].normal_world_b, [1.0, 0, 0, 0])
