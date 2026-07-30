"""
Regression test for ``EntityConverter._convert``'s entity naming.

:class:`~semantic_digital_twin.adapters.multi_sim.MultiSimCamera` (and subclasses like
:class:`~semantic_digital_twin.adapters.multi_sim.MujocoCamera`) carry a plain ``str``
``name`` rather than a :class:`~semantic_digital_twin.datastructures.prefixed_name.PrefixedName`.
``EntityConverter._convert`` used to only honor ``PrefixedName``-typed names, silently
discarding any string name and falling back to an ``id()``-based one instead - so a
:class:`~semantic_digital_twin.adapters.mujoco_video_recording.MujocoVideoRecorder`'s
auto-attached camera could never be found by name after the scene was built.
"""

from semantic_digital_twin.adapters.multi_sim import MujocoCamera, MujocoCameraConverter
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world_description.world_entity import Body


def test_mujoco_camera_converter_honors_a_given_string_name():
    body = Body(name=PrefixedName("world"))
    camera = MujocoCamera(name="my_named_camera", body=body)

    properties = MujocoCameraConverter.convert(camera)

    assert properties["name"] == "my_named_camera"


def test_mujoco_camera_converter_falls_back_to_an_id_based_name_when_unnamed():
    body = Body(name=PrefixedName("world"))
    camera = MujocoCamera(name="", body=body)

    properties = MujocoCameraConverter.convert(camera)

    assert properties["name"] == f"mujococamera_{id(camera)}"
