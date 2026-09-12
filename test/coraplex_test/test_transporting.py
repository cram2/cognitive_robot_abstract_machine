"""
What a transport does about an object it finds inside a container.
"""

from coraplex.config.action_conf import ActionConfig
from coraplex.datastructures.enums import Arms
from coraplex.locations import factories
from coraplex.plans.factories import sequential
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Drawer,
    Handle,
    Milk,
)
from semantic_digital_twin.spatial_types.spatial_types import Pose

# %% fetching an object out of a drawer

DRAWER = "cabinet10_drawer_top"
"""
The apartment drawer the transport opens on its way to the object inside it.
"""

DRAWER_HANDLE = "handle_cab10_t"
"""
The handle of :data:`DRAWER`.
"""


def test_opening_a_container_on_the_way_stands_where_it_is_opened_from(
    mutable_model_world, monkeypatch
):
    """
    An object inside a drawer is fetched by opening the drawer first, and the robot
    stands back for that the way it does for any container rather than as close as it
    would to grasp something that stays put.
    """
    world, robot, context = mutable_model_world
    drawer_body = world.get_body_by_name(DRAWER)
    with world.modify_world():
        world.add_semantic_annotation_recursively(
            Drawer(
                root=drawer_body,
                handle=Handle(root=world.get_body_by_name(DRAWER_HANDLE)),
            )
        )
    transport = TransportAction(
        object_designator=world.get_semantic_annotations_by_type(Milk)[0],
        target_location=Pose(reference_frame=world.root),
        arm=Arms.RIGHT,
    )
    sequential([transport], context)

    asked_for = {}
    build_location = factories.reachability_location
    monkeypatch.setattr(
        factories,
        "reachability_location",
        lambda *args, **kwargs: asked_for.update(kwargs)
        or build_location(*args, **kwargs),
    )

    transport._make_open_container_actions(drawer_body)

    assert asked_for["reach_fraction"] == ActionConfig.accessing_reach_fraction
