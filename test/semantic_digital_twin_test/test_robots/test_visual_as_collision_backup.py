from semantic_digital_twin.robots.garmi import Garmi


def test_a_robot_that_needs_it_gets_its_visual_geometry_as_collision(
    _garmi_world_setup,
):
    """
    GARMI's covers bound its real width but are drawn without collision geometry, so a
    world built from its description has to fall back to their visuals.

    Otherwise every rule naming a cover raises, and the shell is invisible to collision
    checking.
    """
    assert Garmi.uses_visual_as_collision_backup
    for cover_name in (
        "right_side_cover_link",
        "left_side_cover_link",
        "front_cover_link",
        "rear_cover_link",
    ):
        assert _garmi_world_setup.get_body_by_name(cover_name).has_collision()
