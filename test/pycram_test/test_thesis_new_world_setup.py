from demos.thesis_new.src.world_setup import (
    resolve_environment_name,
    resolve_environment_path,
)


def test_resolve_environment_path_accepts_absolute_urdf_path(tmp_path):
    apartment_urdf = tmp_path / "custom_apartment.urdf"
    apartment_urdf.write_text("<robot name='apartment' />", encoding="utf-8")

    assert resolve_environment_name(str(apartment_urdf)) == "custom_apartment"
    assert resolve_environment_path(str(apartment_urdf)) == str(apartment_urdf)


def test_resolve_environment_path_keeps_package_uri():
    package_uri = "package://iai_apartment//urdf/apartment.urdf"

    assert resolve_environment_name(package_uri) == "apartment"
    assert resolve_environment_path(package_uri) == package_uri
