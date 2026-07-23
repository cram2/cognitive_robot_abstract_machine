from pathlib import Path

PACKAGES = [
    "cognitive_robot_abstract_machine",
    "coraplex",
    "experiments",
    "giskardpy",
    "krrood",
    "physics_simulators",
    "probabilistic_model",
    "random_events",
    "robokudo",
    "semantic_digital_twin",
]

VERSION_FILE = Path(__file__).parents[2] / "VERSION"


def test_all_package_versions_match_root_version():
    expected = VERSION_FILE.read_text().strip()

    for package_name in PACKAGES:
        package = __import__(package_name)
        assert package.__version__ == expected, (
            f"{package_name} version {package.__version__} "
            f"does not match root VERSION {expected}"
        )
