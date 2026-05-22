import os

import pytest

from semantic_digital_twin.adapters.partnet_mobility_dataset.loader import (
    PartNetMobilityDatasetLoader,
    SAPIEN_ACCESS_TOKEN_ENVIRONMENT_VARIABLE_NAME,
)

from semantic_digital_twin.world_description.connections import (
    FixedConnection,
    Connection6DoF,
)


@pytest.mark.skipif(
    os.getenv(SAPIEN_ACCESS_TOKEN_ENVIRONMENT_VARIABLE_NAME, None) is None,
    reason="SAPIEN access token not set",
)
def test_loader():
    loader = PartNetMobilityDatasetLoader()
    world = loader.load()
    assert len(world.bodies) > 0
    assert len(world.semantic_annotations) > 0

    unique_connection_types = {type(c) for c in world.connections}
    interesting_connection_types = unique_connection_types - {
        FixedConnection,
        Connection6DoF,
    }
    assert interesting_connection_types != {}
