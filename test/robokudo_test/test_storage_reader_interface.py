"""
Replay of stored recordings into the world perception perceives into.
"""

from __future__ import annotations

# %% Imports
from unittest.mock import Mock

import pytest

from robokudo.cas import CAS
from robokudo.io.exceptions import ReplayWorldShared
from robokudo.io.storage_reader_interface import StorageReaderInterface

from ._perception_world import SharedWorld

# %% replay world


class TestReplayWorld:
    """
    The world a recording is replayed into.
    """

    def test_replaying_into_a_shared_world_raises_before_reading_a_frame(
        self, *, shared_world: SharedWorld
    ) -> None:
        """
        A recording is never replayed into a world other processes share.
        """
        reader = StorageReaderInterface.__new__(StorageReaderInterface)
        reader.reader = Mock()

        with pytest.raises(ReplayWorldShared):
            reader.set_data(cas=CAS(), world=shared_world.world)

        reader.reader.get_next_frame.assert_not_called()
