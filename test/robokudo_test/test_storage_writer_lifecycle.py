from dataclasses import dataclass

import robokudo.annotators.storage as storage_module
from robokudo.annotators.storage import StorageWriter

# %% storage preparation


@dataclass
class StorageResetRecorder:
    """
    Record database resets requested by a storage writer.
    """

    database_name: str
    """Name of the database associated with the recorder."""

    reset_count: int = 0
    """
    Number of recorded database resets.
    """

    def drop_database(self) -> None:
        """
        Record a database reset.
        """
        self.reset_count += 1


def test_storage_writer_defers_database_reset_until_setup(monkeypatch) -> None:
    """
    A storage writer resets its database exactly once during setup.
    """
    descriptor = StorageWriter.Descriptor()
    storage = StorageResetRecorder(database_name=descriptor.parameters.db_name)
    monkeypatch.setattr(storage_module, "Storage", lambda database_name: storage)

    writer = StorageWriter(descriptor=descriptor)

    assert storage.reset_count == 0

    writer.setup()
    writer.setup()

    assert storage.reset_count == 1
