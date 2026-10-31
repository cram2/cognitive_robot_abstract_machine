from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from krrood.ormatic.data_access_objects.helper import to_dao


# %% batched record writing
@dataclass
class BatchedRecordWriter:
    """
    Stores records one at a time, committing and detaching them periodically.

    Detaching, rather than only expiring, is what lets a stored record be released: an
    expired instance stays registered with the session and so stays alive for the whole
    run.
    """

    session: Session
    """
    Session on the processed database.
    """

    label: str
    """
    Name used in the progress output.
    """

    stored_count: int = 0
    """
    Records stored so far.
    """

    commit_batch_size: float = 500
    """
    How many records to stage before committing and detaching them.
    """

    def store(self, record: Any) -> None:
        """
        Convert *record* to its data access object and stage it for the next commit.

        :param record: The processed record to persist.
        """
        self.session.add(to_dao(record))
        self.stored_count += 1
        if self.stored_count % self.commit_batch_size == 0:
            self._commit()
            print(f"  committed {self.stored_count} {self.label}")

    def store_all(self, records: Iterable[Any]) -> None:
        """
        Store every record in *records* and commit what is left over.

        :param records: The processed records to persist.
        """
        for record in records:
            self.store(record)
        self.finish()

    def finish(self) -> None:
        """
        Commit whatever has not been committed yet and report the total.
        """
        self._commit()
        print(f"Stored {self.stored_count} {self.label}.")

    def _commit(self) -> None:
        self.session.commit()
        self.session.expunge_all()
