"""Tests for LongTermMemoryStore using a mocked MongoDB client."""

from unittest.mock import MagicMock, call

import pytest

from llmr.workflows.lg_memory.memory.long_term_memory_store import LongTermMemoryStore


@pytest.fixture
def store() -> LongTermMemoryStore:
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_client.__getitem__.return_value.__getitem__.return_value = mock_collection
    return LongTermMemoryStore(client=mock_client)


class TestSave:
    def test_returns_string_id(self, store: LongTermMemoryStore) -> None:
        store.collection.insert_one.return_value.inserted_id = "507f1f77bcf86cd799439011"
        result = store.save(user_id="u1", content="User likes coffee")
        assert result == "507f1f77bcf86cd799439011"

    def test_calls_insert_once(self, store: LongTermMemoryStore) -> None:
        store.collection.insert_one.return_value.inserted_id = "abc"
        store.save(user_id="u1", content="A fact")
        store.collection.insert_one.assert_called_once()

    def test_doc_has_correct_user_id_and_content(self, store: LongTermMemoryStore) -> None:
        store.collection.insert_one.return_value.inserted_id = "xyz"
        store.save(user_id="test_user", content="some content")
        doc = store.collection.insert_one.call_args[0][0]
        assert doc["user_id"] == "test_user"
        assert doc["content"] == "some content"

    def test_defaults_memory_type_to_fact(self, store: LongTermMemoryStore) -> None:
        store.collection.insert_one.return_value.inserted_id = "id1"
        store.save(user_id="u1", content="c")
        doc = store.collection.insert_one.call_args[0][0]
        assert doc["memory_type"] == "fact"

    def test_custom_memory_type_stored(self, store: LongTermMemoryStore) -> None:
        store.collection.insert_one.return_value.inserted_id = "id2"
        store.save(user_id="u1", content="c", memory_type="preference")
        doc = store.collection.insert_one.call_args[0][0]
        assert doc["memory_type"] == "preference"


class TestUpsert:
    def test_inserts_when_no_existing(self, store: LongTermMemoryStore) -> None:
        store.collection.find_one.return_value = None
        store.collection.insert_one.return_value.inserted_id = "new_id"
        result = store.upsert(user_id="u1", content="new fact")
        store.collection.insert_one.assert_called_once()
        assert result == "new_id"

    def test_updates_when_existing(self, store: LongTermMemoryStore) -> None:
        store.collection.find_one.return_value = {
            "_id": "existing_id", "tags": [], "embedding": None, "metadata": {}
        }
        result = store.upsert(user_id="u1", content="existing fact")
        store.collection.update_one.assert_called_once()
        assert result == "existing_id"


class TestGetAll:
    def test_returns_list(self, store: LongTermMemoryStore) -> None:
        store.collection.find.return_value.sort.return_value.limit.return_value = [
            {"content": "fact1", "memory_type": "fact"},
        ]
        result = store.get_all(user_id="u1")
        assert isinstance(result, list)
        assert result[0]["content"] == "fact1"

    def test_filters_by_memory_type(self, store: LongTermMemoryStore) -> None:
        store.collection.find.return_value.sort.return_value.limit.return_value = []
        store.get_all(user_id="u1", memory_type="preference")
        query = store.collection.find.call_args[0][0]
        assert query["memory_type"] == "preference"


class TestDelete:
    def test_delete_all_returns_count(self, store: LongTermMemoryStore) -> None:
        store.collection.delete_many.return_value.deleted_count = 5
        assert store.delete_all("u1") == 5

    def test_delete_by_type_filters_correctly(self, store: LongTermMemoryStore) -> None:
        store.collection.delete_many.return_value.deleted_count = 2
        store.delete_by_type("u1", "fact")
        query = store.collection.delete_many.call_args[0][0]
        assert query["memory_type"] == "fact"
        assert query["user_id"] == "u1"


class TestStats:
    def test_aggregates_by_type(self, store: LongTermMemoryStore) -> None:
        store.collection.aggregate.return_value = [
            {"_id": "fact", "count": 3},
            {"_id": "preference", "count": 1},
        ]
        assert store.stats("u1") == {"fact": 3, "preference": 1}

    def test_empty_returns_empty_dict(self, store: LongTermMemoryStore) -> None:
        store.collection.aggregate.return_value = []
        assert store.stats("u_new") == {}
