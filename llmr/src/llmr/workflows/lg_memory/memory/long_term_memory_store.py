"""Long-term memory store — persists facts, preferences, and summaries per user in MongoDB."""

from datetime import datetime, timezone
from typing_extensions import Any, Literal

from pymongo import DESCENDING, MongoClient
from pymongo.collection import Collection

from ..config import MONGO_CONFIG

MemoryType = Literal["fact", "preference", "summary", "entity"]


class LongTermMemoryStore:
    """Manages long-term user memories (facts, preferences, summaries, entities) in MongoDB."""

    def __init__(self, client: MongoClient | None = None) -> None:
        _client = client or MongoClient(MONGO_CONFIG.uri)
        db = _client[MONGO_CONFIG.db_name]
        self.collection: Collection = db[MONGO_CONFIG.col_long_term]
        self._ensure_indexes()

    # ── Indexes ────────────────────────────────────────────────────────────────

    def _ensure_indexes(self) -> None:
        self.collection.create_index("user_id")
        self.collection.create_index("memory_type")
        self.collection.create_index("tags")
        self.collection.create_index("created_at")
        self.collection.create_index([("user_id", 1), ("memory_type", 1)])

    # ── Write ──────────────────────────────────────────────────────────────────

    def save(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType = "fact",
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        source_thread_id: str | None = None,
    ) -> str:
        """Save a memory for a user. Returns the inserted MongoDB document ID as string."""
        doc = {
            "user_id": user_id,
            "content": content,
            "memory_type": memory_type,
            "tags": tags or [],
            "embedding": embedding,
            "metadata": metadata or {},
            "source_thread_id": source_thread_id,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        result = self.collection.insert_one(doc)
        return str(result.inserted_id)

    def upsert(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType = "fact",
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Update memory if same content exists for user, otherwise insert.

        Prevents duplicate facts accumulating over time.
        """
        existing = self.collection.find_one({"user_id": user_id, "content": content})
        if existing:
            self.collection.update_one(
                {"_id": existing["_id"]},
                {
                    "$set": {
                        "tags": tags or existing.get("tags", []),
                        "embedding": embedding or existing.get("embedding"),
                        "metadata": metadata or existing.get("metadata", {}),
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
            return str(existing["_id"])
        return self.save(user_id, content, memory_type, tags, embedding, metadata)

    # ── Read ───────────────────────────────────────────────────────────────────

    def get_all(
        self,
        user_id: str,
        memory_type: MemoryType | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """Fetch all memories for a user, optionally filtered by type."""
        query: dict[str, Any] = {"user_id": user_id}
        if memory_type:
            query["memory_type"] = memory_type
        return list(
            self.collection.find(query, {"_id": 0, "embedding": 0})
            .sort("created_at", DESCENDING)
            .limit(limit)
        )

    def get_by_tags(self, user_id: str, tags: list[str], limit: int = 10) -> list[dict]:
        """Fetch memories that match ANY of the given tags."""
        return list(
            self.collection.find(
                {"user_id": user_id, "tags": {"$in": tags}},
                {"_id": 0, "embedding": 0},
            )
            .sort("created_at", DESCENDING)
            .limit(limit)
        )

    def get_recent(self, user_id: str, n: int = 5) -> list[dict]:
        """Fetch the N most recently saved memories."""
        return self.get_all(user_id, limit=n)

    def get_summary(self, user_id: str) -> dict | None:
        """Get the latest conversation summary for a user."""
        return self.collection.find_one(
            {"user_id": user_id, "memory_type": "summary"},
            {"_id": 0, "embedding": 0},
            sort=[("created_at", DESCENDING)],
        )

    # ── Semantic Search (Atlas Vector Search) ──────────────────────────────────

    def semantic_search(
        self,
        query_embedding: list[float],
        user_id: str,
        top_k: int = 5,
        memory_type: MemoryType | None = None,
    ) -> list[dict]:
        """Semantic similarity search via MongoDB Atlas Vector Search (requires Atlas + vector index)."""
        pre_filter: dict[str, Any] = {"user_id": {"$eq": user_id}}
        if memory_type:
            pre_filter["memory_type"] = {"$eq": memory_type}

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "memory_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,
                    "limit": top_k,
                    "filter": pre_filter,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "content": 1,
                    "memory_type": 1,
                    "tags": 1,
                    "created_at": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return list(self.collection.aggregate(pipeline))

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete_by_type(self, user_id: str, memory_type: MemoryType) -> int:
        return self.collection.delete_many(
            {"user_id": user_id, "memory_type": memory_type}
        ).deleted_count

    def delete_all(self, user_id: str) -> int:
        return self.collection.delete_many({"user_id": user_id}).deleted_count

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self, user_id: str) -> dict:
        """Return memory counts per type for a user."""
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$memory_type", "count": {"$sum": 1}}},
        ]
        result = list(self.collection.aggregate(pipeline))
        return {r["_id"]: r["count"] for r in result}
