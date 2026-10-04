"""Hybrid Memory Agent — combines episodic vector memory with Feast user profile.

Architecture: see bonus/ARCHITECTURE.md.

The agent maintains two memory systems:
  1. Episodic memory  — past conversations/documents, stored as vectors in Qdrant
                        with hybrid (BM25 + vector) retrieval via RRF.
  2. Stable profile    — user attributes from Feast Feature Store (topic affinity,
                        preferred language, recent query velocity).

`recall()` merges both sources into a context string that could be fed to an LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os
import re
import time
from dataclasses import dataclass, field

_BIN = Path(sys.executable).parent
if str(_BIN) not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = f"{_BIN}{os.pathsep}{os.environ.get('PATH', '')}"

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi

from app.embeddings import Embedder

TENANT_COLLECTION = "episodic_memory"
RRF_K = 60
DEFAULT_USER_ID = "u_001"
MAX_CHUNK_TOKENS = 200

_TURN_SPLIT = re.compile(
    r"\n*(?:User|Người dùng|Assistant|Trợ lý)[:\s]*\n", re.IGNORECASE
)


@dataclass
class MemoryChunk:
    text: str
    source: str
    user_id: str
    ts: float = field(default_factory=time.time)


class HybridMemoryAgent:
    """AI assistant memory that bridges episodic vector store and Feast profile.

    Parameters
    ----------
    collection_name : str
        Qdrant collection for episodic memories (in-memory by default).
    feast_repo : str | None
        Path to Feast feature repo. None = offline mode (profile is empty).
    """

    def __init__(
        self,
        collection_name: str = TENANT_COLLECTION,
        feast_repo: str | None = "app/feast_repo",
    ) -> None:
        self.embedder = Embedder()
        self.client = QdrantClient(":memory:")
        self.collection = collection_name
        self._init_collection()

        self.bm25_docs: list[str] = []
        self.bm25_corpus: list[list[str]] = []
        self._point_counter: int = 0

        self.feast_store = self._init_feast(feast_repo)

    # ── setup ─────────────────────────────────────────────────────────────

    def _init_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.embedder.dim, distance=Distance.COSINE
            ),
        )

    def _init_feast(self, repo: str | None):
        if repo is None:
            return None
        repo_path = Path(repo)
        if not repo_path.exists() or not (repo_path / "registry.db").exists():
            return None
        try:
            from feast import FeatureStore

            store = FeatureStore(repo_path=str(repo_path))
            store.get_online_features(
                features=["user_profile_features:topic_affinity"],
                entity_rows=[{"user_id": DEFAULT_USER_ID}],
            )
            return store
        except Exception as exc:  # noqa: BLE001
            print(f"[agent] Feast unavailable (offline mode): {exc}")
            return None

    # ── chunking ──────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _chunk_text(self, text: str) -> list[str]:
        turns = [t.strip() for t in _TURN_SPLIT.split(text) if t.strip()]
        if not turns:
            turns = [text]

        chunks: list[str] = []
        for turn in turns:
            toks = self._tokenize(turn)
            if len(toks) <= MAX_CHUNK_TOKENS:
                chunks.append(turn)
            else:
                sentences = re.split(r"(?<=[.!?])\s+", turn)
                buf = ""
                for sent in sentences:
                    if len(self._tokenize(buf + " " + sent)) <= MAX_CHUNK_TOKENS:
                        buf += " " + sent if buf else sent
                    else:
                        if buf:
                            chunks.append(buf)
                        buf = sent
                if buf:
                    chunks.append(buf)
        return chunks

    # ── remember ──────────────────────────────────────────────────────────

    def remember(self, text: str, user_id: str = DEFAULT_USER_ID) -> int:
        """Add a new piece of episodic memory for this user.

        Chunks the text, embeds, and upserts into the user's vector collection.
        Returns the number of chunks stored.
        """
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        vectors = list(self.embedder.embed(chunks))
        now = time.time()
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            self._point_counter += 1
            points.append(PointStruct(
                id=self._point_counter,
                vector=vec.tolist(),
                payload={
                    "user_id": user_id,
                    "text": chunk,
                    "ts": now,
                    "source": f"chunk_{i}",
                },
            ))
            self.bm25_docs.append(chunk)
            self.bm25_corpus.append(self._tokenize(chunk))

        self.client.upsert(collection_name=self.collection, points=points)
        return len(chunks)

    # ── profile ───────────────────────────────────────────────────────────

    def _get_user_profile(self, user_id: str = DEFAULT_USER_ID) -> dict:
        """Fetch stable user profile from Feast. Empty dict if Feast unavailable."""
        if self.feast_store is None:
            return {}
        try:
            feats = [
                "user_profile_features:topic_affinity",
                "user_profile_features:preferred_language",
                "query_velocity_features:queries_last_hour",
                "query_velocity_features:distinct_topics_24h",
            ]
            result = self.feast_store.get_online_features(
                features=feats,
                entity_rows=[{"user_id": user_id}],
            ).to_dict()
            return {k: (v[0] if v else None) for k, v in result.items()}
        except Exception as exc:  # noqa: BLE001
            print(f"[agent] Feast lookup failed for {user_id}: {exc}")
            return {}

    # ── recall ────────────────────────────────────────────────────────────

    def _hybrid_search(
        self, query: str, user_id: str, top_k: int = 5, depth: int = 50
    ) -> list[str]:
        """Hybrid retrieval on episodic memory: BM25 + vector + RRF.

        Filters by user_id so each user only recalls their own memories.
        """
        from qdrant_client import models

        qv = next(self.embedder.embed([query])).tolist()
        user_filter = models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        ])

        vector_hits = self.client.query_points(
            collection_name=self.collection,
            query=qv,
            query_filter=user_filter,
            limit=depth,
        ).points

        if self.bm25_docs:
            tokenized = self._tokenize(query)
            scores = BM25Okapi(self.bm25_corpus).get_scores(tokenized)
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:depth]
            kw_ids = [str(i) for i in ranked]
        else:
            kw_ids = []

        rrf: dict[str, float] = {}
        meta: dict[str, str] = {}

        for rank, p in enumerate(vector_hits, start=1):
            doc_id = str(p.id)
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            meta[doc_id] = p.payload.get("text", "")

        for rank, doc_id in enumerate(kw_ids, start=1):
            if doc_id in meta:
                rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            else:
                rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
                meta[doc_id] = self.bm25_docs[int(doc_id)] if doc_id.isdigit() else ""

        ordered = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]
        return [meta.get(did, "") for did, _ in ordered if meta.get(did)]

    def recall(
        self, query: str, user_id: str = DEFAULT_USER_ID, top_k: int = 5
    ) -> str:
        """Retrieve top-K memories + user profile features → return assembled context.

        Combines episodic memory (vector + keyword via RRF) with stable user
        profile (from Feast). Returns a formatted string ready for an LLM prompt.
        """
        profile = self._get_user_profile(user_id)
        memories = self._hybrid_search(query, user_id, top_k=top_k)

        parts: list[str] = []
        if profile:
            parts.append("=== User Profile ===")
            affinity = profile.get("topic_affinity", "unknown")
            lang = profile.get("preferred_language", "unknown")
            qph = profile.get("queries_last_hour", "?")
            parts.append(f"topic_affinity: {affinity}")
            parts.append(f"preferred_language: {lang}")
            parts.append(f"queries_last_hour: {qph}")
        else:
            parts.append("=== User Profile ===")
            parts.append("(Feast not available — profile context omitted)")

        parts.append("")
        parts.append("=== Episodic Memory (top results) ===")
        if memories:
            for i, mem in enumerate(memories, 1):
                preview = mem[:200] + "..." if len(mem) > 200 else mem
                parts.append(f"[{i}] {preview}")
        else:
            parts.append("(no matching memories found)")

        return "\n".join(parts)
