# Bonus Challenge: Hybrid Memory Agent Architecture

> Contributors: VuHieuThien (2A202602867)
> Date: 2026-10-05
> Status: POC complete — `python bonus/demo.py` exits 0.

## 1. Context

This bonus extends Lab 19 by building a personal AI assistant for Vietnamese users
that maintains two memory systems:

1. **Episodic memory** — past conversations, read documents, user notes.
   Stored as dense vectors in Qdrant with hybrid (BM25 + vector) retrieval via RRF.
2. **Stable profile** — user attributes from Feast Feature Store:
   `topic_affinity`, `preferred_language`, `queries_last_hour`.

The agent (`HybridMemoryAgent`) exposes two methods:
- `remember(text, user_id)` — chunk, embed, upsert to Qdrant.
- `recall(query, user_id)` → context string — fetch profile from Feast,
  hybrid-search episodic memory, assemble into an LLM-ready prompt.

## 2. Architecture Diagram

```mermaid
flowchart TD
    subgraph User["User (u_001)"]
        Q[Query: "Tôi đang quan tâm gì gần đây?"]
    end

    subgraph Agent["HybridMemoryAgent"]
        direction LR
        A1[Chunk text → split by turns, max 200 tokens]
        A2[Embed: bge-small-en-v15, 384-dim]
        A3[Qdrant in-memory: episodic_memory collection]
        A4[Feast online store: SQLite]
    end

    subgraph Profile["Stable Profile (Feast)"]
        P1[topic_affinity: cloud]
        P2[preferred_language: vi]
        P3[queries_last_hour: 11]
    end

    subgraph Memory["Episodic Memory (Qdrant)"]
        M1[BM25 index on chunk text]
        M2[Vector index (COSINE)]
        M3[RRF k=60 fusion]
    end

    subgraph Output["Assembled Context"]
        O1["User Profile section"]
        O2["Episodic Memory top-K chunks"]
    end

    Q --> A1
    A1 --> A2 --> A3
    A3 --> M1
    A3 --> M2
    M1 --> M3
    M2 --> M3
    A4 --> Profile
    Profile --> P1 & P2 & P3
    M3 --> Memory
    P1 & P2 & P3 --> Output
    M3 --> Output
    O1 --> O2 --> C[Context string for LLM]
```

**Data flow:**
1. `remember()` chops incoming text into ≤200-token chunks by conversation turn.
2. Each chunk embedded via `fastembed` → stored in Qdrant with `user_id` payload.
3. `recall()` fetches user profile from Feast online store (<10ms lookup).
4. Query embedded → vector ANN search + BM25 lexical search, both filtered by `user_id`.
5. RRF (k=60) fuses results → top-K memory chunks.
6. Context string assembled: profile + ranked memories.

## 3. Three Architecture Decisions (with explicit tradeoff)

### Decision 1: Chunking Strategy — Semantic Turn Split vs Fixed-Size Sliding Window

**Chosen: Semantic turn split** — split memory text on `User:` / `Assistant:`
markers, cap at 200 tokens. If a turn exceeds 200 tokens, fall back to sentence-level
splitting.

| Approach | Retrieval quality | Storage cost | Context efficiency |
|---|---|---|---|
| Semantic turn split (chosen) | High — each chunk is a complete thought | Moderate — 1-3 chunks per memory | Good — chunks are self-contained |
| Fixed 128-token sliding window | Lower — chunks cut mid-sentence, lose meaning | High — 5-10x more chunks | Poor — many partial chunks with no signal |
| Per-sentence (no merge) | Low-medium — too granular, no context window | Very high — 10x more points | Poor — retrieval signal diluted |

**Why semantic turn:** An LLM context chunk needs enough surrounding conversation
to be meaningful. A sliding window that cuts "Kubernetes HPA auto-scales based on..."
mid-sentence returns low-quality RRF scores. Turn-level chunks preserve the
question-answer pairing that makes retrieval accurate.

### Decision 2: Feature Schema — Tabular Features Only vs Embedding Features

**Chosen: Tabular features only** from Feast — `topic_affinity`,
`preferred_language`, `queries_last_hour`, `distinct_topics_24h`.

| Approach | Latency | Complexity | Retrieval value |
|---|---|---|---|
| Tabular (chosen) | <10ms (SQLite) | Simple — direct field access | High — drives query routing/filter |
| Embedding features | 50-100ms (extra vector search) | High — extra Qdrant collection + sync | Marginal — topic_affinity already covers intent |

**Why tabular:** The profile features serve the agent's retrieval strategy
(route to correct topic, adapt language), not similarity search. Storing
*another* embedding vector for the user profile adds latency without improving
recall — the episodic vector search already handles similarity. Feature Store's
strength is **structured key-value lookup**, not vector storage.

### Decision 3: Freshness Strategy — Immediate Upsert vs Batch Refresh

**Chosen: Immediate upsert** — `remember()` writes to Qdrant synchronously.
Profile reads use Feast online store (already materialized from NB4).

| Approach | Consistency | Latency | Throughput |
|---|---|---|---|
| Immediate upsert (chosen) | Strong — query sees latest memory | ~5ms per write | Low — not for write-heavy streams |
| 5-min batch refresh | Eventual — query misses last 5 min | ~0ms (no write) | High |
| Daily batch | Stale — misses recent activity | ~0ms | Very high |

**Per use case:**

| Use case | Strategy | Rationale |
|---|---|---|
| "Tôi vừa đọc xong tài liệu này, hãy nhớ" | **Immediate** | User expects confirmation it was saved |
| "Recommend đọc gì tiếp" | **Immediate** | Uses latest topic_affinity from profile |
| Analytics dashboard (user behavior trends) | **Daily** | No need sub-second freshness |
| Fraud detection (real-time) | **Streaming** (5-min) | Not in scope for this POC |

## 4. Rejected Alternative

**Considered: Storing episodic memory in Feast as an embedding feature view.**

This would let Feast manage both stable profile and episodic vectors through
unified infrastructure.

**Rejected because:** Episodic memory has a fundamentally different access
pattern and update cadence than stable profile:
- **Profile:** append-only updates, long TTL (30 days), batch materialize fine.
- **Episodic:** append-every-conversation, short relevance window, high write
  rate, no PIT-join needed.

Mixing them in one FeatureView would force the entire collection to use the
shorter TTL, evicting stable profile data prematurely. The two systems also
have incompatible serving patterns: Feast is optimized for key-value lookups,
Qdrant is optimized for k-NN / RRF. Keeping them separate lets each system
optimize for its workload.

## 5. Vietnamese-Context Considerations

1. **Tokenization for BM25:** The agent uses whitespace tokenize
   (`_tokenize()` = `text.lower().split()`) inherited from `app/search.py`.
   This is "good enough" for mixed Vietnamese+English technical text but
   misses subword information. Production should use `underthesea` (word
   segmentation) or `pyvi` for better keyword recall.

2. **Code-switching (Vi/En mix):** The `TOPIC_HINTS` map in `app/agent.py`
   already handles this — e.g., `"cloud"` topic matches both `"đám mây"` and
   `"cloud"`. The agent inherits this pattern from the lab's existing `ToolArgs`
   schema.

3. **Embedding model limitation:** `bge-small-en-v1.5` is English-trained.
   Vietnamese paraphrases suffer (24-32% recall per NB2 results). The agent
   mitigates this by using **both** BM25 (keyword) and vector (semantic) with
   RRF — the keyword signal catches verbatim matches that vector misses. For
   production, swapping to `bge-m3` via `EMBEDDING_BACKEND=bge-m3` (Docker
   path) would improve Vietnamese recall significantly.

4. **Datetime in TTL:** The agent's `ts` field uses Unix timestamps. No
   timezone issues arise because all operations are within one process
   lifetime. For multi-day sessions, store `ts` as timezone-aware UTC.

## 6. Limitations (What This POC Doesn't Handle)

- **Per-user privacy isolation:** In-memory Qdrant doesn't enforce tenant
  isolation. Production needs `user_id` filter on every query (implemented
  here) but also encryption at rest and access control middleware.
- **Memory decay / forgetting:** No TTL-based eviction of old episodic
  memories. A production system should prune memories older than N days.
- **Multi-device sync:** Two devices running the agent separately would
  create divergent memory stores. Need a shared backend (Qdrant server,
  not in-memory).
- **No LLM integration:** `recall()` returns a context string — it does not
  call an LLM to generate the final response. The POC ends at context
  assembly; the "assistant turn" is a stub for the user.
- **No CRUD on memories:** Can only `remember` (append). No delete, update,
  or explicit forget. User can't remove a memory they regret sharing.
