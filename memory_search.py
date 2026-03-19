"""Hybrid memory search for agent_computer.

Combines vector search (via LM Studio embeddings) with keyword search
using Reciprocal Rank Fusion. Stores everything in a local SQLite database.

Keyword search uses SQLite FTS5 (BM25). Vector search uses a pre-built
NumPy matrix for O(1) batch cosine similarity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

logger = logging.getLogger("agent_computer.memory_search")

# Characters with special meaning in FTS5 query syntax
_FTS5_SPECIAL = re.compile(r'[":*^{}()\[\]|&!]')


@dataclass
class MemoryResult:
    source_type: str
    source_id: str
    title: str
    content: str
    score: float


def _parse_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) pairs on ## headings."""
    sections: list[tuple[str, str]] = []
    heading = ""
    lines: list[str] = []
    for line in text.split("\n"):
        if line.startswith("## "):
            if heading:
                sections.append((heading, "\n".join(lines).strip()))
            heading = line[3:].strip()
            lines = []
        elif heading:
            lines.append(line)
    if heading:
        sections.append((heading, "\n".join(lines).strip()))
    return sections


class MemorySearch:
    """Hybrid vector + keyword memory search backed by SQLite."""

    def __init__(self, workspace: str, embedding_base_url: str = "http://10.5.0.2:1234/v1",
                 embedding_model: str = "text-embedding-nomic-embed-text-v1.5", top_k: int = 5):
        self.workspace = Path(workspace)
        self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.memory_dir / "memory.db"
        self.embedding_model = embedding_model
        self.top_k = top_k
        self._embedding_dim: int | None = None
        self._client: OpenAI | None = None
        self._embeddings_available = True

        # Vectorized search state
        self._embedding_matrix: np.ndarray | None = None
        self._embedding_ids: list[int] = []
        self._matrix_dirty: bool = True

        try:
            self._client = OpenAI(base_url=embedding_base_url, api_key="lm-studio")
        except Exception as e:
            logger.warning(f"Failed to create embedding client: {e}")
            self._embeddings_available = False

        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL, source_id TEXT NOT NULL,
                title TEXT NOT NULL, content TEXT NOT NULL,
                embedding BLOB, created_at REAL NOT NULL,
                UNIQUE(source_type, source_id))""")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source_type, source_id)")

            # FTS5 virtual table for keyword search
            conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(title, content, content=memories, content_rowid=id)""")

            # Triggers to keep FTS in sync with the main table
            conn.executescript("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, title, content)
                    VALUES (new.id, new.title, new.content);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, title, content)
                    VALUES ('delete', old.id, old.title, old.content);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, title, content)
                    VALUES ('delete', old.id, old.title, old.content);
                    INSERT INTO memories_fts(rowid, title, content)
                    VALUES (new.id, new.title, new.content);
                END;
            """)

            # One-time backfill: populate FTS from existing rows if needed
            try:
                fts_count = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
                mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
                if fts_count == 0 and mem_count > 0:
                    conn.execute(
                        "INSERT INTO memories_fts(rowid, title, content) "
                        "SELECT id, title, content FROM memories"
                    )
                    logger.info(f"Backfilled {mem_count} rows into FTS5 index")
            except Exception:
                pass  # FTS already populated or table just created

    # ── Embeddings ──

    def _get_embedding(self, text: str) -> np.ndarray | None:
        if not self._client or not self._embeddings_available:
            return None
        try:
            text = text.replace("\n", " ").strip()
            if not text:
                return None
            resp = self._client.embeddings.create(input=[text], model=self.embedding_model)
            vec = np.array(resp.data[0].embedding, dtype=np.float32)
            if self._embedding_dim is None:
                self._embedding_dim = len(vec)
                logger.info(f"Detected embedding dimension: {self._embedding_dim}")
            return vec
        except Exception as e:
            logger.warning(f"Embedding call failed: {e}")
            self._embeddings_available = False
            return None

    # ── Indexing ──

    def index_text(self, source_type: str, source_id: str, title: str, content: str) -> bool:
        """Index a single text entry. Returns True if newly inserted."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            if conn.execute("SELECT 1 FROM memories WHERE source_type=? AND source_id=?",
                            (source_type, source_id)).fetchone():
                return False
            vec = self._get_embedding(f"{title}\n{content}")
            blob = vec.tobytes() if vec is not None else None
            conn.execute("INSERT INTO memories (source_type,source_id,title,content,embedding,created_at) "
                         "VALUES (?,?,?,?,?,?)", (source_type, source_id, title, content, blob, time.time()))
            conn.commit()
            self._matrix_dirty = True
            return True
        finally:
            conn.close()

    def index_all(self) -> dict:
        """Index all existing memory sources. Returns counts of new entries."""
        counts = {"knowledge": 0, "learning": 0, "session_summary": 0}

        for fname, stype in [("knowledge.md", "knowledge"), ("learnings.md", "learning")]:
            path = self.memory_dir / fname
            if not path.exists():
                continue
            for i, (heading, body) in enumerate(_parse_markdown_sections(path.read_text(encoding="utf-8"))):
                sid = _slugify(heading) or f"{stype}_{i}"
                if self.index_text(stype, sid, heading, body):
                    counts[stype] += 1

        index_path = self.memory_dir / "index.json"
        if index_path.exists():
            try:
                sessions = json.loads(index_path.read_text(encoding="utf-8")).get("sessions", {})
                for sid, entry in sessions.items():
                    summary = entry.get("summary", "")
                    if summary and entry.get("status") == "processed":
                        if self.index_text("session_summary", sid, f"Session: {sid}", summary):
                            counts["session_summary"] += 1
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to read index.json: {e}")

        logger.info(f"Indexed {sum(counts.values())} new entries: {counts}")
        return counts

    # ── Embedding matrix ──

    def _rebuild_embedding_matrix(self) -> None:
        """Load all embeddings from SQLite into a single normalized (n, dim) matrix."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()

        if not rows:
            self._embedding_matrix = None
            self._embedding_ids = []
            self._matrix_dirty = False
            return

        ids: list[int] = []
        vecs: list[np.ndarray] = []
        for row_id, blob in rows:
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            ids.append(row_id)
            vecs.append(vec)

        self._embedding_matrix = np.stack(vecs)  # (n, dim)
        self._embedding_ids = ids
        self._matrix_dirty = False

    # ── Search ──

    def search(self, query: str, top_k: int | None = None) -> list[MemoryResult]:
        """Hybrid search: vector similarity + keyword matching merged via RRF."""
        if top_k is None:
            top_k = self.top_k

        candidate_k = 2 * top_k
        vector_ranked = self._vector_search(query, candidate_k)
        keyword_ranked = self._keyword_search(query, candidate_k)
        return self._rrf_merge(vector_ranked, keyword_ranked, top_k)

    def _vector_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return top-k (row_id, cosine_similarity) via matrix-vector multiply."""
        query_vec = self._get_embedding(query)
        if query_vec is None:
            return []

        if self._matrix_dirty:
            self._rebuild_embedding_matrix()

        if self._embedding_matrix is None or len(self._embedding_ids) == 0:
            return []

        # Normalize query vector
        qnorm = np.linalg.norm(query_vec)
        if qnorm > 0:
            query_vec = query_vec / qnorm

        # Single matrix-vector multiply: all cosine similarities at once
        scores = self._embedding_matrix @ query_vec  # (n,)

        # Efficient top-k via argpartition (avoids full sort)
        k = min(top_k, len(scores))
        if k <= 0:
            return []
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self._embedding_ids[i], float(scores[i])) for i in top_indices]

    def _keyword_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Return top-k (row_id, bm25_score) using FTS5 MATCH."""
        # Sanitize: remove FTS5 special characters
        sanitized = _FTS5_SPECIAL.sub(" ", query).strip()
        if not sanitized:
            return []

        conn = sqlite3.connect(str(self.db_path))
        try:
            # FTS5 rank column is negative BM25 (lower = better match),
            # so we negate it to get a positive score and ORDER BY rank (ascending)
            rows = conn.execute(
                "SELECT rowid, -rank FROM memories_fts WHERE memories_fts MATCH ? "
                "ORDER BY rank LIMIT ?",
                (sanitized, top_k),
            ).fetchall()
            return [(row_id, score) for row_id, score in rows]
        except sqlite3.OperationalError:
            # Query produced no valid FTS tokens (e.g. all stop words)
            return []
        finally:
            conn.close()

    def _rrf_merge(self, vector_ranked: list[tuple[int, float]],
                   keyword_ranked: list[tuple[int, float]],
                   top_k: int, k: int = 60) -> list[MemoryResult]:
        """Merge results by Reciprocal Rank Fusion, then fetch only needed rows."""
        rrf: dict[int, float] = {}
        for rank, (row_id, _) in enumerate(vector_ranked):
            rrf[row_id] = rrf.get(row_id, 0.0) + 1.0 / (k + rank + 1)
        for rank, (row_id, _) in enumerate(keyword_ranked):
            rrf[row_id] = rrf.get(row_id, 0.0) + 1.0 / (k + rank + 1)
        if not rrf:
            return []

        # Pick top-k row IDs by RRF score
        top_items = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_ids = [row_id for row_id, _ in top_items]
        score_map = {row_id: score for row_id, score in top_items}

        # Fetch only those rows from SQLite
        placeholders = ",".join("?" for _ in top_ids)
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                f"SELECT id, source_type, source_id, title, content FROM memories "
                f"WHERE id IN ({placeholders})",
                top_ids,
            ).fetchall()
        finally:
            conn.close()

        row_map = {r[0]: r for r in rows}
        results = []
        for row_id in top_ids:
            r = row_map.get(row_id)
            if r:
                results.append(MemoryResult(r[1], r[2], r[3], r[4], round(score_map[row_id], 6)))
        return results

    # ── Async wrappers ──

    async def async_index_all(self) -> dict:
        return await asyncio.to_thread(self.index_all)

    async def async_index_text(self, source_type: str, source_id: str, title: str, content: str) -> bool:
        return await asyncio.to_thread(self.index_text, source_type, source_id, title, content)

    async def async_search(self, query: str, top_k: int | None = None) -> list[MemoryResult]:
        return await asyncio.to_thread(self.search, query, top_k)

    # ── Stats ──

    def stats(self) -> dict:
        conn = sqlite3.connect(str(self.db_path))
        try:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            with_emb = conn.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL").fetchone()[0]
            by_type = dict(conn.execute("SELECT source_type, COUNT(*) FROM memories GROUP BY source_type").fetchall())
        finally:
            conn.close()
        return {"total": total, "with_embeddings": with_emb, "by_type": by_type,
                "embedding_dim": self._embedding_dim, "embeddings_available": self._embeddings_available}


# ── Utilities ──

def _slugify(text: str) -> str:
    return re.sub(r"[\s-]+", "_", re.sub(r"[^a-z0-9\s-]", "", text.lower().strip()))[:80]


# ── CLI test ──

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
    sys.path.insert(0, str(Path(__file__).parent))
    from config import load_config

    cfg = load_config()
    ws = cfg.agent.workspace
    print(f"Workspace: {ws}\nMemory DB: {Path(ws) / 'memory' / 'memory.db'}\n")

    ms = MemorySearch(workspace=ws, embedding_base_url=cfg.memory.embedding_base_url,
                      embedding_model=cfg.memory.embedding_model, top_k=cfg.memory.top_k)

    print("=== Indexing ===")
    print(f"New entries: {ms.index_all()}\n")

    print("=== Stats ===")
    for k, v in ms.stats().items():
        print(f"  {k}: {v}")
    print()

    query = sys.argv[1] if len(sys.argv) > 1 else "What errors or mistakes happened?"
    print(f"=== Search: '{query}' ===")
    results = ms.search(query, top_k=5)
    if not results:
        print("No results found.")
    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {r.score}) ---")
        print(f"  Type: {r.source_type} | ID: {r.source_id}")
        print(f"  Title: {r.title}")
        print(f"  Content: {r.content[:200].replace(chr(10), ' ')}...")
