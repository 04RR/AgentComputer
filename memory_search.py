"""Hybrid memory search for agent_computer.

Combines vector search (via LM Studio embeddings) with keyword search
using Reciprocal Rank Fusion. Stores everything in a local SQLite database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI

logger = logging.getLogger("agent_computer.memory_search")

_STOP_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would "
    "could should may might shall can to of in for on with at by from as into "
    "through during before after and but or nor not so yet it its this that "
    "these those i me my we".split()
)


@dataclass
class MemoryResult:
    source_type: str
    source_id: str
    title: str
    content: str
    score: float


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
            for i, (heading, body) in enumerate(self._parse_markdown_sections(path.read_text(encoding="utf-8"))):
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

    @staticmethod
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

    # ── Search ──

    def search(self, query: str, top_k: int | None = None) -> list[MemoryResult]:
        """Hybrid search: vector similarity + keyword matching merged via RRF."""
        if top_k is None:
            top_k = self.top_k
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute(
                "SELECT id, source_type, source_id, title, content, embedding FROM memories"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return []

        candidate_k = min(2 * top_k, len(rows))
        vector_ranked = self._vector_search(query, rows, candidate_k)
        keyword_ranked = self._keyword_search(query, rows, candidate_k)
        return self._rrf_merge(vector_ranked, keyword_ranked, rows, top_k)

    def _vector_search(self, query: str, rows: list, top_k: int) -> list[tuple[int, float]]:
        query_vec = self._get_embedding(query)
        if query_vec is None:
            return []
        scores = []
        for idx, row in enumerate(rows):
            if row[5] is None:
                continue
            doc_vec = np.frombuffer(row[5], dtype=np.float32).copy()
            dot = float(np.dot(query_vec, doc_vec))
            norm = float(np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            scores.append((idx, dot / norm if norm > 0 else 0.0))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _keyword_search(self, query: str, rows: list, top_k: int) -> list[tuple[int, float]]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        query_counts = Counter(query_tokens)
        num_docs = len(rows)

        doc_freq: Counter = Counter()
        doc_tokens_list: list[list[str]] = []
        for row in rows:
            tokens = _tokenize(f"{row[3]} {row[4]}")
            doc_tokens_list.append(tokens)
            for t in set(tokens):
                doc_freq[t] += 1

        scores = []
        for idx, tokens in enumerate(doc_tokens_list):
            if not tokens:
                scores.append((idx, 0.0))
                continue
            tc = Counter(tokens)
            s = 0.0
            for term, qf in query_counts.items():
                tf = tc.get(term, 0)
                if tf == 0:
                    continue
                idf = math.log((num_docs + 1) / (doc_freq.get(term, 1) + 1)) + 1
                s += qf * (tf * 2.0) / (tf + 1.5) * idf  # BM25-lite
            scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _rrf_merge(self, vector_ranked: list[tuple[int, float]],
                   keyword_ranked: list[tuple[int, float]], rows: list,
                   top_k: int, k: int = 60) -> list[MemoryResult]:
        rrf: dict[int, float] = {}
        for rank, (idx, _) in enumerate(vector_ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(keyword_ranked):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
        if not rrf:
            return []
        results = []
        for idx, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            row = rows[idx]
            results.append(MemoryResult(row[1], row[2], row[3], row[4], round(score, 6)))
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

def _tokenize(text: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
            if len(t) > 1 and t not in _STOP_WORDS]


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
