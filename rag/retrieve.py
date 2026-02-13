from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from rag.index import get_chroma_client, DEFAULT_EMBED_MODEL


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any]
    score: float  # similarity score (approx)


@lru_cache(maxsize=2)
def _get_embedder(model_name: str) -> SentenceTransformer:
    # loads once per process, reused for all queries
    return SentenceTransformer(model_name)


def retrieve(
    persist_dir: Path,
    collection_name: str,
    query: str,
    top_k: int = 5,
    embed_model_name: str = DEFAULT_EMBED_MODEL,
    catalog_filter: Optional[str] = None,
) -> List[RetrievedChunk]:
    client = get_chroma_client(persist_dir)
    col = client.get_or_create_collection(name=collection_name)

    model = _get_embedder(embed_model_name)
    q_emb = model.encode([query], normalize_embeddings=True).tolist()[0]

    where = {"catalog_id": catalog_filter} if catalog_filter else None

    res = col.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    out: List[RetrievedChunk] = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]  # usually cosine distance

    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        score = 1.0 - float(dist) if dist is not None else 0.0
        out.append(RetrievedChunk(chunk_id=cid, text=doc, meta=meta or {}, score=score))

    return out
