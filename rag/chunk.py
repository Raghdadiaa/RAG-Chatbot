from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List

from rag.extract import PageBlock


@dataclass
class Chunk:
    chunk_id: str
    text: str
    meta: Dict[str, Any]


def _clip(s: str, n: int = 1200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def make_chunks(blocks: List[PageBlock], catalog_id: str = "product_catalog_01") -> List[Chunk]:
    """
    Simple, high-quality MVP chunking:
    - Treat each extracted block as a chunk (text/table) to preserve structure.
    - Add metadata needed for citations & grouping.
    """
    chunks: List[Chunk] = []
    idx = 0
    for b in blocks:
        if b.block_type not in ("text", "table"):
            continue
        idx += 1
        chunk_id = f"{catalog_id}_p{b.page}_{b.block_type}_{idx:04d}"
        text = b.text.strip()
        if not text:
            continue

        meta = {
            "catalog_id": catalog_id,
            "page": b.page,
            "block_type": b.block_type,
            "chunk_id": chunk_id,
            # store a short snippet for UI
            "snippet": _clip(text, 400),
        }
        # also store table index if exists
        if "table_index" in b.meta:
            meta["table_index"] = b.meta["table_index"]

        chunks.append(Chunk(chunk_id=chunk_id, text=text, meta=meta))
    return chunks
