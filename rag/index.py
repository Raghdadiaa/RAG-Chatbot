from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from rag.chunk import Chunk


DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_chroma_client(persist_dir: Path) -> chromadb.PersistentClient:
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Disable Chroma telemetry to avoid noisy dependency/telemetry issues
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )



def build_or_rebuild_index(
    persist_dir: Path,
    collection_name: str,
    chunks: List[Chunk],
    embed_model_name: str = DEFAULT_EMBED_MODEL,
) -> Tuple[int, str]:
    """
    Rebuild index each time (simple + reliable for prototype).
    Returns (n_chunks_indexed, embed_model_name).
    """
    client = get_chroma_client(persist_dir)

    # Drop and recreate collection for clean rebuild
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    col = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    model = SentenceTransformer(embed_model_name)

    texts = [c.text for c in chunks]
    ids = [c.chunk_id for c in chunks]
    metas = [c.meta for c in chunks]

    # Embed in batches on CPU
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

    col.add(
        ids=ids,
        documents=texts,
        metadatas=metas,
        embeddings=embeddings.tolist(),
    )
    return len(chunks), embed_model_name
