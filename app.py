from __future__ import annotations

import json
import os
import re
from pathlib import Path

import streamlit as st

from rag.extract import extract_pdf_blocks
from rag.chunk import make_chunks
from rag.index import build_or_rebuild_index, DEFAULT_EMBED_MODEL
from rag.retrieve import retrieve
from rag.answer import build_answer
from rag.observability import RunLogger

APP_TITLE = "Agentic RAG Chatbot — MVP (RAG + Logs)"


# ============================================================
# Minimal Groq helpers (optional)
# ============================================================
def groq_ready() -> bool:
    if not os.getenv("GROQ_API_KEY"):
        return False
    try:
        import groq  # noqa: F401
        return True
    except Exception:
        return False


def groq_chat(messages, max_tokens=220, temperature=0.0) -> str | None:
    if not groq_ready():
        return None
    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        resp = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None


def safe_json(text: str) -> dict | None:
    """Parse JSON even if model wraps it with extra text."""
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def llm_rewrite(query_en: str) -> dict:
    """Return: rewritten_query, intent, prefer_block_type, must_keywords."""
    system = (
        "Return ONLY valid JSON.\n"
        "You are improving retrieval over a German medical product catalog.\n"
        "Schema:\n"
        "{"
        "\"rewritten_query\":\"...\","
        "\"intent\":\"art_nr|pzn|definition|list_sizes|general\","
        "\"prefer_block_type\":\"text|table|any\","
        "\"must_keywords\":[\"...\",...]\n"
        "}\n"
        "Rules:\n"
        "- Keep rewritten_query short.\n"
        "- 'what is / used for' => intent=definition, prefer_block_type=text\n"
        "- Art.-Nr / PZN / list sizes => prefer_block_type=table\n"
    )
    out = groq_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": query_en}],
        max_tokens=180,
        temperature=0.0,
    )
    data = safe_json(out or "")
    if isinstance(data, dict):
        # basic sanitization
        return {
            "rewritten_query": str(data.get("rewritten_query") or query_en).strip(),
            "intent": str(data.get("intent") or "general").strip(),
            "prefer_block_type": str(data.get("prefer_block_type") or "any").strip(),
            "must_keywords": (data.get("must_keywords") or [])[:6] if isinstance(data.get("must_keywords"), list) else [],
        }

    # fallback heuristic
    q = (query_en or "").lower()
    if ("art" in q and "nr" in q) or "article number" in q:
        return {"rewritten_query": query_en, "intent": "art_nr", "prefer_block_type": "table", "must_keywords": []}
    if "pzn" in q:
        return {"rewritten_query": query_en, "intent": "pzn", "prefer_block_type": "table", "must_keywords": []}
    if "list" in q and "size" in q:
        return {"rewritten_query": query_en, "intent": "list_sizes", "prefer_block_type": "table", "must_keywords": []}
    if "what is" in q or "used for" in q:
        return {"rewritten_query": query_en, "intent": "definition", "prefer_block_type": "text", "must_keywords": []}
    return {"rewritten_query": query_en, "intent": "general", "prefer_block_type": "any", "must_keywords": []}


def prefer_block_type_sort(hits, prefer: str):
    """Soft reorder: put preferred block type first."""
    if prefer not in ("text", "table"):
        return hits

    def key(h):
        bt = str((h.meta or {}).get("block_type") or "").lower()
        return 0 if bt == prefer else 1

    return sorted(hits, key=key)


def llm_rerank(query_en: str, hits, keep_k: int):
    """LLM rerank: returns subset of hits in best-first order."""
    if not groq_ready() or not hits:
        return hits[:keep_k]

    candidates = []
    for i, h in enumerate(hits, start=1):
        snippet = (h.meta.get("snippet") or h.text or "").replace("\n", " ")
        snippet = re.sub(r"\s*\|\s*", " ", snippet)
        snippet = re.sub(r"\s+", " ", snippet).strip()[:420]
        candidates.append({"i": i, "page": h.meta.get("page", "?"), "block_type": h.meta.get("block_type", "?"), "snippet": snippet})

    system = (
        "Return ONLY JSON: {\"ranked\":[...]}.\n"
        "Pick the most relevant snippets for answering the question.\n"
        "Rank by 'contains the answer' > 'mentions product' > 'related'.\n"
    )
    user = json.dumps({"question": query_en, "keep_k": keep_k, "candidates": candidates}, ensure_ascii=False)

    out = groq_chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=180,
        temperature=0.0,
    )
    data = safe_json(out or "")
    ranked = data.get("ranked") if isinstance(data, dict) else None
    if not isinstance(ranked, list):
        return hits[:keep_k]

    picked = []
    for x in ranked:
        try:
            idx = int(x)
        except Exception:
            continue
        if 1 <= idx <= len(hits):
            picked.append(hits[idx - 1])
        if len(picked) >= keep_k:
            break
    return picked if picked else hits[:keep_k]


# ============================================================
# App
# ============================================================
def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("PDF → extraction → chunking → embeddings → retrieval → answer + citations + logs")

    root = Path(__file__).parent
    data_dir = root / "data"
    persist_dir = root / "storage" / "chroma"
    log_path = root / "logs" / "run.jsonl"
    logger = RunLogger(log_path)

    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        st.error("No PDFs found in ./data.")
        return

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("1) Ingest & Index")
        pdf_path = st.selectbox("Select PDF", options=[str(p) for p in pdf_files], index=0)

        extract_tables = st.checkbox("Extract tables", value=True)
        max_pages = st.number_input("Max pages (0 = all)", min_value=0, max_value=2000, value=0, step=1)
        max_pages = None if max_pages == 0 else int(max_pages)

        embed_model = st.text_input("Embedding model", value=DEFAULT_EMBED_MODEL)
        collection_name = st.text_input("Chroma collection", value="sanovio_catalog")

        if st.button("Build / Rebuild Index", type="primary"):
            with logger.step("extract_pdf", {"pdf": pdf_path, "tables": extract_tables, "max_pages": max_pages}):
                blocks = extract_pdf_blocks(pdf_path, extract_tables=extract_tables, max_pages=max_pages)

            with logger.step("chunking", {"n_blocks": len(blocks)}):
                chunks = make_chunks(blocks, catalog_id=Path(pdf_path).stem)

            with logger.step("index_build", {"n_chunks": len(chunks), "embed_model": embed_model, "collection": collection_name}):
                n, model_name = build_or_rebuild_index(
                    persist_dir=persist_dir,
                    collection_name=collection_name,
                    chunks=chunks,
                    embed_model_name=embed_model,
                )

            st.session_state["index_ready"] = True
            st.session_state["collection_name"] = collection_name
            st.session_state["embed_model"] = embed_model
            st.session_state["catalog_id"] = Path(pdf_path).stem
            st.success(f"Indexed {n} chunks ✅ (model: {model_name})")

        st.divider()
        st.subheader("2) Ask")

        groq_ok = groq_ready()
        st.caption(f"Groq enabled: {groq_ok}  |  Model: {os.getenv('GROQ_MODEL','llama-3.1-8b-instant')}")

        if not st.session_state.get("index_ready"):
            st.info("Build the index first.")
        else:
            query = st.text_area("Ask in English", value="What is the article number (Art.-Nr.) for Injekt Luer Solo 0.1 ml?", height=90)
            top_k = st.slider("Top-k (final)", min_value=3, max_value=10, value=5)

            use_rewrite = st.checkbox("LLM query rewrite", value=True)
            use_rerank = st.checkbox("LLM rerank", value=True)

            if st.button("Search & Answer"):
                collection = st.session_state["collection_name"]
                embed_model = st.session_state["embed_model"]
                catalog_id = st.session_state["catalog_id"]

                rewrite = {"rewritten_query": query, "intent": "general", "prefer_block_type": "any", "must_keywords": []}
                if groq_ok and use_rewrite:
                    with logger.step("llm_rewrite"):
                        rewrite = llm_rewrite(query)

                st.session_state["rewrite_debug"] = rewrite
                rewritten_query = rewrite.get("rewritten_query") or query
                prefer = rewrite.get("prefer_block_type") or "any"

                fetch_k = max(12, top_k * 3)
                with logger.step("retrieve", {"top_k": fetch_k, "query": rewritten_query}):
                    hits = retrieve(
                        persist_dir=persist_dir,
                        collection_name=collection,
                        query=rewritten_query,
                        top_k=fetch_k,
                        embed_model_name=embed_model,
                        catalog_filter=catalog_id,
                    )

                hits = prefer_block_type_sort(hits, prefer)

                if groq_ok and use_rerank:
                    with logger.step("llm_rerank", {"keep_k": top_k}):
                        hits = llm_rerank(query, hits, keep_k=top_k)
                else:
                    hits = hits[:top_k]

                with logger.step("answer", {"n_hits": len(hits)}):
                    ans = build_answer(query, hits)

                st.session_state["hits"] = hits
                st.session_state["ans"] = ans

    with right:
        st.subheader("Results")

        rewrite_debug = st.session_state.get("rewrite_debug")
        if rewrite_debug:
            st.markdown("### Query rewrite (debug)")
            st.json(rewrite_debug)

        ans = st.session_state.get("ans")
        hits = st.session_state.get("hits", [])

        if ans:
            st.markdown("### Answer (English)")
            st.write(ans.text)

            st.markdown("### Sources (page + snippet)")
            for i, c in enumerate(ans.citations, start=1):
                st.markdown(f"**{i}. p.{c.page}** — score={c.score:.3f} — chunk `{c.chunk_id}`")
                st.code(c.snippet)

        if hits:
            st.divider()
            st.markdown("### Retrieved chunks (debug)")
            for i, h in enumerate(hits, start=1):
                meta = h.meta or {}
                st.markdown(f"**{i}. p.{meta.get('page','?')}** — {meta.get('block_type','?')} — score={h.score:.3f}")
                st.code((meta.get("snippet") or h.text[:400])[:1200])

        st.divider()
        st.subheader("Step logs (latest)")
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").splitlines()[-40:]
            st.code("\n".join(lines), language="json")
        else:
            st.caption("No logs yet.")


if __name__ == "__main__":
    main()
