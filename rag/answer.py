from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from rag.retrieve import RetrievedChunk


# -----------------------------
# Public return objects
# -----------------------------
@dataclass
class Citation:
    page: int
    snippet: str
    chunk_id: str
    score: float


@dataclass
class Answer:
    text: str
    citations: List[Citation]


# -----------------------------
# Regex helpers
# -----------------------------
ARTNR_RE = re.compile(r"\b(\d{6,9}[A-Z]?)\b")   # e.g., 4606027V
PZN_RE = re.compile(r"\b(\d{8})\b")            # e.g., 02057895
SIZE_RE = re.compile(r"\b(\d+(?:[.,]\d+)?\s*ml)\b", re.IGNORECASE)

CTX_WINDOW = 160


def _shorten(s: str, n: int = 520) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


def _clean(s: str) -> str:
    """Remove table pipes + collapse whitespace."""
    s = (s or "").replace("\n", " ")
    s = re.sub(r"\s*\|\s*", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _infer_intent(q: str) -> str:
    ql = (q or "").lower()
    q_compact = re.sub(r"[^a-z0-9]+", "", ql)

    if ("art" in ql and "nr" in ql) or ("artnr" in q_compact) or ("artikelnummer" in ql) or ("article number" in ql):
        return "art_nr"
    if "pzn" in ql or "pzn" in q_compact:
        return "pzn"
    if ("list" in ql and "size" in ql) or ("available" in ql and "size" in ql):
        return "list_sizes"
    if ql.startswith("what is") or "used for" in ql or "definition" in ql:
        return "definition"
    return "general"


def _size_constraint(q: str) -> Optional[str]:
    m = SIZE_RE.search((q or "").lower())
    if not m:
        return None
    return m.group(1).replace(" ", "").replace(",", ".")


def _build_citations(retrieved: List[RetrievedChunk], k: int = 5) -> List[Citation]:
    cits: List[Citation] = []
    for r in retrieved[:k]:
        page = int((r.meta or {}).get("page", -1))
        snippet = (r.meta or {}).get("snippet") or r.text or ""
        cits.append(
            Citation(
                page=page,
                snippet=_shorten(_clean(snippet), 520),
                chunk_id=r.chunk_id,
                score=r.score,
            )
        )
    return cits


def _sources_line(citations: List[Citation]) -> str:
    pages = sorted({c.page for c in citations if c.page != -1})
    return f"Sources: {', '.join(f'p.{p}' for p in pages)}." if pages else ""


# -----------------------------
# Optional Groq (LLM)
# -----------------------------
def _llm_enabled() -> bool:
    if not os.getenv("GROQ_API_KEY"):
        return False
    try:
        import groq  # noqa: F401
        return True
    except Exception:
        return False


def _groq_answer(query_en: str, citations: List[Citation], max_tokens: int = 220) -> Optional[str]:
    """Generate a short English answer grounded ONLY in snippets."""
    if not _llm_enabled():
        return None

    try:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    except Exception:
        return None

    ctx_lines = []
    for i, c in enumerate(citations[:5], start=1):
        page = f"p.{c.page}" if c.page != -1 else "unknown page"
        ctx_lines.append(f"[{i}] ({page}) {_shorten(_clean(c.snippet), 420)}")
    ctx = "\n".join(ctx_lines)

    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    system = (
        "You answer questions about a German medical product PDF catalog.\n"
        "Rules:\n"
        "- Use ONLY the provided snippets as evidence.\n"
        "- Answer in English.\n"
        "- Be concise and helpful.\n"
        "- Do NOT output table pipes '|' or raw table rows.\n"
        "- If the question asks for Art.-Nr or PZN, output the value(s) clearly.\n"
        "- End with: Sources: p.X, p.Y.\n"
    )

    user = f"Question: {query_en}\n\nSnippets:\n{ctx}"

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return text or None
    except Exception:
        return None


# -----------------------------
# Heuristic extractors (no LLM)
# -----------------------------
def _extract_ctx_hits(blob: str, regex: re.Pattern) -> List[tuple[str, str]]:
    out: List[tuple[str, str]] = []
    for m in regex.finditer(blob):
        val = m.group(1)
        start = max(0, m.start() - CTX_WINDOW)
        end = min(len(blob), m.end() + CTX_WINDOW)
        out.append((val, blob[start:end]))
    return out


def _dedupe(values: List[tuple[str, str]]) -> List[tuple[str, str]]:
    seen = set()
    out = []
    for v, ctx in values:
        if v in seen:
            continue
        seen.add(v)
        out.append((v, ctx))
    return out


def _pick_best_by_size(values: List[tuple[str, str]], size: Optional[str]) -> tuple[Optional[str], List[str]]:
    """Pick best match; return (best, alts)."""
    if not values:
        return None, []

    if not size:
        best = values[0][0]
        alts = [v for v, _ in values[1:4]]
        return best, alts

    # rank: contexts containing the exact size first
    ranked = sorted(values, key=lambda x: (size not in x[1].replace(" ", "").replace(",", ".")),)
    best = ranked[0][0]
    alts = [v for v, _ in ranked[1:4]]
    return best, alts


def _fallback_snippet_answer(query_en: str, citations: List[Citation]) -> str:
    lines = ["Here’s what I found in the catalog (based on the retrieved passages):", ""]
    for c in citations[:3]:
        src = f"p.{c.page}" if c.page != -1 else "source"
        lines.append(f"- ({src}) {_shorten(_clean(c.snippet), 420)}")
    sl = _sources_line(citations)
    if sl:
        lines.append("")
        lines.append(sl)
    return "\n".join(lines)


# -----------------------------
# Main entry
# -----------------------------
def build_answer(query_en: str, retrieved: List[RetrievedChunk]) -> Answer:
    if not retrieved:
        return Answer("I couldn’t find relevant information in the catalog for this question.", [])

    intent = _infer_intent(query_en)
    size = _size_constraint(query_en)

    citations = _build_citations(retrieved, k=5)

    # Build blob from top retrieved text/snippets
    blob = "\n".join([(r.meta or {}).get("snippet") or r.text or "" for r in retrieved[:10]])
    blob = _clean(blob)

    # 1) Art.-Nr / PZN / list_sizes => do deterministic extraction first (safer than LLM)
    if intent == "art_nr":
        vals = _dedupe(_extract_ctx_hits(blob, ARTNR_RE))
        # filter out pure 8-digit numbers (likely PZN) unless they have trailing letter
        vals = [(v, ctx) for (v, ctx) in vals if not re.fullmatch(r"\d{8}", v)]
        best, alts = _pick_best_by_size(vals, size)

        if not best:
            # if Groq available, let it answer; else snippet-based
            llm = _groq_answer(query_en, citations)
            return Answer(llm, citations) if llm else Answer(_fallback_snippet_answer(query_en, citations), citations)

        lines = [f"Answer: The best match for the article number (Art.-Nr.) is **{best}**."]
        if alts:
            lines += ["", "Other close matches:"]
            lines += [f"- {a}" for a in alts]
        sl = _sources_line(citations)
        if sl:
            lines += ["", sl]
        return Answer("\n".join(lines), citations)

    if intent == "pzn":
        vals = _dedupe(_extract_ctx_hits(blob, PZN_RE))
        best = vals[0][0] if vals else None
        alts = [v for v, _ in vals[1:4]]

        if not best:
            llm = _groq_answer(query_en, citations)
            return Answer(llm, citations) if llm else Answer(_fallback_snippet_answer(query_en, citations), citations)

        lines = [f"Answer: The best matching PZN is **{best}**."]
        if alts:
            lines += ["", "Other close matches:"]
            lines += [f"- {a}" for a in alts]
        sl = _sources_line(citations)
        if sl:
            lines += ["", sl]
        return Answer("\n".join(lines), citations)

    if intent == "list_sizes":
        sizes = []
        for m in SIZE_RE.finditer(blob):
            sizes.append(m.group(1).replace(" ", "").replace(",", "."))
        sizes = sorted(set(sizes))

        if not sizes:
            llm = _groq_answer(query_en, citations)
            return Answer(llm, citations) if llm else Answer(_fallback_snippet_answer(query_en, citations), citations)

        lines = ["Answer: Available sizes mentioned in the retrieved passages:"]
        lines += [f"- {s}" for s in sizes[:15]]
        sl = _sources_line(citations)
        if sl:
            lines += ["", sl]
        return Answer("\n".join(lines), citations)

    # 2) Definition / general => prefer Groq for clean English
    llm = _groq_answer(query_en, citations)
    if llm:
        return Answer(llm, citations)

    # 3) fallback: snippet-based (still English framing, but German evidence)
    return Answer(_fallback_snippet_answer(query_en, citations), citations)
