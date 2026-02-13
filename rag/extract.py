from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pdfplumber


@dataclass
class PageBlock:
    page: int                 # 1-based page number
    block_type: str           # "text" or "table"
    text: str                 # extracted content
    meta: Dict[str, Any]      # extra info (bbox, rows/cols, etc.)


def _clean(s: str) -> str:
    s = (s or "").replace("\u00ad", "")  # soft hyphen
    s = s.replace("\xa0", " ")           # non-breaking space
    # Normalize whitespace lightly
    s = "\n".join(line.rstrip() for line in s.splitlines())
    return s.strip()


def _table_to_text(table: List[List[Optional[str]]]) -> str:
    """
    Turn a table (rows x cols) into a readable, retrievable text block.
    We keep row structure and delimit columns with ' | ' so retrieval works on specs like Art.-Nr., PZN, etc.
    """
    rows = []
    for r in table:
        cells = [(_clean(c) if c else "") for c in r]
        # drop fully empty rows
        if any(cells):
            rows.append(" | ".join(cells))
    return _clean("\n".join(rows))


def extract_pdf_blocks(
    pdf_path: str,
    extract_tables: bool = True,
    max_pages: Optional[int] = None,
) -> List[PageBlock]:
    """
    Extracts per-page text plus (optionally) tables as separate blocks.
    Returns a list of PageBlock entries with page numbers and metadata for logging/observability.
    
    - If tables exist, we extract them into pipe-delimited rows.
    - We still keep the main page text for context.
    """
    blocks: List[PageBlock] = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        n_pages = min(total, max_pages) if max_pages else total

        for i in range(n_pages):
            page = pdf.pages[i]
            page_num = i + 1

            # 1) Main page text
            text = _clean(page.extract_text() or "")
            if text:
                blocks.append(
                    PageBlock(
                        page=page_num,
                        block_type="text",
                        text=text,
                        meta={"chars": len(text)},
                    )
                )

            # 2) Tables (as separate blocks)
            if extract_tables:
                try:
                    tables = page.extract_tables() or []
                except Exception as e:
                    tables = []
                    blocks.append(
                        PageBlock(
                            page=page_num,
                            block_type="table_error",
                            text=f"Table extraction error: {e}",
                            meta={},
                        )
                    )

                for t_idx, table in enumerate(tables):
                    t_text = _table_to_text(table)
                    if t_text:
                        blocks.append(
                            PageBlock(
                                page=page_num,
                                block_type="table",
                                text=t_text,
                                meta={
                                    "table_index": t_idx,
                                    "rows": len(table),
                                    "cols": max((len(r) for r in table), default=0),
                                    "chars": len(t_text),
                                },
                            )
                        )

    return blocks
