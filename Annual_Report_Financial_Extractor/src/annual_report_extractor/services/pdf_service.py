from __future__ import annotations

from pathlib import Path

import fitz


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    document = fitz.open(pdf_path)
    pages: list[dict] = []
    try:
        for idx, page in enumerate(document, start=1):
            text = page.get_text("text")
            pages.append({"page_number": idx, "text": text})
    finally:
        document.close()
    return pages


def collect_pages(pages: list[dict], page_numbers: list[int]) -> str:
    lookup = {page["page_number"]: page["text"] for page in pages}
    ordered_numbers = sorted(set(page_numbers))
    chunks = []
    for number in ordered_numbers:
        text = lookup.get(number, "")
        chunks.append(f"[Page {number}]\n{text}")
    return "\n\n".join(chunks)
