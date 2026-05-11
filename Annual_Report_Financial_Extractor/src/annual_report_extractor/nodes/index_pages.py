from __future__ import annotations

from collections import defaultdict

from annual_report_extractor.models import GraphState


STATEMENT_KEYWORDS = {
    "balance_sheet": ["balance sheet", "balance sheets", "statement of financial position"],
    "profit_and_loss": ["statement of profit and loss", "profit and loss", "income statement"],
    "cash_flow": ["cash flow statement", "statement of cash flows"],
}


def _is_consolidated(text: str) -> bool:
    lowered = text.lower()
    return "consolidated" in lowered and "standalone" not in lowered


def _candidate_score(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def index_pages(state: GraphState) -> GraphState:
    page_map: dict[str, list[int]] = defaultdict(list)
    for page in state["pages"]:
        text = page["text"]
        if not text.strip():
            continue
        for statement_type, keywords in STATEMENT_KEYWORDS.items():
            score = _candidate_score(text, keywords)
            if score and _is_consolidated(text):
                page_map[statement_type].append(page["page_number"])

    compact_page_map: dict[str, list[int]] = {}
    for statement_type, page_numbers in page_map.items():
        unique_pages = sorted(set(page_numbers))
        compact_page_map[statement_type] = unique_pages[:3]

    return {"page_map": compact_page_map}
