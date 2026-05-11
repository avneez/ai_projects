from __future__ import annotations

import json

from annual_report_extractor.models import GraphState, StatementExtraction
from annual_report_extractor.prompts import EXTRACTION_PROMPT
from annual_report_extractor.services.llm_service import LLMService
from annual_report_extractor.services.pdf_service import collect_pages


def extract_statements(state: GraphState) -> GraphState:
    llm = LLMService()
    statements: dict[str, dict] = {}
    for statement_type in ("balance_sheet", "profit_and_loss", "cash_flow"):
        page_numbers = state.get("page_map", {}).get(statement_type, [])
        if not page_numbers:
            continue
        statement_text = collect_pages(state["pages"], page_numbers)
        payload = llm.invoke_json(
            EXTRACTION_PROMPT,
            (
                f"Company metadata: {json.dumps(state['company'])}\n"
                f"Statement type: {statement_type}\n"
                f"Statement text:\n{statement_text}\n\n"
                "Return JSON with keys: statement_type, title, fiscal_year, prior_fiscal_year, page_numbers, line_items. "
                "Each line item must include line_item, current_fy_value, prior_fy_value, unit, notes."
            ),
        )
        payload["statement_type"] = statement_type
        payload["page_numbers"] = page_numbers
        statements[statement_type] = StatementExtraction.model_validate(payload).model_dump()
    return {"statements": statements}
