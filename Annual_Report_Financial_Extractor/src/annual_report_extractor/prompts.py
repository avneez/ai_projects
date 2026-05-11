PAGE_INDEX_PROMPT = """
You are indexing an Indian annual report. Identify the page numbers for the consolidated financial statements.

Rules:
- Prefer consolidated statements over standalone statements.
- Look for balance sheet, statement of profit and loss, and cash flow statement.
- Return only statement pages that actually contain tabular financial values.
- If a statement spans multiple adjacent pages, include all of them.
"""


EXTRACTION_PROMPT = """
Extract every visible line item from the provided consolidated financial statement text.

Rules:
- Preserve the line item wording as closely as possible.
- Capture current fiscal year and prior fiscal year values.
- Capture the unit exactly if present; otherwise infer the dominant page unit.
- Add note references or caveats in notes.
- Ignore standalone statement data.
- Return a complete structured response.
"""
