# Technical Approach

## Objective

Build an end-to-end annual report financial extraction workflow that accepts a PDF, finds the consolidated financial statements, extracts structured data for balance sheet, profit and loss, and cash flow statements, validates the output, and exports a reviewable Excel workbook.

## Design

The solution uses a LangGraph `StateGraph` so each stage is isolated, testable, and easy to extend:

1. `load_pdf`
   - Reads the uploaded PDF with PyMuPDF.
   - Stores extracted text page by page.

2. `index_pages`
   - Uses statement keywords plus a consolidated/standalone filter to identify candidate pages.
   - Keeps the logic deterministic and inexpensive before using an LLM.

3. `extract_statements`
   - Collects the candidate pages for each statement.
   - Sends the statement text plus company metadata to the LLM.
   - Requests fully structured JSON covering all visible line items, both fiscal years, units, and notes.

4. `validate_output`
   - Checks that all three required statements exist.
   - Flags empty extractions.
   - Applies a basic balance sheet consistency check where possible.

5. `export_workbook`
   - Produces the required `Index`, `<Ticker>_BS`, `<Ticker>_PL`, and `<Ticker>_CF` sheets.

## LLM Choice

This starter uses OpenAI through `langchain-openai` because it is straightforward to integrate with LangGraph, supports predictable JSON-style extraction flows, and works well for table-heavy financial text when prompts are constrained carefully. The interface is isolated in `llm_service.py`, so swapping to Gemini, Claude, or an open-source model later is low-friction.

## Trade-offs

- A deterministic page indexer is cheaper and easier to debug, but it may miss unusual titles or split layouts.
- PDF text extraction works well for digital PDFs, but scanned reports will require OCR for strong coverage.
- The current validation is intentionally light; stronger checks can compare subtotals, sign conventions, and cross-statement consistency.

## Next Improvements

- Add OCR fallback for image PDFs.
- Add an LLM-assisted page index fallback when keyword matching is ambiguous.
- Improve numeric normalization across brackets, commas, negative values, and units.
- Add regression tests using a small set of known annual reports.
