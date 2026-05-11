# Annual Report Financial Extractor

This project is a submission-ready starter for the HENNGE annual report extractor assignment. It uses a LangGraph pipeline to:

- accept uploaded annual report PDFs,
- locate consolidated financial statement pages,
- extract structured line items with an LLM,
- run basic validation checks, and
- export the results into a multi-sheet Excel workbook.

## Tech Choices

- Python 3.11+
- LangGraph for the stateful extraction workflow
- PyMuPDF for PDF text extraction
- OpenAI via `langchain-openai` for structured statement extraction
- Openpyxl for Excel generation
- Streamlit for the UI

## Project Structure

```text
.
|-- app.py
|-- data/
|   `-- companies.json
|-- docs/
|   `-- technical_approach.md
|-- pyproject.toml
`-- src/
    `-- annual_report_extractor/
        |-- config.py
        |-- graph.py
        |-- models.py
        |-- prompts.py
        |-- nodes/
        |-- services/
        `-- utils/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
copy .env.example .env
```

Set `OPENAI_API_KEY` in `.env`.

## Run

```bash
streamlit run app.py
```

## Workflow

1. `load_pdf`: reads the PDF and extracts page text.
2. `index_pages`: finds candidate consolidated statement pages.
3. `extract_statements`: asks the LLM to return structured line items for each statement.
4. `validate_output`: checks completeness and basic accounting consistency.
5. `export_workbook`: writes the assignment-specific Excel structure.

## Current Gaps Before Submission

- The page indexer is rule-based and should be upgraded with an LLM fallback for harder report layouts.
- OCR support for scanned PDFs is not yet included.
- The current UI processes uploaded files against one selected company at a time.
- You still need to run the pipeline across at least 10 real company reports and attach the generated workbook.

## Deliverables Checklist

- Source code in this repository
- `financial_data.xlsx` generated after running the pipeline on 10 companies
- Demo video link
- Technical approach document in [docs/technical_approach.md](/c:/Users/dream/d%20drive/ai_projects/Annual_Report_Financial_Extractor/docs/technical_approach.md)
