from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from annual_report_extractor.config import get_settings
from annual_report_extractor.graph import run_pipeline
from annual_report_extractor.models import WorkbookIndexRow
from annual_report_extractor.services.excel_service import export_workbook
from annual_report_extractor.utils.company_metadata import load_companies


st.set_page_config(page_title="Annual Report Financial Extractor", page_icon=":bar_chart:")

settings = get_settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)

st.title("Annual Report Financial Extractor")
st.caption("Upload annual report PDFs and generate a structured Excel workbook.")

companies = load_companies()
company_lookup = {f"{item.ticker} - {item.name}": item for item in companies}

selected_company_key = st.selectbox("Company", options=list(company_lookup.keys()))
uploaded_files = st.file_uploader(
    "Annual report PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload one or more annual reports. Use consolidated annual reports when possible.",
)

if st.button("Process PDFs", type="primary", disabled=not uploaded_files):
    company_results: list[dict] = []
    index_rows: list[WorkbookIndexRow] = []
    progress = st.progress(0.0, text="Starting extraction...")
    chosen_company = company_lookup[selected_company_key]

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        destination = settings.upload_dir / uploaded_file.name
        destination.write_bytes(uploaded_file.getbuffer())
        result = run_pipeline(str(destination), chosen_company.model_dump())
        company_results.append(result)
        index_rows.append(
            WorkbookIndexRow(
                company_name=chosen_company.name,
                ticker=chosen_company.ticker,
                sector=chosen_company.sector,
                report_fy=result.get("report_fy", ""),
                filename=uploaded_file.name,
            )
        )
        progress.progress(
            idx / len(uploaded_files),
            text=f"Processed {idx} of {len(uploaded_files)} file(s).",
        )
        with st.expander(f"Validation summary: {uploaded_file.name}", expanded=False):
            issues = result.get("validation_issues", [])
            if issues:
                for issue in issues:
                    st.warning(issue)
            else:
                st.success("No validation issues found in the basic checks.")

    output_path = export_workbook(index_rows, company_results, settings.output_dir / "financial_data.xlsx")
    st.success("Workbook generated successfully.")
    st.download_button(
        label="Download Excel workbook",
        data=Path(output_path).read_bytes(),
        file_name="financial_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
