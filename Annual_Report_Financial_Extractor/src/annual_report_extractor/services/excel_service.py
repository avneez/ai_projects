from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Font

from annual_report_extractor.models import WorkbookIndexRow


def export_workbook(
    index_rows: list[WorkbookIndexRow],
    company_results: list[dict],
    output_path: str | Path,
) -> Path:
    workbook = Workbook()
    index_sheet = workbook.active
    index_sheet.title = "Index"
    index_sheet.append(["Company name", "Ticker", "Sector", "Report FY", "Filename"])
    for cell in index_sheet[1]:
        cell.font = Font(bold=True)
    for row in index_rows:
        index_sheet.append(
            [row.company_name, row.ticker, row.sector, row.report_fy, row.filename]
        )

    sheet_suffix = {
        "balance_sheet": "BS",
        "profit_and_loss": "PL",
        "cash_flow": "CF",
    }
    for result in company_results:
        ticker = result["company"]["ticker"]
        statements = result["statements"]
        for key, statement in statements.items():
            sheet = workbook.create_sheet(f"{ticker}_{sheet_suffix[key]}")
            sheet.append(
                [
                    "Line Item",
                    "Current FY Value",
                    "Prior FY Value",
                    "Unit (₹ Cr / Lakhs)",
                    "Notes",
                ]
            )
            for cell in sheet[1]:
                cell.font = Font(bold=True)
            for item in statement["line_items"]:
                sheet.append(
                    [
                        item["line_item"],
                        item["current_fy_value"],
                        item["prior_fy_value"],
                        item["unit"],
                        item["notes"],
                    ]
                )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    return output_path
