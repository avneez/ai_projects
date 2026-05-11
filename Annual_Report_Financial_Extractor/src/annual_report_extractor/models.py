from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


StatementType = Literal["balance_sheet", "profit_and_loss", "cash_flow"]


class LineItem(BaseModel):
    line_item: str = Field(..., description="Statement line item name.")
    current_fy_value: str = Field(..., description="Current fiscal year value as written in the report.")
    prior_fy_value: str = Field(..., description="Prior fiscal year value as written in the report.")
    unit: str = Field(..., description="Reporting unit such as Rs. crore, lakhs, millions, or actuals.")
    notes: str = Field(default="", description="Any note number, continuation hint, or extraction caveat.")


class StatementExtraction(BaseModel):
    statement_type: StatementType
    title: str
    fiscal_year: str
    prior_fiscal_year: str
    page_numbers: list[int] = Field(default_factory=list)
    line_items: list[LineItem] = Field(default_factory=list)


class WorkbookIndexRow(BaseModel):
    company_name: str
    ticker: str
    sector: str
    report_fy: str
    filename: str


class CompanyMetadata(BaseModel):
    name: str
    ticker: str
    sector: str


class GraphState(TypedDict, total=False):
    pdf_path: str
    filename: str
    company: dict
    pages: list[dict]
    page_map: dict
    statements: dict
    validation_issues: list[str]
    report_fy: str
