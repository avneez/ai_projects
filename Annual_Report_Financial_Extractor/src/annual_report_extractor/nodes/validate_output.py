from __future__ import annotations

from annual_report_extractor.models import GraphState
from annual_report_extractor.utils.validation import normalize_label, parse_numeric_value


def _find_first_value(line_items: list[dict], keywords: tuple[str, ...]) -> float | None:
    for item in line_items:
        label = normalize_label(item["line_item"])
        if any(keyword in label for keyword in keywords):
            value = parse_numeric_value(item["current_fy_value"])
            if value is not None:
                return value
    return None


def validate_output(state: GraphState) -> GraphState:
    issues: list[str] = []
    statements = state.get("statements", {})
    required_statements = {"balance_sheet", "profit_and_loss", "cash_flow"}
    missing_statements = sorted(required_statements - statements.keys())
    if missing_statements:
        issues.append(f"Missing statements: {', '.join(missing_statements)}")

    balance_sheet = statements.get("balance_sheet")
    if balance_sheet:
        line_items = balance_sheet["line_items"]
        total_assets = _find_first_value(line_items, ("total assets",))
        total_equity = _find_first_value(line_items, ("total equity", "equity attributable"))
        total_liabilities = _find_first_value(
            line_items,
            ("total liabilities", "total liabilities and equity", "non current liabilities"),
        )
        if total_assets is None:
            issues.append("Could not find Total Assets for balance sheet validation.")
        if total_assets is not None and total_equity is not None and total_liabilities is not None:
            if abs(total_assets - (total_equity + total_liabilities)) > 1.0:
                issues.append("Balance sheet equation failed: Assets != Liabilities + Equity.")

    for statement_name, statement in statements.items():
        if not statement["line_items"]:
            issues.append(f"{statement_name} contains no extracted line items.")

    return {
        "validation_issues": issues,
        "report_fy": next(
            (
                statement["fiscal_year"]
                for statement in statements.values()
                if statement.get("fiscal_year")
            ),
            "",
        ),
    }
