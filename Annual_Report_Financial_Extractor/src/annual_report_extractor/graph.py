from __future__ import annotations

from langgraph.graph import END, StateGraph

from annual_report_extractor.models import GraphState
from annual_report_extractor.nodes.extract_statements import extract_statements
from annual_report_extractor.nodes.index_pages import index_pages
from annual_report_extractor.nodes.validate_output import validate_output
from annual_report_extractor.services.pdf_service import extract_pdf_pages


def load_pdf(state: GraphState) -> GraphState:
    return {"pages": extract_pdf_pages(state["pdf_path"])}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("load_pdf", load_pdf)
    graph.add_node("index_pages", index_pages)
    graph.add_node("extract_statements", extract_statements)
    graph.add_node("validate_output", validate_output)
    graph.set_entry_point("load_pdf")
    graph.add_edge("load_pdf", "index_pages")
    graph.add_edge("index_pages", "extract_statements")
    graph.add_edge("extract_statements", "validate_output")
    graph.add_edge("validate_output", END)
    return graph.compile()


def run_pipeline(pdf_path: str, company: dict) -> dict:
    graph = build_graph()
    initial_state: GraphState = {
        "pdf_path": pdf_path,
        "filename": pdf_path.split("\\")[-1],
        "company": company,
    }
    return graph.invoke(initial_state)
