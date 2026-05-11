from __future__ import annotations

import re


def normalize_label(label: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", label.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_numeric_value(raw_value: str) -> float | None:
    cleaned = raw_value.replace(",", "").replace("(", "-").replace(")", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None
