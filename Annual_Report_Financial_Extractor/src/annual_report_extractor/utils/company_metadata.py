from __future__ import annotations

import json
from pathlib import Path

from annual_report_extractor.models import CompanyMetadata


def load_companies(data_path: str | Path = "data/companies.json") -> list[CompanyMetadata]:
    payload = json.loads(Path(data_path).read_text(encoding="utf-8"))
    return [CompanyMetadata.model_validate(item) for item in payload]
