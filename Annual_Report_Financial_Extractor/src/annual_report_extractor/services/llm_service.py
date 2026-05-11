from __future__ import annotations

import json
from typing import Any

from langchain_openai import ChatOpenAI

from annual_report_extractor.config import get_settings


class LLMService:
    def __init__(self) -> None:
        settings = get_settings()
        if settings.llm_provider != "openai":
            raise ValueError("Only OpenAI provider is configured in this starter project.")
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=0,
        )

    def invoke_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self.client.invoke(
            [
                ("system", system_prompt),
                (
                    "user",
                    user_prompt
                    + "\n\nReturn valid JSON only with double quotes and no markdown fences.",
                ),
            ]
        )
        content = response.content if isinstance(response.content, str) else "".join(response.content)
        return json.loads(content)
