from __future__ import annotations

import json
import os
from typing import Any

import requests


class OllamaClientError(Exception):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = (
            base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "phi4-mini")
        self.timeout = timeout

    def chat_json(
        self,
        user_prompt: str,
        system_prompt: str,
        schema: dict[str, Any],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model or self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": schema,
                "options": {"temperature": temperature},
            },
            timeout=self.timeout,
        )
        self._raise_for_status(response)

        try:
            content = response.json()["message"]["content"]
        except (KeyError, TypeError, ValueError) as exc:
            raise OllamaClientError(
                f"Unexpected Ollama response shape: {response.text}"
            ) from exc

        return self._parse_json_content(content)

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise OllamaClientError(f"Ollama returned invalid JSON: {content}") from exc

        if not isinstance(parsed, dict):
            raise OllamaClientError(
                f"Expected JSON object but got: {type(parsed).__name__}"
            )
        return parsed

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise OllamaClientError(
                f"Ollama request failed with status {response.status_code}: {response.text}"
            ) from exc
