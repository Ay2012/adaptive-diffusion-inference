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
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "phi4-mini")
        self.timeout = timeout

    def list_models(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/tags",
            timeout=self.timeout,
        )
        self._raise_for_status(response)
        return response.json()

    def is_running(self) -> bool:
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.0,
        stream: bool = False,
        keep_alive: int | str = 0,
        response_format: str | dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
            "keep_alive": keep_alive,
            "options": {
                "temperature": temperature,
                **(options or {}),
            },
        }

        if response_format is not None:
            payload["format"] = response_format

        response = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        self._raise_for_status(response)
        return response.json()

    def chat_json(
        self,
        user_prompt: str,
        system_prompt: str | None = None,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        keep_alive: int | str = 0,
    ) -> dict[str, Any]:
        messages: list[dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": user_prompt})

        raw_response = self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            stream=False,
            keep_alive=keep_alive,
            response_format=schema if schema is not None else "json",
        )

        try:
            content = raw_response["message"]["content"]
        except KeyError as exc:
            raise OllamaClientError(f"Unexpected Ollama response shape: {raw_response}") from exc

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
            raise OllamaClientError(f"Expected JSON object but got: {type(parsed).__name__}")

        return parsed

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text
            raise OllamaClientError(
                f"Ollama request failed with status {response.status_code}: {body}"
            ) from exc