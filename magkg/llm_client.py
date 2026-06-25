from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: str | Path = PROJECT_ROOT / ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class LLMClient:
    """Small OpenAI-compatible chat client used by optional LLM workflows."""

    def __init__(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        temperature: float | None = None,
    ) -> None:
        load_env_file()
        self.api_base = (api_base or os.getenv("KG_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.getenv("KG_API_KEY", "")
        self.model = model or os.getenv("KG_MODEL", "")
        self.timeout = int(timeout or os.getenv("KG_TIMEOUT", "120"))
        self.temperature = float(temperature if temperature is not None else os.getenv("KG_TEMPERATURE", "0"))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_base and self.api_key and self.model)

    def require_configured(self) -> None:
        if not self.is_configured:
            raise RuntimeError(
                "LLM API is not configured. Copy .env.example to .env and fill "
                "KG_API_BASE, KG_API_KEY, and KG_MODEL."
            )

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        self.require_configured()
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        return result["choices"][0]["message"].get("content", "")
