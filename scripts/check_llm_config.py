from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from magkg.llm_client import LLMClient, load_env_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Check optional LLM API configuration.")
    parser.add_argument("--env", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--ping", action="store_true", help="Send a minimal test request to the configured endpoint.")
    args = parser.parse_args()

    load_env_file(args.env)
    client = LLMClient()
    payload = {
        "env_file": str(args.env),
        "api_base_set": bool(client.api_base),
        "api_key_set": bool(client.api_key),
        "model": client.model or None,
        "configured": client.is_configured,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.ping:
        text = client.chat("You are a concise assistant.", "Reply with MAGKG.")
        print(text)


if __name__ == "__main__":
    main()
