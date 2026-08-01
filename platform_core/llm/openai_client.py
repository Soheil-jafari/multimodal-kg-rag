"""OpenAI implementation of :class:`~platform_core.llm.base.LLMClient`.

Reads ``OPENAI_API_KEY`` (and optional ``OPENAI_BASE_URL`` / ``OPENAI_MODEL``)
from the environment or the project ``.env`` (never committed). Used for both
answer generation and closed-schema KG extraction. Tracks token usage per call
and cumulatively so experiments can report cost.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import pathlib
from collections.abc import Sequence
from typing import Optional

from platform_core.llm.base import LLMClient

_ZERO = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


class OpenAIClient(LLMClient):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None) -> None:
        from dotenv import load_dotenv

        load_dotenv(pathlib.Path(__file__).resolve().parents[2] / ".env")
        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY not set — add it to the project .env or the environment."
            )
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        # The SDK's default of 2 retries is not enough for a long evaluation run: this
        # account's gpt-4o limit is 30k tokens/minute and an answer carries ~10 retrieved
        # chunks, so the ablation sits against the cap for its whole duration and a 429 is
        # the expected steady state, not an anomaly. An unretried 429 killed a six-config
        # run three configs in. The SDK backs off exponentially and honours Retry-After.
        retries = int(os.environ.get("OPENAI_MAX_RETRIES", "10"))
        base = (base_url or os.environ.get("OPENAI_BASE_URL") or "").strip()
        if base:
            self.client = OpenAI(api_key=key, base_url=base, max_retries=retries)
        else:
            os.environ.pop("OPENAI_BASE_URL", None)  # empty value breaks the SDK's env-var fallback
            self.client = OpenAI(api_key=key, max_retries=retries)
        self.last_usage = dict(_ZERO)
        self.total_usage = dict(_ZERO)

    def _track(self, usage) -> None:
        u = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        self.last_usage = u
        for k in self.total_usage:
            self.total_usage[k] += u[k]

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        self._track(r.usage)
        return r.choices[0].message.content or ""

    def answer_with_images(self, system: str, user: str, image_paths: Sequence[str],
                           temperature: float = 0.0) -> str:
        """Send crops inline as base64 data URLs (no upload step, no expiring links)."""
        content: list[dict] = [{"type": "text", "text": user}]
        for p in image_paths:
            mime = mimetypes.guess_type(p)[0] or "image/png"
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            content.append({
                "type": "image_url",
                # "high" detail: these are dense tables — downsampling loses the digits
                "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
            })
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
        )
        self._track(r.usage)
        return r.choices[0].message.content or ""

    def complete_json(self, system: str, user: str, schema: Optional[dict] = None,
                      temperature: float = 0.0) -> dict:
        r = self.client.chat.completions.create(
            model=self.model, temperature=temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        self._track(r.usage)
        txt = r.choices[0].message.content or "{}"
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return {"_raw": txt, "_parse_error": True}
