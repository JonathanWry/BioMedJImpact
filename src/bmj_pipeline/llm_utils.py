"""
Reusable helpers for Gemma/vLLM prompting.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from vllm import LLM, SamplingParams

from .parsing import strip_markdown
from .prompts import (
    SYSTEM_PROMPT_STAGE1,
    SYSTEM_PROMPT_STAGE2,
    SYSTEM_PROMPT_STAGE3,
)


def render_gemma(messages: list[dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m["role"]
        if role == "assistant":
            role = "model"
        parts.append(f"<start_of_turn>{role}\n{m['content']}\n<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def _find_block(label: str, text: str) -> str | None:
    pattern = re.compile(rf"(?is){label}\s*[:：]?\s*(\{{[^{{}}]*\}}|\[[^\[\]]*\])")
    match = pattern.search(text)
    return match.group(1) if match else None


def _parse_items(block: str | None) -> set[str]:
    if not block:
        return set()
    payload = block.strip()
    if payload.startswith("{") and payload.endswith("}"):
        payload = f"[{payload[1:-1]}]"
    try:
        loaded = json.loads(payload)
        if isinstance(loaded, list):
            return {str(item).strip().strip('"').strip("'") for item in loaded if str(item).strip()}
        if isinstance(loaded, str):
            return {part.strip() for part in loaded.split(",") if part.strip()}
    except json.JSONDecodeError:
        payload = payload.strip("{}[]")
        return {part.strip().strip('"').strip("'") for part in payload.split(",") if part.strip()}
    return set()


def stage_yes_no(llm: LLM, abstract: str, temperature: float = 0.1) -> bool:
    sp = SamplingParams(temperature=temperature, max_tokens=256, stop=["<end_of_turn>"])
    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT_STAGE1}\n\n{abstract}"}]
    prompt = render_gemma(messages)
    outputs = llm.generate([prompt], sp)
    content = outputs[0].outputs[0].text.strip()
    return "YES" in content


def stage_keywords_categories(llm: LLM, abstract: str, temperature: float = 0.5) -> tuple[set[str], set[str]]:
    sp = SamplingParams(temperature=temperature, max_tokens=2048, stop=["<end_of_turn>"])
    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT_STAGE2}\n\n{abstract}"}]
    prompt = render_gemma(messages)
    outputs = llm.generate([prompt], sp)
    content = strip_markdown(outputs[0].outputs[0].text.strip())
    form_a_block = _find_block(r"FORM\s*A", content)
    form_b_block = _find_block(r"FORM\s*B", content)
    return _parse_items(form_a_block), _parse_items(form_b_block)


def stage_keywords_check(llm: LLM, keywords: Iterable[str], temperature: float = 0.1) -> bool:
    sp = SamplingParams(temperature=temperature, max_tokens=256, stop=["<end_of_turn>"])
    keyword_payload = "[" + ", ".join(f'"{kw}"' for kw in keywords) + "]"
    messages = [{"role": "user", "content": f"{SYSTEM_PROMPT_STAGE3}\n\n{keyword_payload}"}]
    prompt = render_gemma(messages)
    outputs = llm.generate([prompt], sp)
    content = outputs[0].outputs[0].text.strip()
    return "YES" in content


__all__ = [
    "stage_yes_no",
    "stage_keywords_categories",
    "stage_keywords_check",
    "render_gemma",
]
