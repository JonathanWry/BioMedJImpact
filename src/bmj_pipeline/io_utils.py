"""
Shared I/O helpers: configs, resume files, JSONL streaming.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator
import json

import yaml


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, ensure_ascii=False)
            fh.write("\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            json.dump(row, fh, ensure_ascii=False)
            fh.write("\n")


def load_processed_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def append_processed(path: Path, items: Iterable[str]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for item in items:
            fh.write(f"{item}\n")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
