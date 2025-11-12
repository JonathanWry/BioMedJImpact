"""
Streaming statistics utilities for journal-level tracking and checkpointing.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple
import json


@dataclass
class ArticleRecord:
    meta_tags: set[str] = field(default_factory=set)
    form_a_keywords: set[str] = field(default_factory=set)
    form_b_categories: set[str] = field(default_factory=set)


@dataclass
class JournalYearStats:
    tag_counts: Counter[str] = field(default_factory=Counter)
    category_counts: Counter[str] = field(default_factory=Counter)
    journal_count: int = 0
    ai_count: int = 0
    articles: Dict[str, ArticleRecord] = field(default_factory=dict)

    def increment_total(self, amount: int = 1) -> None:
        self.journal_count += amount

    def add_positive_article(
        self,
        title: str,
        keywords: Iterable[str],
        categories: Iterable[str],
        meta_tags: Iterable[str],
    ) -> None:
        self.ai_count += 1
        self.tag_counts.update(keywords)
        self.category_counts.update(cat for cat in categories if cat and cat != "NONE")
        record = self.articles.setdefault(title, ArticleRecord())
        record.meta_tags.update(meta_tags)
        record.form_a_keywords.update(keywords)
        record.form_b_categories.update(categories)

    def to_json(self, issn_key: str, year: int) -> dict:
        return {
            "ISSN_EISSN": issn_key,
            "year": year,
            "Tag_Counts": dict(self.tag_counts),
            "Category_Counts": dict(self.category_counts),
            "Journal_Count": self.journal_count,
            "AI_Count": self.ai_count,
            "Articles": {
                title: {
                    "Meta_Tags": sorted(rec.meta_tags),
                    "Form_A_Keywords": sorted(rec.form_a_keywords),
                    "Form_B_Categories": sorted(rec.form_b_categories),
                }
                for title, rec in self.articles.items()
            },
        }

    @classmethod
    def from_json(cls, payload: dict) -> Tuple[str, int, "JournalYearStats"]:
        stats = cls()
        stats.tag_counts.update(payload.get("Tag_Counts", {}))
        stats.category_counts.update(payload.get("Category_Counts", {}))
        stats.journal_count = payload.get("Journal_Count", 0)
        stats.ai_count = payload.get("AI_Count", 0)
        for title, rec in payload.get("Articles", {}).items():
            stats.articles[title] = ArticleRecord(
                meta_tags=set(rec.get("Meta_Tags", [])),
                form_a_keywords=set(rec.get("Form_A_Keywords", [])),
                form_b_categories=set(rec.get("Form_B_Categories", [])),
            )
        return payload["ISSN_EISSN"], payload["year"], stats


class StatsStore:
    def __init__(self) -> None:
        self._data: dict[str, dict[int, JournalYearStats]] = defaultdict(dict)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                payload = json.loads(line)
                issn, year, stats = JournalYearStats.from_json(payload)
                self._data[issn][year] = stats

    def increment_total(self, issn_eissn: str, year: int) -> None:
        stats = self._data.setdefault(issn_eissn, {}).setdefault(year, JournalYearStats())
        stats.increment_total()

    def add_positive_article(
        self,
        issn_eissn: str,
        year: int,
        title: str,
        keywords: Iterable[str],
        categories: Iterable[str],
        meta_tags: Iterable[str],
    ) -> None:
        stats = self._data.setdefault(issn_eissn, {}).setdefault(year, JournalYearStats())
        stats.add_positive_article(title, keywords, categories, meta_tags)

    def to_jsonl(self) -> Iterator[dict]:
        for issn, year_map in self._data.items():
            for year, stats in year_map.items():
                yield stats.to_json(issn, year)

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for payload in self.to_jsonl():
                json.dump(payload, fh, ensure_ascii=False)
                fh.write("\n")


def consolidate_stats(files: Iterable[Path]) -> dict[tuple[str, int], JournalYearStats]:
    store: dict[tuple[str, int], JournalYearStats] = {}
    for fp in files:
        with fp.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                issn, year, stats = JournalYearStats.from_json(json.loads(line))
                key = (issn, year)
                if key not in store:
                    store[key] = stats
                else:
                    existing = store[key]
                    existing.tag_counts.update(stats.tag_counts)
                    existing.category_counts.update(stats.category_counts)
                    existing.journal_count += stats.journal_count
                    existing.ai_count += stats.ai_count
                    for title, rec in stats.articles.items():
                        article = existing.articles.setdefault(title, ArticleRecord())
                        article.meta_tags.update(rec.meta_tags)
                        article.form_a_keywords.update(rec.form_a_keywords)
                        article.form_b_categories.update(rec.form_b_categories)
    return store
