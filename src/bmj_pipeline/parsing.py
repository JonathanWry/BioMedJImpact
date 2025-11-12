"""
XML parsing helpers and ISSN/year filters for PMC archives.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple
import io
import re
import tarfile

import pandas as pd
from lxml import etree as ET

YEAR_MIN_DEFAULT = 2016
YEAR_MAX_DEFAULT = 2023


@dataclass(frozen=True)
class ArticleMetadata:
    issn_eissn: str
    year: int
    title: str
    abstract: str
    meta_tags: list[str]
    tar_member: str


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def strip_markdown(text: str) -> str:
    text = text.replace("**", "")
    return re.sub(r"```.*?\n(.*?)```", r"\1", text, flags=re.DOTALL)


def _safe_itertext(element: ET.Element) -> str:
    return normalize_whitespace(" ".join(element.itertext()))


def extract_abstract(root: ET.Element) -> Optional[str]:
    abstract = root.xpath(".//*[local-name()='abstract']")
    if abstract:
        return _safe_itertext(abstract[0])
    return None


def extract_title(root: ET.Element) -> Optional[str]:
    title = root.xpath(".//*[local-name()='article-title']")
    if title:
        return _safe_itertext(title[0])
    return None


def collect_meta_tags(root: ET.Element) -> list[str]:
    tags = set()
    subj_xpath = ".//*[local-name()='article-categories']/*[local-name()='subj-group']/*[local-name()='subject']"
    kwd_xpath = ".//*[local-name()='kwd-group']/*[local-name()='kwd']"
    for el in root.xpath(f"{subj_xpath} | {kwd_xpath}"):
        if el.text:
            tags.add(normalize_whitespace(el.text))
    return sorted(tags)


def safe_split_issn(value: str) -> tuple[Optional[str], Optional[str]]:
    if isinstance(value, str) and "_" in value:
        return tuple(value.split("_", 1))
    return (None, None)


def load_journal_tables(path1: Path, path2: Path) -> tuple[pd.DataFrame, Set[str]]:
    j1 = pd.read_csv(path1)
    j2 = pd.read_csv(path2, encoding="ISO-8859-1")
    merged = j1.merge(j2, on="ISSN_EISSN", how="inner").copy()
    merged[["issn", "eissn"]] = merged["ISSN_EISSN"].apply(
        lambda s: pd.Series(safe_split_issn(s))
    )
    valid = set(merged["issn"].dropna()) | set(merged["eissn"].dropna())
    return merged, valid


def _iter_metadata(xml_bytes: bytes, valid_issns: Set[str], year_min: int, year_max: int) -> Optional[Tuple[str, int]]:
    buf = io.BytesIO(xml_bytes)
    year_ok = False
    issn_key = None
    year_val = None

    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1].lower()

    for _, el in ET.iterparse(buf, events=("start",)):
        tag = local(el.tag)
        if tag == "subject" and (el.text or "").strip().lower() == "retraction":
            return None
        if tag == "article-title" and (el.text or "").lower().startswith("[retracted"):
            return None
        if tag == "pub-date" and not year_ok:
            y = el.find("year")
            if y is not None and y.text:
                try:
                    year_val = int(y.text.strip())
                except ValueError:
                    return None
                year_ok = year_min <= year_val <= year_max
                if not year_ok:
                    return None
        if tag == "journal-meta" and issn_key is None:
            issn_el = el.find("issn")
            if issn_el is not None and issn_el.text:
                issn = issn_el.text.strip()
                if issn not in valid_issns:
                    return None
                issn_key = issn
        if year_ok and issn_key:
            break
    if year_ok and issn_key and year_val is not None:
        return issn_key, year_val
    return None


def iter_valid_articles(
    tar_path: Path,
    journal_df: pd.DataFrame,
    valid_issns: Set[str],
    processed: set[str],
    year_min: int = YEAR_MIN_DEFAULT,
    year_max: int = YEAR_MAX_DEFAULT,
) -> Iterator[ArticleMetadata]:
    with tarfile.open(tar_path, "r|gz") as tf:
        for member in tf:
            if not member.name.endswith(".xml") or member.name in processed:
                continue
            extracted = tf.extractfile(member)
            if not extracted:
                continue
            xml_bytes = extracted.read()
            metadata = _iter_metadata(xml_bytes, valid_issns, year_min, year_max)
            if metadata is None:
                continue
            issn_key_raw, year = metadata
            row = journal_df[
                (journal_df["issn"] == issn_key_raw) | (journal_df["eissn"] == issn_key_raw)
            ]
            if row.empty:
                continue
            issn_eissn = row.iloc[0]["ISSN_EISSN"]
            root = ET.fromstring(xml_bytes, ET.XMLParser(recover=True))
            abstract = extract_abstract(root)
            title = extract_title(root)
            if not abstract or not title:
                continue
            yield ArticleMetadata(
                issn_eissn=issn_eissn,
                year=year,
                title=title,
                abstract=abstract,
                meta_tags=collect_meta_tags(root),
                tar_member=member.name,
            )
