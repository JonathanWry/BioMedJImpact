"""
Balanced sampling utility for human evaluation studies.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vllm import LLM

from . import __version__
from .io_utils import (
    append_processed,
    ensure_parent,
    load_processed_set,
    read_yaml,
    write_jsonl,
)
from .llm_utils import (
    stage_keywords_categories,
    stage_keywords_check,
    stage_yes_no,
)
from .parsing import iter_valid_articles, load_journal_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce balanced AI vs non-AI samples for human eval")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path (see configs/sample.yaml)")
    parser.add_argument("--csv-out", type=Path, dest="csv_out")
    parser.add_argument("--excel-out", type=Path, dest="excel_out")
    parser.add_argument("--jsonl-out", type=Path, dest="jsonl_out")
    parser.add_argument("--version", action="version", version=f"bmj-pipeline {__version__}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    outputs = config.get("outputs", {})
    csv_out = Path(args.csv_out or outputs["csv"]).expanduser()
    excel_out = Path(args.excel_out or outputs["excel"]).expanduser()
    jsonl_out = Path(args.jsonl_out or outputs["jsonl"]).expanduser()

    ensure_parent(csv_out)
    ensure_parent(excel_out)
    ensure_parent(jsonl_out)

    archive = Path(config["archive"]).expanduser()
    processed_path = Path(config["processed_file"]).expanduser() if config.get("processed_file") else None
    if processed_path:
        ensure_parent(processed_path)
    journal1 = Path(config["journal_csvs"]["dataset1"]).expanduser()
    journal2 = Path(config["journal_csvs"]["dataset2"]).expanduser()

    journal_df, valid_issns = load_journal_tables(journal1, journal2)
    processed = load_processed_set(processed_path) if processed_path else set()

    model_cfg = config.get("model", {})
    model_ref = model_cfg.get("local_path") or model_cfg.get("repo_id")
    if not model_ref:
        raise ValueError("model.local_path or model.repo_id must be provided in the config.")
    llm_kwargs = {
        "model": model_ref,
        "max_model_len": model_cfg.get("max_model_len", 40_000),
        "gpu_memory_utilization": model_cfg.get("gpu_memory_utilization", 0.9),
    }
    if model_cfg.get("tensor_parallel_size"):
        llm_kwargs["tensor_parallel_size"] = model_cfg["tensor_parallel_size"]
    llm = LLM(**llm_kwargs)

    targets = config.get("balanced_targets", {"positive": 75, "negative": 75})
    pos_target = targets.get("positive", 75)
    neg_target = targets.get("negative", 75)

    samples: list[dict] = []
    logs: list[dict] = []
    pos_count = 0
    neg_count = 0

    iterator = iter_valid_articles(
        tar_path=archive,
        journal_df=journal_df,
        valid_issns=valid_issns,
        processed=processed,
        year_min=config.get("year_range", {}).get("min", 2016),
        year_max=config.get("year_range", {}).get("max", 2023),
    )

    for article in tqdm(iterator, desc="Balanced sampling", unit="xml"):
        if pos_count >= pos_target and neg_count >= neg_target:
            break
        processed.add(article.tar_member)

        # Match the extraction pipeline's first-stage behavior to guarantee identical inclusion criteria.
        abstract_for_stage1 = f"{article.abstract}\n{article.abstract}"
        is_ai = stage_yes_no(llm, abstract_for_stage1)
        if not is_ai:
            if neg_count < neg_target:
                neg_count += 1
                record = {
                    "label": "negative",
                    "title": article.title,
                    "issn_eissn": article.issn_eissn,
                    "year": article.year,
                    "meta_tags": article.meta_tags,
                    "keywords": [],
                    "categories": [],
                    "tar_member": article.tar_member,
                    "abstract": article.abstract,
                    "stage": "stage1_reject",
                }
                samples.append(record)
                logs.append(record)
            continue

        keywords, categories = stage_keywords_categories(llm, article.abstract)
        if not keywords or not categories:
            continue
        if not stage_keywords_check(llm, keywords):
            continue

        if pos_count < pos_target:
            pos_count += 1
            record = {
                "label": "positive",
                "title": article.title,
                "issn_eissn": article.issn_eissn,
                "year": article.year,
                "meta_tags": article.meta_tags,
                "keywords": sorted(keywords),
                "categories": sorted(categories),
                "tar_member": article.tar_member,
                "abstract": article.abstract,
                "stage": "three_stage_pass",
            }
            samples.append(record)
            logs.append(record)

    if not samples:
        raise RuntimeError("No samples collected; check targets or filters.")

    df = pd.DataFrame(samples)
    df.to_csv(csv_out, index=False)
    df.to_excel(excel_out, index=False)
    write_jsonl(jsonl_out, logs)

    if processed_path and samples:
        append_processed(processed_path, [s["tar_member"] for s in samples])

    print(f"✅ Balanced sample saved to {csv_out}, {excel_out}, and {jsonl_out}")


if __name__ == "__main__":
    main()
