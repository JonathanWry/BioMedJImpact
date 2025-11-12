"""
CLI for full-archive extraction and streaming stats generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm
from vllm import LLM

from . import __version__
from .io_utils import append_processed, ensure_parent, load_processed_set, read_yaml
from .llm_utils import (
    stage_keywords_categories,
    stage_keywords_check,
    stage_yes_no,
)
from .parsing import iter_valid_articles, load_journal_tables
from .stats import StatsStore


def _update_config(cfg: dict, overrides: argparse.Namespace) -> dict:
    if overrides.archive:
        cfg["archive"] = str(overrides.archive)
    if overrides.output:
        cfg["output"] = str(overrides.output)
    if overrides.processed_file:
        cfg["processed_file"] = str(overrides.processed_file)
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream PMC archive and build AI/ML journal stats")
    parser.add_argument("--config", type=Path, required=True, help="YAML config path (see configs/extract.yaml)")
    parser.add_argument("--archive", type=Path, help="Override archive path from config")
    parser.add_argument("--output", type=Path, help="Override JSONL stats output path")
    parser.add_argument("--processed-file", type=Path, dest="processed_file", help="Override processed checkpoint file")
    parser.add_argument("--flush-every", type=int, help="Override flush interval")
    parser.add_argument("--version", action="version", version=f"bmj-pipeline {__version__}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    config = _update_config(config, args)
    if args.flush_every:
        config["flush_every"] = args.flush_every

    archive = Path(config["archive"]).expanduser()
    output = Path(config["output"]).expanduser()
    processed_path = Path(config["processed_file"]).expanduser()
    journal1 = Path(config["journal_csvs"]["dataset1"]).expanduser()
    journal2 = Path(config["journal_csvs"]["dataset2"]).expanduser()

    ensure_parent(output)
    ensure_parent(processed_path)

    journal_df, valid_issns = load_journal_tables(journal1, journal2)
    store = StatsStore()
    store.load(output)
    processed = load_processed_set(processed_path)

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

    flush_every = config.get("flush_every", 500)
    year_range = config.get("year_range", {"min": 2016, "max": 2023})
    year_min = year_range.get("min", 2016)
    year_max = year_range.get("max", 2023)

    new_processed: list[str] = []
    processed_since_flush = 0

    iterator = iter_valid_articles(
        tar_path=archive,
        journal_df=journal_df,
        valid_issns=valid_issns,
        processed=processed,
        year_min=year_min,
        year_max=year_max,
    )

    for article in tqdm(iterator, desc="Streaming archive", unit="xml"):
        store.increment_total(article.issn_eissn, article.year)
        new_processed.append(article.tar_member)
        processed.add(article.tar_member)
        processed_since_flush += 1

        # Stage 1 reuses the legacy behavior of sending the abstract twice to mimic the original prompt context.
        abstract_for_stage1 = f"{article.abstract}\n{article.abstract}"
        if not stage_yes_no(llm, abstract_for_stage1):
            pass
        else:
            keywords, categories = stage_keywords_categories(llm, article.abstract)
            if not keywords or not categories:
                continue
            if not stage_keywords_check(llm, keywords):
                continue
            store.add_positive_article(
                article.issn_eissn,
                article.year,
                article.title,
                keywords,
                categories,
                article.meta_tags,
            )

        if processed_since_flush >= flush_every:
            store.save(output)
            append_processed(processed_path, new_processed)
            new_processed.clear()
            processed_since_flush = 0

    if processed_since_flush:
        store.save(output)
        append_processed(processed_path, new_processed)

    print(f"✅ Tag statistics written to {output.resolve()}")


if __name__ == "__main__":
    main()
