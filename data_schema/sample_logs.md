## `sample_XX_XX_logs_YY.jsonl`

Emitted by `bmj_pipeline.sample_balanced` to preserve the provenance of each sampled abstract.

| Field | Type | Description |
| --- | --- | --- |
| `label` | string | `positive` if all three stages passed, `negative` if rejected in stage 1. |
| `title` | string | Article title extracted from the XML. |
| `issn_eissn` | string | Canonical journal key (`Dataset1_final` × `Dataset2_final`). |
| `year` | int | Publication year. |
| `meta_tags` | string[] | Unique metadata tags gathered from `<article-categories>` / `<kwd-group>`. |
| `keywords` | string[] | FORM-A keywords (only present for positive samples). |
| `categories` | string[] | FORM-B categories (only present for positive samples). |
| `tar_member` | string | Relative path inside the PMC `.tar(.gz)` archive (used for resume). |
| `abstract` | string | Normalized abstract text inspected by the pipeline. |
| `stage` | string | `three_stage_pass` or `stage1_reject` to make sampling decisions auditable. |

The JSONL log is the authoritative record for human-labeling spreadsheets and can be replayed to regenerate CSV/Excel tables if needed.
