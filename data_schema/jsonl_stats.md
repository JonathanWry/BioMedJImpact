## `journal_tag_stats_XX.jsonl`

Each line captures one `(ISSN_EISSN, year)` aggregate emitted by `bmj_pipeline.extract_archive`.

| Field | Type | Description |
| --- | --- | --- |
| `ISSN_EISSN` | string | Canonical identifier from Dataset1/Dataset2 join (e.g., `0928-4249_1297-9716`). |
| `year` | int | Publication year pulled from `<pub-date>` (bounded via `year_range`). |
| `Journal_Count` | int | Total number of articles processed for this journal-year (AI + non-AI). |
| `AI_Count` | int | Articles that passed the full 3-stage AI detection pipeline. |
| `Tag_Counts` | object | Map of FORM-A keywords → aggregated counts for this journal-year. |
| `Category_Counts` | object | Map of LIST-A categories → aggregated counts for this journal-year. |
| `Articles` | object | Nested map `title → {Meta_Tags[], Form_A_Keywords[], Form_B_Categories[]}` detailing de-duplicated article level signals. |

Notes:

- Counters are streaming-safe; partial checkpoints overwrite the JSONL file each flush.
- Lists in `Articles` are unique and sorted to simplify downstream diffing.
- To resume, the extractor reloads this JSONL file and merges counters before continuing.
