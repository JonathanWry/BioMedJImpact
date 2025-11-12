## Run `extract_archive` Locally

1. **Create/activate environment**
   ```bash
   conda env create -f environment.yml
   conda activate bmj-pipeline
   ```

2. **Edit a working config**
   ```bash
   cp configs/extract.yaml configs/extract.local.yaml
   ```
   Update archive paths, output locations, and model store paths for your workstation (network paths must exist).

3. **Kick off the run**
   ```bash
   PYTHONPATH=src python -m bmj_pipeline.extract_archive \
     --config configs/extract.local.yaml \
     --flush-every 250
   ```
   - The script resumes from any prior `output` JSONL and `processed_file`.
   - Checkpoints are flushed every `flush_every` articles (default 500).

4. **Inspect outputs**
   - `journal_tag_stats_XX.jsonl`: streaming stats (see `data_schema/jsonl_stats.md`).
   - `processed_XX.txt`: archive member names that have been consumed (safe to delete to force reprocessing).
