## Run `sample_balanced` Locally

Use this workflow to spin up a balanced (AI vs non-AI) abstract set without Slurm.

1. **Environment**
   ```bash
   conda activate bmj-pipeline
   ```

2. **Config**
   ```bash
   cp configs/sample.yaml configs/sample.local.yaml
   ```
   Edit archive, output, and processed paths. Adjust `balanced_targets` for the number of positive/negative samples you need.

3. **Run**
   ```bash
   PYTHONPATH=src python -m bmj_pipeline.sample_balanced \
     --config configs/sample.local.yaml
   ```

4. **Outputs**
   - `sample_XX_XX_table.csv`: flat table for spreadsheet workflows.
   - `sample_XX_XX_labels.xlsx`: identical content but formatted for annotators.
   - `sample_XX_XX_logs.jsonl`: machine-readable log (see `data_schema/sample_logs.md`).
   - Optional `sample_processed.txt`: reuse to skip already sampled PMC files on the next run.
