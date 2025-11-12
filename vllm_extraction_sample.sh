


# --- Config ---
SCRIPT="code/vllm_keyword_sample.py"
OA_SECTION="11"

# Balanced sampling targets
N_POS=75
N_NEG=75

CSV_OUT="/users/rwan388/result/sample_${N_POS}_${N_NEG}_table_${OA_SECTION}.csv"
EXCEL_OUT="/users/rwan388/result/sample_${N_POS}_${N_NEG}_labels_${OA_SECTION}.xlsx"
JSONL_OUT="/users/rwan388/journal_result/sample_${N_POS}_${N_NEG}_logs_${OA_SECTION}.jsonl"

JOURNAL1="/scratch/rwan388/data/Dataset1_final.csv"
JOURNAL2="/scratch/rwan388/data/Dataset2_final.csv"
GPU_UTIL=0.9

# --- Conda env (non-interactive) ---
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate gemma

# --- Run (full 3-stage pipeline; do NOT pass --form_only) ---
python "$SCRIPT" \
  --OA_SECTION "$OA_SECTION" \
  --n_pos "$N_POS" \
  --n_neg "$N_NEG" \
  --csv_out "$CSV_OUT" \
  --excel_out "$EXCEL_OUT" \
  --jsonl_out "$JSONL_OUT" \
  --journal1 "$JOURNAL1" \
  --journal2 "$JOURNAL2" \
  --gpu_memory_utilization "$GPU_UTIL"