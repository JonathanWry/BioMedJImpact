set -euo pipefail

module purge
conda init bash >/dev/null 2>&1
source ~/.bashrc
conda activate bmj-pipeline

export PYTHONPATH=/users/rwan388/code/BioMedJImpact-LLM/src:${PYTHONPATH:-}

python -m bmj_pipeline.sample_balanced \
  --config /users/rwan388/code/BioMedJImpact-LLM/configs/sample.yaml
