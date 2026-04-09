
set -e
cd /data/ishita_workspace/SLM-GAURD/slmguard/src

GUARD_GPU=cuda:1
LLM_GPU=cuda:2
LOG_DIR=../logs/middleware_ood

mkdir -p $LOG_DIR

# Representative target models (covers different providers and sizes)
MODELS=(
  "qwen25_1b"
  "qwen25_7b"
  "mistral_7b"
  "gemma3_4b"
)

# OOD sources to test
SOURCES=(
  "jailbreakhub"
  "toxicchat"
  "advbench"
)

for source in "${SOURCES[@]}"; do
  for model in "${MODELS[@]}"; do
    echo ""
    echo "=================================================="
    echo " OOD Source: $source  |  Target: $model"
    echo "=================================================="
    python middleware_eval_ood.py \
      --target $model \
      --source $source \
      --n_attacks 100 \
      --n_benign 50 \
      --guard_gpu $GUARD_GPU \
      --llm_gpu $LLM_GPU \
      2>&1 | tee $LOG_DIR/${source}_${model}.log
    echo "Done: $source x $model"
  done
done

echo ""
echo "All OOD evaluations complete."
echo "Results in: /data/ishita_workspace/SLM-GAURD/slmguard/results/"
