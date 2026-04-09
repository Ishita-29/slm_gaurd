
set -e
cd /data/ishita_workspace/SLM-GAURD/slmguard/src

GUARD_GPU=cuda:1
LLM_GPU=cuda:2
SAMPLES=20
BENIGN=50
LOG_DIR=../logs/middleware

mkdir -p $LOG_DIR

MODELS=(
  "qwen25_1b"
  "qwen25_7b"
  "phi4mini"
  "deepseek_r1"
  "llama31_8b"
  "gemma2_2b"
  "gemma3_4b"
  "mistral_7b"
)

for model in "${MODELS[@]}"; do
  echo ""
  echo "=================================================="
  echo " Evaluating target: $model"
  echo "=================================================="
  python middleware_eval.py \
    --target $model \
    --samples $SAMPLES \
    --benign $BENIGN \
    --guard_gpu $GUARD_GPU \
    --llm_gpu $LLM_GPU \
    2>&1 | tee $LOG_DIR/${model}.log
  echo "Done: $model"
done

echo ""
echo "All evaluations complete."
echo "Results in: /data/ishita_workspace/SLM-GAURD/slmguard/results/"
