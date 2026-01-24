# =========================
# Paths & env
# =========================
LOCAL_DIR=/root/sober-reasoning
RESULT_ROOT=/root/autodl-fs/result/icml/math
VENV=/root/miniconda3/etc/profile.d/conda.sh
mkdir -p "$RESULT_ROOT"

MODELS=(
  "Zachary1150/math_mix_1.5B"
)

# =========================
# Generation config
# =========================
TOP_P=0.95
TEMP=0.6
MAX_MODEL_LENGTH=32768
MAX_TOKENS=32768

source "$VENV"
conda activate sober
cd "$LOCAL_DIR"

# =========================
# Unified tasks (no seed)
# =========================
TASKS="aime24_,amc23,math_500_,minerva,olympiadbench"

SCRIPT_START=$(date +"%Y-%m-%d %H:%M:%S")
SCRIPT_START_SEC=$SECONDS

echo "=============================================="
echo "Job started at: $SCRIPT_START"
echo "Models:"
for m in "${MODELS[@]}"; do echo "  - $m"; done
echo "Tasks: $TASKS"
echo "=============================================="
echo

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL")
  OUTPUT_DIR="${RESULT_ROOT}/${MODEL_NAME}"
  mkdir -p "$OUTPUT_DIR/logs"

  START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
  START_SEC=$SECONDS

  echo "----------------------------------------------"
  echo "Running model: $MODEL"
  echo "Output dir  : $OUTPUT_DIR"
  echo "Start time  : $START_TIME"
  echo "----------------------------------------------"

  python main.py \
      --model "$MODEL" \
      --task "$TASKS" \
      --temperature "$TEMP" \
      --top_p "$TOP_P" \
      --output_dir "$OUTPUT_DIR" \
      --max_new_tokens "$MAX_TOKENS" \
      --max_model_length "$MAX_MODEL_LENGTH" \
      --custom_tasks_directory lighteval_tasks.py \
      --use_chat_template

  END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
  ELAPSED_SEC=$((SECONDS - START_SEC))

  echo "----------------------------------------------"
  echo "Finished model: $MODEL"
  echo "End time    : $END_TIME"
  echo "Elapsed     : $((ELAPSED_SEC/3600))h $((ELAPSED_SEC%3600/60))m $((ELAPSED_SEC%60))s"
  echo "----------------------------------------------"
  echo
done

SCRIPT_END=$(date +"%Y-%m-%d %H:%M:%S")
TOTAL_SEC=$((SECONDS - SCRIPT_START_SEC))

echo "=============================================="
echo "All jobs finished at: $SCRIPT_END"
echo "Total time: $((TOTAL_SEC/3600))h $((TOTAL_SEC%3600/60))m $((TOTAL_SEC%60))s"
echo "=============================================="
