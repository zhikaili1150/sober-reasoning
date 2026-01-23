LOCAL_DIR=/local/scratch/zli2255/workspace/sober-reasoning
OUTPUT_DIR=result/icml/math_test
PARTITION=h100
VENV=/local/scratch/zli2255/anaconda3/etc/profile.d/conda.sh
mkdir -p $OUTPUT_DIR/logs

source $VENV
conda activate sob
cd $LOCAL_DIR

MODEL=Qwen/Qwen3-4B-Instruct-2507
TOP_P=0.9
TEMP=0.8
MAX_MODEL_LENGTH=32768
MAX_TOKENS=32768

# =========================
# Unified tasks (no seed)
# =========================
TASKS="aime24_,amc23_"
# TASKS="aime24,amc23,math_500,minerva,olympiadbench"

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