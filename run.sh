LOCAL_DIR=/root/sober-reasoning
OUTPUT_DIR=/root/autodl-fs/result/icml/math
PARTITION=h100
VENV=/root/miniconda3/etc/profile.d/conda.sh
mkdir -p $OUTPUT_DIR/logs

source $VENV
conda activate sober
cd $LOCAL_DIR

MODEL=Zachary1150/math_acc_4B
TOP_P=0.9
TEMP=0.8
MAX_MODEL_LENGTH=32768
MAX_TOKENS=32768

# =========================
# Unified tasks (no seed)
# =========================
TASKS="aime24_,amc23,math_500_,minerva,olympiadbench"

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