#!/bin/bash

LOCAL_DIR=/local/scratch/zli2255/workspace/sober-reasoning
OUTPUT_DIR=result/val_v1v2
PARTITION=h100
VENV=/local/scratch/zli2255/anaconda3/etc/profile.d/conda.sh
mkdir -p $OUTPUT_DIR/logs

MODELS=(
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/reward_expert_v0/acc_expert
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/reward_expert_v0/cos_expert
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/reward_expert_v0/fmt_expert

    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_accfmt_v0/fmt0.1acc0.9
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_accfmt_v0/fmt0.2acc0.8
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_accfmt_v0/fmt0.3acc0.7
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_accfmt_v0/fmt0.4acc0.6
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_accfmt_v0/fmt0.5acc0.5
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_accfmt_v0/fmt0.6acc0.4
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_accfmt_v0/fmt0.7acc0.3
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_accfmt_v0/fmt0.8acc0.2
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_accfmt_v0/fmt0.9acc0.1

    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_cosfmt_v0/fmt0.1cos0.9
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_cosfmt_v0/fmt0.2cos0.8
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_cosfmt_v0/fmt0.3cos0.7
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_cosfmt_v0/fmt0.4cos0.6
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_cosfmt_v0/fmt0.5cos0.5
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_cosfmt_v0/fmt0.6cos0.4
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_cosfmt_v0/fmt0.7cos0.3
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_cosfmt_v0/fmt0.8cos0.2
    /local/scratch/zli2255/workspace/open-r1/experiments/exp_grpo_fft/ckpt/merged_policy_cosfmt_v0/fmt0.9cos0.1

    # SpiceRL/DRA-GRPO
    # SpiceRL/DRA-DR.GRPO
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    # knoveleng/Open-RS3
    # knoveleng/Open-RS2
    # knoveleng/Open-RS1
    # agentica-org/DeepScaleR-1.5B-Preview
    # simplescaling/s1.1-7B
    # open-thoughts/OpenThinker-7B
    # Intelligent-Internet/II-Thought-1.5B-Preview
    # sail/Qwen2.5-Math-1.5B-Oat-Zero
    # simplescaling/s1.1-32B
    # deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
    # open-thoughts/OpenThinker-32B
    # GAIR/LIMO
    # bespokelabs/Bespoke-Stratos-32B
)

TOP_PS=(
    0.9
)


TEMPS=(
    0.8
)

MAX_MODEL_LENGTHS=(
    32768
)

MAX_TOKENS_LIST=(
    # 4096
    # 8192
    # 16384
    32768
)

for MAX_MODEL_LENGTH in "${MAX_MODEL_LENGTHS[@]}"; do
for MAX_TOKENS in "${MAX_TOKENS_LIST[@]}"; do
for MODEL in "${MODELS[@]}"; do
for TOP_P in "${TOP_PS[@]}"; do
for TEMP in "${TEMPS[@]}"; do
echo "Submitting $MODEL job for temperature $TEMP, top_p $TOP_P, MAX_MODEL_LENGTH $MAX_MODEL_LENGTH, MAX_TOKENS $MAX_TOKENS"
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=eval-$MODEL-$SEED-$TEMP-$TOP_P-$MAX_MODEL_LENGTH-$MAX_TOKENS
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=$OUTPUT_DIR/logs/%j.out
#SBATCH --error=$OUTPUT_DIR/logs/%j.err
#SBATCH --partition=$PARTITION

source $VENV
conda activate sober
cd $LOCAL_DIR

set -x

SEEDS_2=(
    0
    1
    2
)

TASKS_2=(
    "custom|math_validation_v1|0|0"
    "custom|math_validation_v2|0|0"
)

for SEED in "\${SEEDS_2[@]}"; do
for TASK in "\${TASKS_2[@]}"; do
    python main.py \
        --model $MODEL \
        --task \$TASK \
        --temperature $TEMP \
        --top_p $TOP_P \
        --seed \$SEED \
        --output_dir $OUTPUT_DIR \
        --max_new_tokens $MAX_TOKENS \
        --max_model_length $MAX_MODEL_LENGTH \
        --custom_tasks_directory lighteval_tasks.py \
        --use_chat_template
done
done

EOT

done
done
done
done
done
