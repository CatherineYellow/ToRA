#!/bin/bash
set -ex

# 模型检查点列表
# MODEL_STEPS=("global_step_300" "global_step_400" "global_step_500" "global_step_540" )
MODEL_STEPS=("xDAN-R2-Thinking-0401")
# max_tokens_per_call 选择
MAX_TOKENS=(8192 16384)

# 数据集配置
DATA_NAME="math500"
SPLIT="test"
PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=-1

# 输出 CSV 文件
# OUTPUT_FILE="aime2024_test.csv"
OUTPUT_FILE=${DATA_NAME}_${SPLIT}.csv
#echo "model, max_tokens, mean_score" > ${OUTPUT_FILE}  # 先写入表头
if [ ! -f ${OUTPUT_FILE} ]; then
    echo "model, max_tokens, mean_score" > ${OUTPUT_FILE}  # 先写入表头
fi
# 运行实验
for MODEL_STEP in "${MODEL_STEPS[@]}"; do
    #MODEL_NAME_OR_PATH="/data/vayu/train/models/ckpts/ToRL/rl.grpo_qwen.base_7b_torl_data_numcall1/${MODEL_STEP}"
    MODEL_NAME_OR_PATH="/data/vayu/train/models/${MODEL_STEP}"
    for MAX_TOKENS_VALUE in "${MAX_TOKENS[@]}"; do
        echo "Running model: ${MODEL_STEP}, max_tokens: ${MAX_TOKENS_VALUE}"
        
        # 运行推理
        OUTPUT=$(CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOKENIZERS_PARALLELISM=false \
        python -um infer.inference \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --data_name ${DATA_NAME} \
        --split ${SPLIT} \
        --max_tokens_per_call ${MAX_TOKENS_VALUE} \
        --prompt_type ${PROMPT_TYPE} \
        --use_train_prompt_format \
        --num_test_sample ${NUM_TEST_SAMPLE} \
        --seed 0 \
        --temperature 0 \
        --n_sampling 1 \
        --top_p 1 \
        --start 0 \
        --end -1)

        # 提取 Mean Score
        MEAN_SCORE=$(echo "$OUTPUT" | grep "Mean score" | awk -F '[][]' '{print $2}')
        
        # 保存结果到 CSV
        echo "${MODEL_STEP}, ${MAX_TOKENS_VALUE}, ${MEAN_SCORE}" >> ${OUTPUT_FILE}
    done
done

echo "All experiments completed! Results saved in ${OUTPUT_FILE}"

# set -ex
# MODEL_NAME_OR_PATH="/data/vayu/train/models/ckpts/ToRL/rl.grpo_qwen.base_7b_torl_data_numcall1/global_step_540"
# # MODEL_NAME_OR_PATH="RUC-AIBOX/STILL-3-TOOL-32B"
# # MODEL_NAME_OR_PATH="/data/vayu/train/models/xDAN-L1-Qwen25-7B-Instruct"
# # MODEL_NAME_OR_PATH="RUC-AIBOX/STILL-3-1.5B-preview"
# # MODEL_NAME_OR_PATH="llm-agents/tora-70b-v1.0"

# # DATA_LIST = ['math', 'gsm8k', 'gsm-hard', 'svamp', 'tabmwp', 'asdiv', 'mawps']

# DATA_NAME="AIME_2024"
# # DATA_NAME="gsm8k"

# SPLIT="test"
# PROMPT_TYPE="tora"
# NUM_TEST_SAMPLE=-1


# CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
# python -um infer.inference \
# --model_name_or_path ${MODEL_NAME_OR_PATH} \
# --data_name ${DATA_NAME} \
# --split ${SPLIT} \
# --max_tokens_per_call 1024 \
# --prompt_type ${PROMPT_TYPE} \
# --use_train_prompt_format \
# --num_test_sample ${NUM_TEST_SAMPLE} \
# --seed 0 \
# --temperature 0 \
# --n_sampling 1 \
# --top_p 1 \
# --start 0 \
# --end -1 \
