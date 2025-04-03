set -ex
MODEL_NAME_OR_PATH="/data/vayu/train/models/ckpts/ToRL/rl.grpo_qwen.base_7b_torl_data_numcall1/global_step_160/actor_hf"
# MODEL_NAME_OR_PATH="RUC-AIBOX/STILL-3-TOOL-32B"
# MODEL_NAME_OR_PATH="/data/vayu/train/models/xDAN-L1-Qwen25-7B-Instruct"
# MODEL_NAME_OR_PATH="RUC-AIBOX/STILL-3-1.5B-preview"
# MODEL_NAME_OR_PATH="llm-agents/tora-70b-v1.0"

# DATA_LIST = ['math', 'gsm8k', 'gsm-hard', 'svamp', 'tabmwp', 'asdiv', 'mawps']

DATA_NAME="AIME_2024"
# DATA_NAME="gsm8k"

SPLIT="test"
PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=-1


CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false \
python -um infer.inference \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--data_name ${DATA_NAME} \
--split ${SPLIT} \
--max_tokens_per_call 16000 \
--prompt_type ${PROMPT_TYPE} \
--use_train_prompt_format \
--num_test_sample ${NUM_TEST_SAMPLE} \
--seed 0 \
--temperature 0 \
--n_sampling 1 \
--top_p 1 \
--start 0 \
--end -1 \
