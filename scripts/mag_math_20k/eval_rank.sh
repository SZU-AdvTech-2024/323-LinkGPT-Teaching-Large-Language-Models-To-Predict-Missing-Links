#!/bin/bash

DATASET_NAME=mag_math_20k
LINKGPT_DATA_PATH=/home/wlhuang/llm/LinkGPT/data # 你可以更改为其他任何存储数据的路径
PROJECT_PATH=/home/wlhuang/llm/LinkGPT # 确保你已设置正确的项目路径
WANDB_KEY=d7f4fdb74ad61dff6f2dc5a6522b521c0a28e4ac # 替换为你自己的wandb密钥
NUM_OF_EXAMPLES=4 # you can change this to 0, 2, 4
LINKGPT_MODEL_NAME=linkgpt-llama2-7b-cgtp\

# 设置环境变量
export PYTHONPATH=/home/wlhuang/llm/LinkGPT:$PYTHONPATH
echo "PYTHONPATH is set to: $PYTHONPATH"
# 打印路径信息
echo "Project Path: ${PROJECT_PATH}"
echo "Data Path: ${LINKGPT_DATA_PATH}"
echo "WANDB_KEY: ${WANDB_KEY}"

python ${PROJECT_PATH}/linkgpt/eval/eval_yn.py \
	--model_name_or_path /home/wlhuang/llm/LinkGPT/mate-llama/Llama-2-7b-hf \
	--text_embedding_method cgtp \
	--text_embedding_folder_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ \
	--dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
	--eval_dataset_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/eval_yn_dataset_${NUM_OF_EXAMPLES}_examples.pkl \
	--ppr_data_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ppr_data.pt \
	--dataset_name ${DATASET_NAME} \
	--ft_model_path ${LINKGPT_DATA_PATH}/models/${DATASET_NAME}/${LINKGPT_MODEL_NAME} \
	--stage 2 \
	--device cuda \
	--output_path ${LINKGPT_DATA_PATH}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/eval_yn_dataset_${NUM_OF_EXAMPLES}_examples_output.json \
	--max_hop 0 \
	--fp16 \
	--report_to wandb \
	--wandb_key ${WANDB_KEY} \
	--wandb_project_name ${DATASET_NAME}_eval \
	--wandb_run_name ${LINKGPT_MODEL_NAME}-${NUM_OF_EXAMPLES}-examples \
    --verbose \
    --inference_output_path ${LINKGPT_DATA_PATH}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/results.json
