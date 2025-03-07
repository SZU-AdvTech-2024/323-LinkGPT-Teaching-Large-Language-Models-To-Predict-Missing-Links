#!/bin/bash

DATASET_NAME=amazon_clothing_20k
LINKGPT_DATA_PATH=/home/wlhuang/llm/LinkGPT/data # 你可以更改为其他任何存储数据的路径
PROJECT_PATH=/home/wlhuang/llm/LinkGPT # 确保你已设置正确的项目路径
WANDB_KEY=d7f4fdb74ad61dff6f2dc5a6522b521c0a28e4ac # 替换为你自己的wandb密钥
LINKGPT_MODEL_NAME=linkgpt-llama2-7b-cgtp

# evaluate the pipeline without retrieval
python ${PROJECT_PATH}/linkgpt/eval/eval_yn.py \
	--model_name_or_path /home/wlhuang/llm/LinkGPT/mate-llama/Llama-2-7b-hf \
	--text_embedding_method cgtp \
	--text_embedding_folder_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ \
	--dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
	--eval_dataset_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/eval_yn_dataset_large_candidate_set.pkl \
	--ppr_data_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ppr_data.pt \
	--dataset_name ${DATASET_NAME} \
	--ft_model_path ${LINKGPT_DATA_PATH}/models/${DATASET_NAME}/${LINKGPT_MODEL_NAME} \
	--stage 2 \
	--device cuda \
	--output_path ${LINKGPT_DATA_PATH}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/eval_yn_dataset_large_candidate_set_output.json \
	--max_hop 0 \
	--fp16 \
	--report_to wandb \
	--wandb_key ${WANDB_KEY} \
	--wandb_project_name ${DATASET_NAME}_eval \
	--wandb_run_name ${LINKGPT_MODEL_NAME}-large-candidate-set \

# generate neighbor predictions
python ${PROJECT_PATH}/linkgpt/retrieval/generate_neighbor_predictions.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --text_embedding_method cgtp \
    --text_embedding_folder_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME} \
    --dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
    --ppr_data_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/ppr_data.pt \
    --dataset_name ${DATASET_NAME} \
    --output_dir ${DATASET_NAME}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/predicted_neighbors.json \
    --ft_model_path ${LINKGPT_DATA_PATH}/models/${DATASET_NAME}/${LINKGPT_MODEL_NAME} \
    --stage 2 \
    --max_hop 0 \
    --device cuda \
    --fp16 \
    --max_context_neighbors 2 \
    --max_new_tokens 50 \
    --max_num 200 \
    --apply_get_diverse_answers \
    --top_p 0.8 \
    --diversity_penalty 1.2 \
    --num_beam_groups 5 \
    --num_beam_per_group 3

# evaluate the retrieval rerank pipeline
python ${PROJECT_PATH}/linkgpt/retrieval/eval_retrieval_rerank.py \
    --prediction_list_path ${DATASET_NAME}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/predicted_neighbors.json \
    --dataset_for_lm_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/dataset_for_lm.pkl \
    --eval_dataset_path ${LINKGPT_DATA_PATH}/datasets/${DATASET_NAME}/eval_yn_dataset_large_candidate_set.pkl \
    --eval_output_path ${DATASET_NAME}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/eval_yn_dataset_large_candidate_set_output.json \
    --result_saving_path ${DATASET_NAME}/eval_output/${DATASET_NAME}/${LINKGPT_MODEL_NAME}/retrieval_rerank_results.txt \
    --dataset_name ${DATASET_NAME} \
    --num_neg_tgt 1800 \
    --num_to_retrieve 30 \
    --apply_dist_based_grouping \
    --max_dist 2 \
    --beta 0.65