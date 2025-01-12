from typing import List, Optional, Tuple, Union
import pickle
import json
import os
import sys
import argparse
from unittest.mock import patch
import copy
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
import dgl
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, \
                             LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import DataCollatorForLanguageModeling, Trainer, HfArgumentParser
from transformers.trainer_pt_utils import LengthGroupedSampler, RandomSampler
import wandb

import llmtuner
from llmtuner.model.patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model
from llmtuner.hparams.parser import get_train_args
import llmtuner.hparams.parser as llm_tuner_parser
from llmtuner.extras.misc import count_parameters
from llmtuner.model.loader import init_adapter

# 设置项目路径
project_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../..")) # path to LinkGPT
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# 导入自定义模块
from linkgpt.dataset.tag_dataset_for_lm import TAGDatasetForLM, tag_dataset_for_lm_to_dgl_graph
from linkgpt.pairwise_encoding.lpformer_dataset import get_lpformer_dataset
from linkgpt.pairwise_encoding.models.link_transformer import LinkTransformer
from linkgpt.pairwise_encoding.lpformer_model_api import get_lpformer_model
from linkgpt.model.linkgpt_model import LinkGPTForCausalLM, LinkGPTConfig, \
    unfreeze_graph_related_modules, unfreeze_lora_adapter, freeze_all_parameters, \
        save_lora_model, get_model_and_tokenizer, get_tokenizer, load_model_and_tokenizer
from linkgpt.dataset.linkgpt_dataset import LinkGPTDataset, LinkGPTDataCollator
from linkgpt.dataset.yn_dataset import YNDataset, YNDatasetConfig
from linkgpt.dataset.np_dataset import NPDataset, NPDatasetConfig
from linkgpt.dataset.utils import NODE_START_TOKEN, NODE_TOKEN, PAIRWISE_START_TOKEN, \
    PAIRWISE_TOKEN, LINKGPT_SPECIAL_TOKENS
from linkgpt.utils import basics

# 初始化 WandB
# 如果当前环境无法访问外网，可以设置为离线模式
import wandb
wandb.init(mode="offline")
# 暂时禁用梯度缩放
from torch.cuda.amp import GradScaler
import logging

logging.basicConfig(level=logging.DEBUG)
scaler = GradScaler(enabled=False)

# linkgpt/utils/basics.py

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_mrr(rank_ls: List[float]) -> float:
    reciprocal_ranks = [1.0 / rank for rank in rank_ls]
    return np.mean(reciprocal_ranks)

def calculate_hit(rank_ls: List[float], n: int) -> float:
    hits = [1 if rank <= n else 0 for rank in rank_ls]
    return np.mean(hits)

def confusion_matrix_fn(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()


def evaluate_model(model, tokenizer, dataset, device, batch_size=32, verbose=False):
    """
    在给定的数据集上评估模型，计算 MRR、Hit@1、Hit@10 等指标。

    Args:
        model: 训练好的模型。
        tokenizer: 分词器。
        dataset: 要评估的数据集。
        device: 设备（CPU 或 GPU）。
        batch_size: 批次大小。
        verbose: 是否打印详细信息。

    Returns:
        mrr: Mean Reciprocal Rank。
        hit1: Hit@1。
        hit10: Hit@10。
    """
    model.eval()
    rank_list = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in dataloader:
            prompts = batch['prompt']
            labels = batch['label']  # 假设数据集中有 'label' 字段，1 表示存在边，0 表示不存在

            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # 获取最后一个 token 的 logits

            # 假设 'Yes' 和 'No' 是模型输出的两个选项
            yes_id = tokenizer.convert_tokens_to_ids(": Yes")
            no_id = tokenizer.convert_tokens_to_ids(": No")
            yes_logits = logits[:, yes_id]
            no_logits = logits[:, no_id]
            probs = F.softmax(logits, dim=-1)
            yes_probs = probs[:, yes_id]
            no_probs = probs[:, no_id]
            predictions = torch.argmax(probs[:, [yes_id, no_id]], dim=-1)  # 0 表示 'Yes', 1 表示 'No'

            # 计算排名
            for i in range(len(labels)):
                rank = 1 if predictions[i].item() == labels[i].item() else 2
                rank_list.append(rank)
                y_true.append(labels[i].item())
                y_pred.append(predictions[i].item())

                if verbose:
                    print("--------------------------------------------------")
                    print(f"Prompt: {prompts[i]}")
                    print(f"Prediction: {'Yes' if predictions[i].item() == 0 else 'No'}")
                    print(f"Yes Logit: {yes_logits[i].item():.4f}, No Logit: {no_logits[i].item():.4f}")
                    print(f"Yes Probability: {yes_probs[i].item():.4f}, No Probability: {no_probs[i].item():.4f}")

    # 计算指标
    mrr = basics.calculate_mrr(rank_ls=rank_list)
    hit1 = basics.calculate_hit(rank_ls=rank_list, n=1)
    hit10 = basics.calculate_hit(rank_ls=rank_list, n=10)

    # 打印混淆矩阵
    cm = basics.confusion_matrix(y_true, y_pred)
    basics.plot_confusion_matrix(cm, class_names=["Yes", "No"])

    return mrr, hit1, hit10

def main():
    basics.set_seeds(42)

    parser = argparse.ArgumentParser()

    # data path and project description
    parser.add_argument('--model_name_or_path', required=True)
    parser.add_argument('--text_embedding_method', required=True)
    parser.add_argument('--text_embedding_folder_path', required=True)
    parser.add_argument('--max_hop', default=0, type=int)
    parser.add_argument('--dataset_for_lm_path', required=True)
    parser.add_argument('--ppr_data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dataset_name', required=True)
    
    parser.add_argument('--lp_dataset_path', required=True, help='Link prediction (Yes/No) dataset path')
    parser.add_argument('--lp_ablate_pairwise_encoding', action='store_true')
    parser.add_argument('--lp_ablate_node_encoding', action='store_true')
    parser.add_argument('--lp_learn_text', action='store_true')
    parser.add_argument('--lp_learn_all', action='store_true')
    
    parser.add_argument('--np_dataset_path', default=None, help='Neighbor prediction dataset path (optional)')
    parser.add_argument('--np_ablate_node_encoding', action='store_true')
    parser.add_argument('--np_learn_src_text', action='store_true')
    parser.add_argument('--np_learn_all', action='store_true')
    
    parser.add_argument('--device_setting', default=None, choices=['cpu', 'cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])

    # wandb
    parser.add_argument('--wandb_key', default=None, type=str)
    parser.add_argument('--project_name', default='LinkGPT-ft', type=str)
    parser.add_argument('--run_name', default='untitled_run', type=str)

    # Huggingface args
    parser.add_argument('--finetuning_type', default='lora')
    parser.add_argument('--per_device_train_batch_size', default=4, type=int)
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int)
    parser.add_argument('--lr_scheduler_type', default='cosine')
    parser.add_argument('--logging_steps', default=10, type=int)
    parser.add_argument('--save_steps', default=10000, type=int)
    parser.add_argument('--learning_rate', default=5e-5, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--warmup_ratio', default=0, type=float)
    parser.add_argument('--max_grad_norm', default=1, type=float)

    parser.add_argument('--num_train_epochs_stage1', default=1, type=int)
    parser.add_argument('--num_train_epochs_stage2', default=1, type=int)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--report_to', default=None)
    parser.add_argument('--dataloader_num_workers', default=4, type=int)
    parser.add_argument('--dataloader_prefetch_factor', default=8, type=int)

    parser.add_argument('--lora_target', default='q_proj,v_proj')
    parser.add_argument('--lora_alpha', default=16, type=float)
    parser.add_argument('--lora_rank', default=8, type=int)
    parser.add_argument('--lora_dropout', default=0.0, type=float)


    # others
    parser.add_argument('--freeze_graph_related_modules_in_stage2', action='store_true')
    parser.add_argument('--stage1_task', default='lp,np') # tasks to train in stage 1. lp means link prediction, np means neighbor prediction.
    parser.add_argument('--stage2_task', default='lp,np') # tasks to train in stage 2.
    parser.add_argument('--node_proj_num_layers', default=1, type=int) # number of layers for the node projection

    # 新增评估参数
    parser.add_argument('--validation_dataset_path', required=True, help='Validation dataset path for evaluation')
    parser.add_argument('--eval_batch_size', default=32, type=int, help='Batch size for evaluation')
    parser.add_argument('--save_best_model', action='store_true', help='Whether to save the best model based on validation metrics')
    parser.add_argument('--early_stopping_patience', default=3, type=int, help='Early stopping patience for validation metrics')

    args = parser.parse_args()
    print(vars(args))
        
    args.fp16 = False

    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化 WandB
    if args.wandb_key:
        if args.wandb_key == 'None':
            args.report_to = None
        else:
            wandb.login(key=args.wandb_key)
            os.environ["WANDB_PROJECT"] = args.project_name
        
    if args.device_setting is None:
        device = basics.get_device()
        print(f"No device setting is provided. Using {device}", flush=True)
    else:
        device = args.device_setting

    # 加载文本嵌入
    text_emb_list = []
    for i in range(args.max_hop + 1):
        if i == 0:
            text_emb_path = os.path.join(args.text_embedding_folder_path, f'text_emb_{args.text_embedding_method}.pt')
        else:
            text_emb_path = os.path.join(args.text_embedding_folder_path, f'text_emb_{args.text_embedding_method}_{i}hop.pt')
        text_emb = torch.load(text_emb_path, map_location=device)
        text_emb_list.append(text_emb)
    
    # 加载语言模型数据集
    dataset_for_lm = basics.load_pickle(args.dataset_for_lm_path)
    ppr_data = torch.load(args.ppr_data_path).to(device)
    
    # 加载链接预测（LP）数据集
    lp_dataset = basics.load_pickle(args.lp_dataset_path)
    lp_dataset.config.ablate_pairwise_encoding = args.lp_ablate_pairwise_encoding
    lp_dataset.config.ablate_node_encoding = args.lp_ablate_node_encoding
    lp_dataset.config.learn_text = args.lp_learn_text
    lp_dataset.config.learn_all = args.lp_learn_all
    lp_dataset.config.node_encoding_max_hop = args.max_hop
    
    # 加载邻居预测（NP）数据集（可选）
    if args.np_dataset_path is not None:
        np_dataset = basics.load_pickle(args.np_dataset_path)
        np_dataset.config.ablate_node_encoding = args.np_ablate_node_encoding
        np_dataset.config.np_learn_all = args.np_learn_all
        np_dataset.config.learn_src_text = args.np_learn_src_text
        np_dataset.config.node_encoding_max_hop = args.max_hop

    # 构建 DGL 图并赋值节点特征，添加一致性检查
    dgl_graph = tag_dataset_for_lm_to_dgl_graph(dataset_for_lm, include_valid=True).to(device)
    
    print(f"Graph has {dgl_graph.number_of_nodes()} nodes.")
    print(f"Text embeddings have {text_emb_list[0].shape[0]} features.")
    if dgl_graph.number_of_nodes() != text_emb_list[0].shape[0]:
        raise ValueError(f"Number of text embeddings ({text_emb_list[0].shape[0]}) does not match number of graph nodes ({dgl_graph.number_of_nodes()}).")
    dgl_graph.ndata['feat'] = text_emb_list[0]

    # 加载配对编码器（Pairwise Encoder）
    lpformer_dataset = get_lpformer_dataset(args.dataset_name, dataset_for_lm.edge_split, dgl_graph, ppr_data, device)
    lpformer_model = get_lpformer_model(lpformer_dataset, device).to(device) # randomly initialized, not pre-trained
    
    # 加载模型和分词器
    hf_parser = HfArgumentParser(llm_tuner_parser._TRAIN_ARGS)
    hf_args_dict = {
        'do_train': True,
        'stage': 'pt',
        'lora_target': 'q_proj,v_proj',
        'overwrite_output_dir': True,
        'resize_vocab': True,
        'model_name_or_path': args.model_name_or_path,
        'output_dir': args.output_dir,
        'dataset': args.dataset_name,
        'finetuning_type': args.finetuning_type,
        'per_device_train_batch_size': args.per_device_train_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'lr_scheduler_type': args.lr_scheduler_type,
        'logging_steps': args.logging_steps,
        'save_steps': args.save_steps,
        'learning_rate': args.learning_rate,
        'num_train_epochs': args.num_train_epochs_stage1,
        'fp16': args.fp16,
        'lora_rank': args.lora_rank,
        'lora_target': args.lora_target,
        'lora_dropout': args.lora_dropout,
        'lora_alpha': args.lora_alpha,
        'report_to': args.report_to,
        'dataloader_prefetch_factor': args.dataloader_prefetch_factor,
        'dataloader_num_workers': args.dataloader_num_workers,
    }
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(hf_args_dict)
     # 确保 training_args.fp16 为 False
    training_args.fp16 = False
    model, tokenizer = get_model_and_tokenizer(
        model_args,
        finetuning_args,
        text_emb_list,
        lpformer_model,
        is_trainable=True,
        device=device,
        apply_lora=True,
        node_proj_num_layers=args.node_proj_num_layers
    )
     # 转换模型为 float32
    model = model.float()

    # 检查所有参数的数据类型
    for name, param in model.named_parameters():
        if param.dtype != torch.float32:
            print(f"Parameter {name} is not float32: {param.dtype}")
    linkgpt_data_collator = LinkGPTDataCollator(tokenizer)
    
    
    # stage 1
    if not args.lp_ablate_pairwise_encoding or not args.lp_ablate_node_encoding:
        # 需要训练与图相关的部分
        print('Stage 1: Only train the graph-related parts', flush=True)
        freeze_all_parameters(model)
        unfreeze_graph_related_modules(model)
        trainable_param, total_param = count_parameters(model)
        print(f'Num of trainable params: {trainable_param}', flush=True)
        print(f'Total num of params: {total_param}', flush=True)
        print(f'ratio: {trainable_param / total_param * 100: .3}%', flush=True)
        
        print(f'Training tasks: {args.stage1_task}')
        dataset_list = []
        for task in args.stage1_task.split(','):
            if task == 'np':
                dataset_list.append(copy.deepcopy(np_dataset))
            elif task == 'lp':
                dataset_list.append(copy.deepcopy(lp_dataset))
            else:
                raise ValueError('Invalid task name. Can only be np or lp.')
        training_args.num_train_epochs = args.num_train_epochs_stage1
        
        # 不混合数据，以便模型可以很好地学习每个任务
        for dataset in dataset_list:
            stage1_trainer = Trainer(
                model=model,
                args=training_args,
                data_collator=linkgpt_data_collator,
                train_dataset=dataset,
            )
            train_result = stage1_trainer.train()
            if args.save_best_model:
                # 假设您有一种方法来确定最佳模型
                save_lora_model(stage1_trainer.model, os.path.join(args.output_dir, 'stage1_best'))
        save_lora_model(stage1_trainer.model, os.path.join(args.output_dir, 'stage1'))
    else:
        # 同时消融配对编码和节点编码，不需要训练与图相关的部分
        print('Stage 1 skipped, as no graph-related part is used.')

    
    # stage 2
    print('Stage 2: Tune both the LLM and the graph-related parts')
    freeze_all_parameters(model)
    if not args.freeze_graph_related_modules_in_stage2:
        unfreeze_graph_related_modules(model)
    unfreeze_lora_adapter(model)
    trainable_param, total_param = count_parameters(model)
    print(f'Num of trainable params: {trainable_param}')
    print(f'Total num of params: {total_param}')
    print(f'ratio: {trainable_param / total_param * 100: .3}%')
    
    print(f'Training tasks: {args.stage2_task}')
    if args.stage2_task == 'none':
        return
    dataset_list = []
    for task in args.stage2_task.split(','):
        if task == 'np':
            dataset_list.append(copy.deepcopy(np_dataset))
        elif task == 'lp':
            dataset_list.append(copy.deepcopy(lp_dataset))
        else:
            raise ValueError('Invalid task name. Can only be np or lp.')
    linkgpt_dataset = LinkGPTDataset(dataset_list)
    training_args.num_train_epochs = args.num_train_epochs_stage2
        
    # 混合数据以便模型可以同时学习多个任务
    stage2_trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=linkgpt_data_collator,
        train_dataset=linkgpt_dataset,
    )
    train_result = stage2_trainer.train()
    if args.save_best_model:
        # 假设您有一种方法来确定最佳模型
        save_lora_model(stage2_trainer.model, os.path.join(args.output_dir, 'stage2_best'))
    save_lora_model(stage2_trainer.model, os.path.join(args.output_dir, 'stage2'))

    # 评估阶段
    print("Starting evaluation on the validation dataset...")
    validation_dataset = basics.load_pickle(args.validation_dataset_path)
    validation_dataset.config.ablate_pairwise_encoding = args.lp_ablate_pairwise_encoding
    validation_dataset.config.ablate_node_encoding = args.lp_ablate_node_encoding
    validation_dataset.config.learn_text = args.lp_learn_text
    validation_dataset.config.learn_all = args.lp_learn_all
    validation_dataset.config.node_encoding_max_hop = args.max_hop

    mrr, hit1, hit10 = evaluate_model(model, tokenizer, validation_dataset, device, batch_size=args.eval_batch_size, verbose=True)
    print(f"Validation Metrics:\nMRR: {mrr:.4f}\nHit@1: {hit1:.4f}\nHit@10: {hit10:.4f}")

    # 将评估结果记录到 WandB（如果启用）
    if args.report_to == 'wandb':
        wandb.log({
            'Validation MRR': mrr,
            'Validation Hit@1': hit1,
            'Validation Hit@10': hit10,
        })

if __name__ == '__main__':
    main()
