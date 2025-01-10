## 数据准备

从这里下载数据[here](https://drive.google.com/file/d/1-_57MT-Mtp_oYnqSc0Kos7BpDBAyPuy5/view?usp=drive_link) 并将其解压到`LinkGPT/data`. 在运行脚本之前，你需要正确替换在所有的脚本中的 `LINKGPT_DATA_PATH` 变量。 数据的结构需要和下面的文件结构保持一致。

```bash
.
└── datasets
    ├── amazon_clothing_20k
    │   ├── dataset_for_lm.pkl
    │   ├── eval_yn_dataset_0_examples.pkl
    │   ├── eval_yn_dataset_2_examples.pkl
    │   ├── eval_yn_dataset_4_examples.pkl
    │   ├── eval_yn_dataset_large_candidate_set.pkl
    │   ├── ft_np_dataset.pkl
    │   ├── ft_yn_dataset.pkl
    │   ├── ppr_data.pt
    │   └── text_emb_cgtp.pt
    ├── amazon_sports_20k
    │   └── ... (same as above)
    ├── mag_geology_20k
    │   └── ...
    └── mag_math_20k
        └── ...
```

## 训练

使用如下的命令训练模型。 模型的检查点会被保存到 `LinkGPT/data/models`.

```bash
bash scripts/{dataset_name}/train_linkgpt.sh
```

你可以从这个链接中获取[微调模型 ](https://drive.google.com/file/d/17h3ToYyZFp9dcQ9FJjLL6KT-KvrN1BpH/view?usp=sharing
) 并将其解压到 `LinkGPT/data`.  数据的结构需要和下面的文件结构保持一致。

```bash
└── models
    ├── amazon_clothing_20k
    │   └── linkgpt-llama2-7b-cgtp
    │       ├── stage1
    │       │   ├── linkgpt_special_token_emb.pt
    │       │   ├── lora_model
    │       │   │   ├── adapter_config.json
    │       │   │   ├── adapter_model.safetensors
    │       │   │   └── README.md
    │       │   ├── node_alignment_proj.pt
    │       │   ├── pairwise_alignment_proj.pt
    │       │   └── pairwise_encoder.pt
    │       └── stage2
    │           └── ... (same as stage1)
    ├── amazon_sports_20k
    │   └── ... (same as above)
    ├── mag_geology_20k
    │   └── ...
    └── mag_math_20k
        └── ...
```

## 评估

使用如下的命令评估模型。

```bash
bash scripts/{dataset_name}/eval_rank.sh
```
