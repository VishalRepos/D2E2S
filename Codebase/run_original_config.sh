#!/bin/bash
# Match old codebase configuration that achieves 75-80% F1
# Key: Use DeBERTa-v2-XXLarge with larger dimensions and lower LR

python ./train.py \
    --dataset 14res \
    --batch_size 16 \
    --lr 5e-6 \
    --epochs 120 \
    --gcn_type adaptive \
    --attention_heads 1 \
    --pretrained_deberta_name microsoft/deberta-v2-xxlarge \
    --emb_dim 1536 \
    --hidden_dim 768 \
    --deberta_feature_dim 1536 \
    --gcn_dim 300 \
    --mem_dim 768 \
    --sampling_processes 4 \
    --lr_warmup 0.2 \
    --weight_decay 0.01 \
    --device cuda
