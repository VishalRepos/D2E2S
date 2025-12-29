#!/bin/bash
# Optimized training parameters with lower LR and early stopping
python ./train.py \
    --dataset 15res \
    --batch_size 12 \
    --lr 0.0002 \
    --epochs 50 \
    --gcn_type adaptive \
    --attention_heads 8 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --emb_dim 768 \
    --hidden_dim 384 \
    --deberta_feature_dim 768 \
    --gcn_dim 768 \
    --mem_dim 768 \
    --sampling_processes 4 \
    --lr_warmup 0.2 \
    --weight_decay 0.01 \
    --device cuda
