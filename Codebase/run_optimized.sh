#!/bin/bash
# Optimized training with reduced dimensions, higher LR, and aggressive augmentation
python ./train.py \
    --dataset 14res \
    --batch_size 12 \
    --lr 0.0003 \
    --epochs 100 \
    --gcn_type adaptive \
    --attention_heads 8 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --emb_dim 768 \
    --hidden_dim 256 \
    --deberta_feature_dim 768 \
    --gcn_dim 512 \
    --mem_dim 512 \
    --sampling_processes 8 \
    --lr_warmup 0.3 \
    --weight_decay 0.01 \
    --device cuda
