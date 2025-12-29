#!/bin/bash
# Optimized training with Focal Loss, Label Smoothing, and aggressive augmentation
# Reduced batch size to 8 for larger 14res dataset
python ./train.py \
    --dataset 14res \
    --batch_size 8 \
    --lr 0.0003 \
    --epochs 100 \
    --gcn_type adaptive \
    --attention_heads 8 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --emb_dim 768 \
    --hidden_dim 384 \
    --deberta_feature_dim 768 \
    --gcn_dim 768 \
    --mem_dim 768 \
    --sampling_processes 4 \
    --lr_warmup 0.3 \
    --weight_decay 0.01 \
    --device cuda
