#!/bin/bash
# Optimized training parameters for Kaggle GPU constraints
python ./train.py \
    --dataset 15res \
    --batch_size 12 \
    --lr 0.0005 \
    --epochs 150 \
    --gcn_type adaptive \
    --attention_heads 8 \
    --pretrained_deberta_name microsoft/deberta-v3-base \
    --emb_dim 768 \
    --hidden_dim 384 \
    --deberta_feature_dim 768 \
    --gcn_dim 768 \
    --mem_dim 768 \
    --sampling_processes 4 \
    --device cuda
