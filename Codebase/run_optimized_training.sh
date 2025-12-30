#!/bin/bash

# Optimized training script matching old codebase hyperparameters
# Expected performance: 75-80% F1 (matching old codebase)

echo "=========================================="
echo "D2E2S Optimized Training"
echo "Key Changes:"
echo "  - Learning Rate: 5e-6 (60x lower)"
echo "  - Model: deberta-v2-xxlarge (1.5B params)"
echo "  - Dimensions: emb_dim=1536, hidden_dim=768"
echo "  - Batch Size: 16"
echo "=========================================="

# Train on 14res dataset (1266 samples - best balance)
python train_optimized.py \
    --dataset 14res \
    --lr 5e-6 \
    --batch_size 16 \
    --epochs 120 \
    --pretrained_deberta_name microsoft/deberta-v2-xxlarge \
    --deberta_feature_dim 1536 \
    --hidden_dim 768 \
    --emb_dim 1536 \
    --gcn_dropout 0.1 \
    --prop_drop 0.05 \
    --drop_out_rate 0.3 \
    --num_layers 2 \
    --max_span_size 6 \
    --neg_entity_count 50 \
    --neg_triple_count 50 \
    --weight_decay 0.01 \
    --lr_warmup 0.1 \
    --max_grad_norm 1.0 \
    --seed 42

echo "Training complete!"
