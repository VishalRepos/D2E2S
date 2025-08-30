# Optimal Hyperparameters for D2E2S Model

## Overview
These hyperparameters are based on the best performing configuration from hyperparameter optimization (Trial 4), achieving:
- **NER F1-Score (Micro):** 87.72%
- **Sentiment F1-Score (Micro):** 76.54%
- **Overall Performance:** Best among all trials

## Model Architecture Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 8 | Training batch size |
| `gcn_type` | "hybrid" | Type of GCN to use |
| `gcn_layers` | 2 | Number of GCN layers |
| `attention_heads` | 8 | Number of attention heads |
| `hidden_dim` | 768 | Hidden layer dimension |
| `gcn_dim` | 768 | Dimension of GCN |

## Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `epochs` | 40 | Number of training epochs |
| `lr` | 1.71e-06 | Learning rate |
| `lr_warmup` | 0.15 | Learning rate warmup proportion |
| `weight_decay` | 0.01 | Weight decay for regularization |
| `max_grad_norm` | 1.0 | Maximum gradient norm for clipping |

## GCN & Attention Features

| Parameter | Value | Description |
|-----------|-------|-------------|
| `use_residual` | True | Use residual connections |
| `use_layer_norm` | True | Use layer normalization |
| `use_multi_scale` | True | Use multi-scale features |
| `use_graph_attention` | True | Use graph attention mechanism |
| `use_relative_position` | True | Use relative position encoding |
| `use_global_context` | True | Use global context modeling |

## Dropout & Regularization

| Parameter | Value | Description |
|-----------|-------|-------------|
| `drop_out_rate` | 0.3 | Main dropout rate |
| `gcn_dropout` | 0.1 | GCN-specific dropout |
| `prop_drop` | 0.05 | D2E2S-specific dropout |

## Other Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `max_pairs` | 800 | Maximum entity pairs to process |
| `max_span_size` | 6 | Maximum size of entity spans |
| `neg_entity_count` | 50 | Negative entity samples per sample |
| `neg_triple_count` | 50 | Negative triplet samples per sample |
| `sen_filter_threshold` | 0.5 | Sentiment filter threshold |

## Implementation Notes

These parameters are now set as **defaults** in `Codebase/Parameter.py`, so you can:

1. **Use directly:** Run training without specifying any arguments
2. **Override selectively:** Pass specific arguments to override any defaults
3. **Experiment further:** Use these as a baseline for additional tuning

## Usage Examples

```bash
# Use optimal defaults
python train.py

# Override specific parameters
python train.py --batch_size 16 --lr 2e-06

# Use different dataset
python train.py --dataset 14res
```

## Performance Comparison

| Trial | NER F1 (Micro) | Sentiment F1 (Micro) | Overall Rank |
|-------|----------------|----------------------|--------------|
| **Trial 4 (This)** | **87.72%** | **76.54%** | **🥇 1st** |
| Trial 7 | 85.04% | 70.77% | 🥈 2nd |
| Trial 8 | 85.01% | 71.32% | 🥉 3rd |
| Trial 6 | 84.86% | 72.30% | 4th |
| Trial 3 | 84.88% | 71.23% | 5th |

## Key Insights

1. **GCN Layers:** 2 layers performed better than 3-4 layers
2. **Attention Heads:** 8 heads worked better than 12 heads  
3. **Batch Size:** 8 provided optimal balance between memory and performance
4. **Hybrid GCN:** The hybrid approach consistently outperformed other GCN types
5. **Learning Rate:** Lower learning rate (1.71e-06) provided better convergence

## File Locations

- **Parameter Definitions:** `Codebase/Parameter.py`
- **Hyperparameter Results:** `Codebase/hyperparameter_results/` (gitignored)
- **This Documentation:** `Codebase/OPTIMAL_HYPERPARAMETERS.md`

---
*Last Updated: Based on hyperparameter optimization results from Trial 4*
