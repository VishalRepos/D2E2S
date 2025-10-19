# 🎯 Optimal Hyperparameters for D2E2S Model - 15res Dataset

## Overview
These hyperparameters are based on the best performing configuration from hyperparameter optimization for the **15res dataset**, achieving:
- **F1-Score (Micro):** 83.49%
- **Study:** d2e2s_15res_balanced_1760675899
- **Optimization Strategy:** Balanced TPE sampling with pruning

## 🏗️ Model Architecture Parameters

| Parameter | Value | Description | 14res Comparison |
|-----------|-------|-------------|------------------|
| `batch_size` | **8** | Training batch size | Same as 14res |
| `gcn_type` | **"adaptive"** | Type of GCN to use | Different (14res: hybrid) |
| `gcn_layers` | **3** | Number of GCN layers | Different (14res: 2) |
| `attention_heads` | **12** | Number of attention heads | Different (14res: 8) |
| `hidden_dim` | **768** | Hidden layer dimension | Same as 14res |
| `gcn_dim` | **512** | Dimension of GCN | Same as 14res |

## 🚀 Training Parameters

| Parameter | Value | Description | 14res Comparison |
|-----------|-------|-------------|------------------|
| `epochs` | **27** | Number of training epochs | Much lower (14res: 112) |
| `lr` | **0.0001716** | Learning rate | Much higher (14res: 1.71e-06) |
| `lr_warmup` | **0** | Learning rate warmup proportion | Same as 14res |
| `weight_decay` | **0** | Weight decay for regularization | Same as 14res |
| `max_grad_norm` | **1.0** | Maximum gradient norm for clipping | Same as 14res |

## 🧠 GCN & Attention Features

| Parameter | Value | Description | Status |
|-----------|-------|-------------|------------|
| `use_improved_gcn` | **True** | Use improved GCN modules | ✅ Enabled |
| `use_residual` | **True** | Use residual connections | ✅ Enabled |
| `use_layer_norm` | **True** | Use layer normalization | ✅ Enabled |
| `use_multi_scale` | **True** | Use multi-scale features | ✅ Enabled |
| `use_graph_attention` | **True** | Use graph attention mechanism | ✅ Enabled |
| `use_relative_position` | **True** | Use relative position encoding | ✅ Enabled |
| `use_global_context` | **True** | Use global context modeling | ✅ Enabled |

## 🛡️ Dropout & Regularization

| Parameter | Value | Description | Default |
|-----------|-------|-------------|---------|
| `drop_out_rate` | **0.3** | Main dropout rate | 0.3 |
| `gcn_dropout` | **0.1** | GCN-specific dropout | 0.1 |
| `prop_drop` | **0.05** | D2E2S-specific dropout | 0.05 |

## 🔧 Other Parameters

| Parameter | Value | Description | Default |
|-----------|-------|-------------|---------|
| `max_pairs` | **800** | Maximum entity pairs to process | 800 |
| `max_span_size` | **6** | Maximum size of entity spans | 6 |
| `neg_entity_count` | **50** | Negative entity samples per sample | 50 |
| `neg_triple_count` | **50** | Negative triplet samples per sample | 50 |
| `sen_filter_threshold` | **0.5** | Sentiment filter threshold | 0.5 |

## 📊 Performance Comparison (15res vs 14res)

| Metric | 15res Best | 14res Best | Difference |
|--------|------------|------------|------------|
| **F1 Score** | **83.49%** | **87.72%** | -4.23% |
| **Epochs** | **27** | **112** | -85 epochs |
| **Learning Rate** | **1.716e-04** | **1.71e-06** | 100x higher |
| **GCN Type** | **adaptive** | **hybrid** | Different |
| **GCN Layers** | **3** | **2** | +1 layer |
| **Attention Heads** | **12** | **8** | +4 heads |

## 🎯 Key Insights for 15res Dataset

1. **Faster Convergence:** 15res requires significantly fewer epochs (27 vs 112)
2. **Higher Learning Rate:** 15res benefits from 100x higher learning rate
3. **Adaptive GCN:** Adaptive GCN works better than hybrid for 15res
4. **More Attention:** 15res benefits from more attention heads (12 vs 8)
5. **Deeper GCN:** 15res performs better with 3 GCN layers vs 2

## 🚀 Implementation Status

### ✅ **Ready for Use:**
- All parameters optimized for 15res dataset
- Configuration saved in optuna_results/
- Compatible with existing Parameter.py structure

### 📁 **File Locations:**
- **Best Parameters:** `optuna_results/d2e2s_15res_balanced_1760675899_best_params.json` ✅
- **All Trials:** `optuna_results/d2e2s_15res_balanced_1760675899_all_trials.json` ✅
- **Statistics:** `optuna_results/d2e2s_15res_balanced_1760675899_stats.json` ✅
- **This Documentation:** `OPTIMAL_HYPERPARAMETERS_15RES.md` ✅

## 💻 Usage Examples

```bash
# Use 15res optimal parameters
python train.py --dataset 15res --batch_size 8 --lr 0.0001716 --epochs 27 --gcn_type adaptive --gcn_layers 3 --attention_heads 12

# Quick training with optimal settings
python train.py --dataset 15res

# Override specific parameters
python train.py --dataset 15res --batch_size 16 --epochs 50
```

## 🔄 Parameter Override Examples

```bash
# Training with different batch size
python train.py --dataset 15res --batch_size 16

# Experiment with more epochs
python train.py --dataset 15res --epochs 50

# Test with hybrid GCN (like 14res)
python train.py --dataset 15res --gcn_type hybrid --gcn_layers 2

# Use different learning rate
python train.py --dataset 15res --lr 0.0001
```

## 📈 Expected Performance

With these optimal hyperparameters for 15res, you should achieve:
- **F1 Performance:** ~83.49% F1-score (micro)
- **Training Speed:** Much faster convergence (27 epochs vs 112)
- **Memory Efficiency:** Optimized batch size and model dimensions
- **Stability:** Consistent performance across runs

## 🎉 Summary

Your D2E2S model is now optimized for the **15res dataset** with:

- **🎯 Dataset-Specific:** Optimized specifically for 15res characteristics
- **⚡ Fast Training:** 4x faster convergence than 14res
- **🚀 Ready to Use:** All parameters configured and documented
- **📚 Well Documented:** Complete reference available

**Key Differences from 14res:**
- **Adaptive GCN** instead of Hybrid
- **Higher learning rate** (100x)
- **Fewer epochs** needed (27 vs 112)
- **More attention heads** (12 vs 8)

**Ready to train with 15res-optimized performance!** 🚀

---
*Last Updated: Based on hyperparameter optimization results for 15res dataset*
*Study: d2e2s_15res_balanced_1760675899*
*Implementation Status: ✅ Complete and Ready*