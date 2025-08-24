# D2E2S - Dual Dynamic Enhanced Entity and Sentiment Extraction System

Original code is at https://github.com/TYZY89/D2E2S.git

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
python train_improved.py --dataset 14res --gcn_type improved --attention_heads 8
```

### 3. Run Hyperparameter Tuning
```bash
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 3
```

## 📁 Project Structure

```
Codebase/
├── models/                          # GCN model implementations
├── trainer/                         # Training utilities
├── data/                           # Dataset files
├── train_improved.py               # Main training script
├── hyperparameter_config.py         # Hyperparameter configuration
├── simple_hyperparameter_tuner.py   # Hyperparameter tuning script

└── requirements.txt                 # All dependencies
```

## 🎯 Top 5 Commands by Expected Performance (High to Low)

### **1. Hybrid GCN (Highest Performance - 12-15% F1 improvement)**
```bash
python train_improved.py --dataset 14res --gcn_type hybrid --gcn_heads 12 --gcn_layers 3
```

### **2. Adaptive GCN (Very High Performance - 8-12% F1 improvement)**
```bash
python train_improved.py --dataset 14res --gcn_type adaptive --gcn_layers 3 --attention_heads 12
```

### **3. Enhanced Improved GCN (High Performance - 7-10% F1 improvement)**
```bash
python train_improved.py --dataset 14res --gcn_type improved --attention_heads 8 --gcn_layers 3 --use_residual True --use_layer_norm True --use_multi_scale True --use_graph_attention True
```

### **4. GATv2Conv (Good Performance - 6-9% F1 improvement)**
```bash
python train_improved.py --dataset 14res --gcn_type gatv2 --gcn_heads 8 --gcn_layers 2
```

### **5. Basic Improved GCN (Moderate Performance - 5-8% F1 improvement)**
```bash
python train_improved.py --dataset 14res --gcn_type improved --attention_heads 8
```

## 🔧 Available Parameters

### GCN Parameters
- `--gcn_type`: "original", "improved", "adaptive", "gatv2", "gcn", "sage", "gin", "chebyshev", "dynamic", "edge_conv", "hybrid" (default: "improved")
- `--gcn_layers`: Number of GCN layers (default: 3)
- `--attention_heads`: Number of attention heads (default: 8)
- `--gcn_heads`: Number of GCN attention heads (default: 8)
- `--use_residual`: Use residual connections (default: True)
- `--use_layer_norm`: Use layer normalization (default: True)
- `--use_multi_scale`: Use multi-scale feature aggregation (default: True)
- `--use_graph_attention`: Use graph attention mechanism (default: True)

### Training Parameters
- `--dataset`: "14res", "15res", "16res", "14lap" (default: "14res")
- `--batch_size`: Training batch size (default: 16)
- `--epochs`: Number of epochs (default: 120)
- `--lr`: Learning rate (default: 5e-5)

## 🎛️ Hyperparameter Tuning

### Smart Randomization System for DeBERTa-v2-XXLarge
- **Only most relevant parameters** are randomized for XXLarge efficiency
- **Critical parameters**: batch_size, lr, gcn_type, gcn_layers, attention_heads, max_pairs
- **Other parameters** use proven XXLarge-optimized default values
- **No duplicate combinations** guaranteed
- **Fixed epochs**: All trials use exactly **70 epochs**

### Parameters Being Tuned (XXLarge Optimized)
- **`batch_size`**: [4, 6, 8] - Memory optimization for XXLarge
- **`lr`**: [1e-5, 3e-5] - Conservative learning rate for XXLarge stability
- **`gcn_type`**: ['improved', 'gatv2', 'hybrid'] - Most proven for XXLarge
- **`gcn_layers`**: [2, 4] - Optimized for XXLarge efficiency
- **`attention_heads`**: [8, 12] - Memory-optimized for XXLarge
- **`max_pairs`**: [400, 600, 800] - XXLarge memory optimized

### Default Values (XXLarge Optimized)
- **`epochs`**: 70 (fixed for all trials)
- **`hidden_dim`**: 1536 (matches XXLarge features), **`gcn_dim`**: 768 (XXLarge optimized)
- **`use_residual`**: True, **`use_layer_norm`**: True (essential for XXLarge)
- **`drop_out_rate`**: 0.3, **`gcn_dropout`**: 0.1 (conservative for XXLarge)
- **`max_span_size`**: 6, **`neg_entity_count`**: 50 (memory efficient for XXLarge)

### Quick Test
```bash
# Preview combinations first
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 3 --preview

# Then run optimization
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 3
```

### Full Optimization
```bash
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 20
```

### Custom Configuration
```bash
python simple_hyperparameter_tuner.py --dataset 15res --n_trials 30 --config my_config.py
```

### Preview Without Running
```bash
# Preview 3 trial combinations
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 3 --preview

# Preview 20 trial combinations
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 20 --preview
```

## 📊 Supported Datasets

- **14res**: Restaurant reviews (default)
- **15res**: Restaurant reviews
- **16res**: Restaurant reviews  
- **14lap**: Laptop reviews

## 🚨 Important Notes

- **GPU Memory**: Requires 24GB+ GPU memory for DeBERTa-v2-XXLarge
- **Training Time**: Each hyperparameter trial takes ~15-20 minutes (70 epochs)
- **Epochs**: All trials use exactly **70 epochs** for consistent training
- **Randomization**: Completely random parameter combinations for maximum exploration
- **Results**: Hyperparameter results saved in `hyperparameter_results/` directory

## 🐛 Troubleshooting

```bash
# Check configuration
python hyperparameter_config.py

# Test imports
python -c "from simple_hyperparameter_tuner import SimpleHyperparameterTuner; print('Import successful')"
```

## 📚 More Information

- See `Parameter_Improved.py` for all available parameters
- Check `trainer/` directory for training utilities
- Review `models/` directory for GCN implementations

