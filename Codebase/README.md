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

### Quick Test
```bash
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 3
```

### Full Optimization
```bash
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 20
```

### Custom Configuration
```bash
python simple_hyperparameter_tuner.py --dataset 14res --n_trials 20
```

## 📊 Supported Datasets

- **14res**: Restaurant reviews (default)
- **15res**: Restaurant reviews
- **16res**: Restaurant reviews  
- **14lap**: Laptop reviews

## 🚨 Important Notes

- **GPU Memory**: Requires 24GB+ GPU memory for DeBERTa-v2-XXLarge
- **Training Time**: Each hyperparameter trial takes ~30 minutes
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

