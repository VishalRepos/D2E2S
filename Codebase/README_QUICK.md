## 🚀 **Top 5 Commands by Performance (High to Low)**

### **1. 🥇 HIGHEST PERFORMANCE: Full Enhanced Training**
```bash
python train_improved.py --dataset 14res --gcn_type improved --gcn_layers 4 --attention_heads 12 --use_residual --use_layer_norm --use_multi_scale --use_graph_attention --use_adaptive_edges --use_relative_position --use_global_context --batch_size 16 --epochs 100
```
**Performance Level:** ⭐⭐⭐⭐⭐ (Highest)
- Uses all advanced GCN improvements
- Multi-scale feature processing
- Adaptive edge learning
- Global context modeling
- 4 GCN layers with 12 attention heads

### **2. 🥈 HIGH PERFORMANCE: Advanced Configuration**
```bash
python train_improved.py --dataset 14res --gcn_type adaptive --gcn_layers 3 --attention_heads 8 --use_residual --use_layer_norm --use_multi_scale --use_graph_attention --batch_size 16 --epochs 80
```
**Performance Level:** ⭐⭐⭐⭐ (Very High)
- Adaptive GCN with dynamic edge weights
- Multi-scale aggregation
- Graph attention mechanisms
- 3 GCN layers with 8 attention heads

### **3. 🥉 MEDIUM-HIGH PERFORMANCE: Standard Improved**
```bash
python train_improved.py --dataset 14res --gcn_type improved --gcn_layers 3 --attention_heads 8 --use_residual --use_layer_norm --batch_size 16 --epochs 60
```
**Performance Level:** ⭐⭐⭐ (High)
- Improved GCN modules
- Residual connections
- Layer normalization
- Balanced performance vs training time

### **4. �� MEDIUM PERFORMANCE: Basic Improved**
```bash
python train_improved.py --dataset 14res --gcn_type improved --gcn_layers 2 --attention_heads 4 --batch_size 16 --epochs 50
```
**Performance Level:** ⭐⭐ (Medium)
- Basic improved GCN
- Fewer layers and attention heads
- Faster training, moderate performance

### **5. �� LOWEST PERFORMANCE: Original Model**
```bash
python train.py --dataset 14res --seed 42 --max_span_size 8 --batch_size 16 --epochs 50
```
**Performance Level:** ⭐ (Lowest)
- Original D2E2S model
- Basic GCN implementation
- No improvements
- Baseline performance
