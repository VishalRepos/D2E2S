I'll help you identify the top 5 commands from the README_COMMANDS.md file, ordered from highest to lowest expected performance. Let me first check the current README_COMMANDS.md file to see what commands are available.

Read file: Codebase/README_COMMANDS.md
Now let me also check the original README_improvments_2.md to see if there are any performance benchmarks mentioned there that can help me rank the commands by expected performance.

Read file: Codebase/README_improvments_2.md
Let me continue reading to see if there are more performance details:

Ran tool
Based on the performance information from both README files, I can now provide you with the top 5 commands ordered from highest to lowest expected performance:

## Top 5 Commands by Expected Performance (High to Low)

### **1. Hybrid GCN (Highest Performance - 12-15% F1 improvement)**
```bash
python train_improved.py --dataset 14res --gcn_type hybrid --gcn_heads 12 --gcn_layers 3
```
*Expected Performance: 12-15% F1 improvement*
*Why: Hybrid combines multiple GCN approaches for maximum performance*

### **2. Adaptive GCN (Very High Performance - 8-12% F1 improvement)**
```bash
python train_improved.py --dataset 14res --gcn_type adaptive --gcn_layers 3 --attention_heads 12
```
*Expected Performance: 8-12% F1 improvement*
*Why: Adaptive learning with dynamic features, best overall performance*

### **3. Enhanced Improved GCN (High Performance - 7-10% F1 improvement)**
```bash
python train_improved.py --dataset 14res --gcn_type improved --attention_heads 8 --gcn_layers 3 --use_residual True --use_layer_norm True --use_multi_scale True --use_graph_attention True
```
*Expected Performance: 7-10% F1 improvement*
*Why: Improved GCN with all advanced features enabled*

### **4. GATv2Conv (Good Performance - 6-9% F1 improvement)**
```bash
python train_improved.py --dataset 14res --gcn_type gatv2 --gcn_heads 8 --gcn_layers 2
```
*Expected Performance: 6-9% F1 improvement*
*Why: Graph Attention Network v2, similar to TIN upgrade*

### **5. Basic Improved GCN (Moderate Performance - 5-8% F1 improvement)**
```bash
python train_improved.py --dataset 14res --gcn_type improved --attention_heads 8
```
*Expected Performance: 5-8% F1 improvement*
*Why: Standard improved GCN, good balance of performance and speed*

## Performance Ranking Summary:
1. **Hybrid GCN** - Best performance (12-15% improvement)
2. **Adaptive GCN** - Very high performance (8-12% improvement)  
3. **Enhanced Improved GCN** - High performance (7-10% improvement)
4. **GATv2Conv** - Good performance (6-9% improvement)
5. **Basic Improved GCN** - Moderate performance (5-8% improvement)

**Recommendation**: Start with command #5 (Basic Improved GCN) to test your setup, then move up to higher performance commands as needed.