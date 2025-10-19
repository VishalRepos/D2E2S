# Quick Kaggle Setup Script
# Run this in a Kaggle notebook cell to get started quickly

import os
import zipfile

# 1. Extract dataset (update path to your uploaded dataset)
dataset_path = '/kaggle/input/d2e2s-dataset/d2e2s_dataset.zip'
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall('/kaggle/working/')

# 2. Install dependencies
os.system('pip install torch==2.4.0 torch-geometric==2.3.1 transformers==4.28.1 tensorboardX==2.6')

# 3. Add to Python path
import sys
sys.path.append('/kaggle/working')

# 4. Optimal config
OPTIMAL_CONFIG = {
    "batch_size": 6, "lr": 0.000312, "lr_warmup": 0.12, "weight_decay": 0.0012,
    "gcn_type": "hybrid", "gcn_layers": 2, "attention_heads": 16, 
    "hidden_dim": 1024, "gcn_dim": 768, "epochs": 40
}

print("✅ Quick setup complete! Ready to train D2E2S model.")
print(f"Expected F1 Score: ~0.8644")