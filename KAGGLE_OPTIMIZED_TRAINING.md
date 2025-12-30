# Kaggle Training Guide - Optimized Hyperparameters

## Quick Setup

### 1. Create New Kaggle Notebook
- Go to https://www.kaggle.com/code
- Click "New Notebook"
- Settings → Accelerator: **GPU T4 x2** (or P100)
- Settings → Internet: **ON**

### 2. Clone Repository

```python
!git clone https://github.com/VishalRepos/D2E2S.git
%cd D2E2S/Codebase
```

### 3. Install Dependencies

```python
!pip install -q transformers torch torchvision torchaudio
!pip install -q scikit-learn tqdm
```

### 4. Download DeBERTa Model (IMPORTANT)

```python
# Download deberta-v2-xxlarge (1.5B params)
from transformers import AutoTokenizer, AutoModel

print("Downloading DeBERTa-v2-xxlarge...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")
model = AutoModel.from_pretrained("microsoft/deberta-v2-xxlarge")

# Save locally
tokenizer.save_pretrained("./deberta-v2-xxlarge")
model.save_pretrained("./deberta-v2-xxlarge")
print("Download complete!")
```

### 5. Run Optimized Training

```python
!python train.py \
    --dataset 14res \
    --lr 5e-6 \
    --batch_size 16 \
    --epochs 120 \
    --pretrained_deberta_name ./deberta-v2-xxlarge \
    --deberta_feature_dim 1536 \
    --hidden_dim 768 \
    --emb_dim 1536
```

---

## Alternative: Use Shell Script

```bash
!chmod +x run_optimized_training.sh
!./run_optimized_training.sh
```

---

## Expected Results

- **Training Time:** ~20-25 hours on T4 x2
- **Expected F1:** 75-80% on 14res
- **Memory Usage:** ~14-15GB GPU RAM

---

## If GPU Memory Issues

### Option 1: Reduce Batch Size
```python
!python train.py --dataset 14res --batch_size 8
```

### Option 2: Use Gradient Accumulation
Add to Parameter_Optimized.py:
```python
parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
```

### Option 3: Use Smaller Model (Compromise)
```python
!python train.py \
    --dataset 14res \
    --pretrained_deberta_name microsoft/deberta-v3-large \
    --deberta_feature_dim 1024 \
    --hidden_dim 512 \
    --emb_dim 1024 \
    --lr 1e-5
```
Expected: 65-70% F1 (better than 53%, worse than 80%)

---

## Complete Kaggle Notebook Code

```python
# Cell 1: Setup
!git clone https://github.com/VishalRepos/D2E2S.git
%cd D2E2S/Codebase

# Cell 2: Install Dependencies
!pip install -q transformers torch scikit-learn tqdm

# Cell 3: Download Model
from transformers import AutoTokenizer, AutoModel
print("Downloading DeBERTa-v2-xxlarge (1.5B params)...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")
model = AutoModel.from_pretrained("microsoft/deberta-v2-xxlarge")
tokenizer.save_pretrained("./deberta-v2-xxlarge")
model.save_pretrained("./deberta-v2-xxlarge")
print("✅ Model downloaded!")

# Cell 4: Verify Files
!ls -la Parameter_Optimized.py run_optimized_training.sh

# Cell 5: Start Training
!python train.py --dataset 14res

# Cell 6: Monitor Training (in separate cell)
!tail -f log/14res/train_*.csv
```

---

## Monitoring Training

### Check Progress
```python
import pandas as pd
df = pd.read_csv('log/14res/train_log.csv')
print(df.tail(10))
```

### Plot F1 Score
```python
import matplotlib.pyplot as plt
plt.plot(df['epoch'], df['triplet_f1'])
plt.xlabel('Epoch')
plt.ylabel('Triplet F1')
plt.title('Training Progress')
plt.show()
```

---

## Save Best Model

The model automatically saves to:
```
savemodels/14res/best_model.pt
```

Download from Kaggle:
- Right-click on file → Download
- Or use Kaggle API

---

## Troubleshooting

### "CUDA out of memory"
```python
# Reduce batch size
!python train.py --dataset 14res --batch_size 8
```

### "Model download failed"
```python
# Use Kaggle dataset instead
# Add microsoft/deberta-v2-xxlarge as dataset input
```

### "Training too slow"
- Ensure GPU is enabled (Settings → Accelerator)
- Use T4 x2 or P100 for faster training
- Consider using deberta-v3-large instead

---

## Key Parameters (Already Set in Parameter_Optimized.py)

```python
lr = 5e-6                    # Critical: 60x lower than before
model = deberta-v2-xxlarge   # 1.5B params
batch_size = 16              # Stable batch size
hidden_dim = 768             # Matching model output
emb_dim = 1536               # Matching model output
prop_drop = 0.05             # Conservative dropout
gcn_dropout = 0.1            # Conservative dropout
max_span_size = 6            # Reduced noise
```

---

## Expected Timeline

| Epoch | Time | Expected F1 |
|-------|------|-------------|
| 10 | ~3h | ~40% |
| 30 | ~9h | ~60% |
| 60 | ~18h | ~75% |
| 80 | ~24h | ~78-80% |

Early stopping likely around epoch 60-80.

---

## After Training

### Evaluate on Test Set
```python
!python train.py --dataset 14res --final_eval
```

### Check Results
```python
!cat log/14res/result6.txt
```

### Download Model
```python
from google.colab import files
files.download('savemodels/14res/best_model.pt')
```

---

## Summary

1. ✅ Clone repo
2. ✅ Install dependencies
3. ✅ Download deberta-v2-xxlarge
4. ✅ Run `python train.py --dataset 14res`
5. ⏳ Wait ~20-25 hours
6. ✅ Achieve 75-80% F1

**All hyperparameters are pre-configured in Parameter_Optimized.py!**
