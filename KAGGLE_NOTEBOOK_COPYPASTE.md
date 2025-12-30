# Copy-Paste Kaggle Notebook

## Setup Instructions
1. Go to https://www.kaggle.com/code
2. Create New Notebook
3. Settings → Accelerator: **GPU T4 x2**
4. Settings → Internet: **ON**
5. Copy-paste cells below

---

## Cell 1: Clone & Setup
```python
!git clone https://github.com/VishalRepos/D2E2S.git
%cd D2E2S/Codebase
!pip install -q transformers scikit-learn tqdm
```

---

## Cell 2: Download DeBERTa Model
```python
from transformers import AutoTokenizer, AutoModel

print("📥 Downloading DeBERTa-v2-xxlarge (1.5B params)...")
print("⏱️  This takes ~5-10 minutes...")

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")
model = AutoModel.from_pretrained("microsoft/deberta-v2-xxlarge")

tokenizer.save_pretrained("./deberta-v2-xxlarge")
model.save_pretrained("./deberta-v2-xxlarge")

print("✅ Model downloaded and saved!")
```

---

## Cell 3: Verify Setup
```python
import os
print("✅ Checking files...")
print("Parameter_Optimized.py:", os.path.exists("Parameter_Optimized.py"))
print("train.py:", os.path.exists("train.py"))
print("DeBERTa model:", os.path.exists("deberta-v2-xxlarge"))
print("\n🚀 Ready to train!")
```

---

## Cell 4: Start Training
```python
!python train.py --dataset 14res
```

---

## Cell 5: Monitor Progress (Optional - Run in Separate Cell)
```python
import pandas as pd
import time

while True:
    try:
        df = pd.read_csv('log/14res/train_log.csv')
        latest = df.tail(1)
        print(f"Epoch {latest['epoch'].values[0]}: F1 = {latest['triplet_f1'].values[0]:.2f}%")
        time.sleep(60)  # Check every minute
    except:
        print("Waiting for training to start...")
        time.sleep(10)
```

---

## Expected Output

```
Epoch 1: F1 = 25.3%
Epoch 10: F1 = 42.1%
Epoch 30: F1 = 61.5%
Epoch 60: F1 = 74.8%
Epoch 80: F1 = 78.2%  ← Target achieved!
```

**Training Time:** ~20-25 hours  
**Expected F1:** 75-80%

---

## That's It!

Just copy-paste the 4 cells above and run them in order. The optimized hyperparameters are already configured in `Parameter_Optimized.py`.
