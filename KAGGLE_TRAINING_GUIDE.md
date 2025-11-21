# D2E2S Kaggle Training Guide (Web Interface)

## 📦 Files Prepared
- ✅ `D2E2S_Codebase.zip` - Your complete codebase (ready to upload)
- ✅ `D2E2S_Kaggle_Training.ipynb` - Training notebook

---

## 🚀 Step-by-Step Instructions

### **Step 1: Upload Codebase as a Dataset**

1. Go to [Kaggle Datasets](https://www.kaggle.com/datasets)
2. Click **"New Dataset"** button (top right)
3. Click **"Upload"** and select `D2E2S_Codebase.zip`
4. Fill in details:
   - **Title**: `D2E2S Codebase`
   - **Subtitle**: `Dual-channel Enhanced Entity-Sentiment Model`
   - **Description**: `Complete codebase for D2E2S ABSA model training`
5. Click **"Create"**
6. **Note the dataset URL** (e.g., `yourusername/d2e2s-codebase`)

---

### **Step 2: Create a New Kaggle Notebook**

1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Click **"New Notebook"**
3. Click **"File" → "Upload Notebook"**
4. Select `D2E2S_Kaggle_Training.ipynb`

---

### **Step 3: Add Your Dataset to the Notebook**

1. In the notebook, click **"+ Add data"** (right sidebar)
2. Search for your dataset: `d2e2s-codebase`
3. Click **"Add"** to attach it to your notebook

---

### **Step 4: Enable GPU Accelerator**

1. Click **"Settings"** (right sidebar)
2. Under **"Accelerator"**, select **"GPU T4 x2"** or **"GPU P100"**
3. Click **"Save"**

---

### **Step 5: Modify Notebook Paths**

In the notebook, update the path to your codebase:

```python
# After installing dependencies, add this cell:
import os
import sys

# Path to your uploaded dataset
CODEBASE_PATH = "/kaggle/input/d2e2s-codebase/Codebase"
sys.path.insert(0, CODEBASE_PATH)
os.chdir(CODEBASE_PATH)
```

---

### **Step 6: Run Training**

1. Click **"Run All"** or run cells sequentially
2. Monitor training progress in the output
3. Training will take ~2-4 hours depending on GPU

---

### **Step 7: Download Results**

After training completes:

1. Results will be in `/kaggle/working/` directory
2. Click **"File" → "Download"** to get:
   - Trained model weights
   - Training logs
   - Performance metrics

---

## 🎯 Quick Training Command

If you want to run training directly in a code cell:

```python
!cd /kaggle/input/d2e2s-codebase/Codebase && \
python train.py \
  --dataset 15res \
  --batch_size 8 \
  --lr 0.0001716 \
  --epochs 100 \
  --gcn_type adaptive \
  --attention_heads 12 \
  --device cuda
```

---

## 📊 Expected Results (15res dataset)

Based on optimal hyperparameters:
- **Accuracy**: ~82-84%
- **F1 Score**: ~75-77%
- **Training Time**: ~2-3 hours on GPU

---

## ⚠️ Important Notes

1. **GPU Quota**: Kaggle provides 30 hours/week of GPU time
2. **Session Timeout**: Sessions timeout after 12 hours
3. **Save Frequently**: Use checkpoints to save progress
4. **Internet Access**: Enable if you need to download models

---

## 🔧 Troubleshooting

### Issue: "Module not found"
**Solution**: Make sure you've added the codebase path to sys.path

### Issue: "CUDA out of memory"
**Solution**: Reduce batch_size to 4 or 2

### Issue: "Dataset not found"
**Solution**: Verify dataset is attached in the "Data" tab (right sidebar)

---

## 📝 Next Steps After Training

1. Download trained model from `/kaggle/working/`
2. Evaluate on test set
3. Compare results with local training
4. Save notebook version for reproducibility

---

**Good luck with your training! 🚀**
