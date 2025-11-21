# ✅ Kaggle Training - Simplified!

## 🎯 3-Step Process

### Step 1: Open Notebook
https://www.kaggle.com/code/uom239364u/d2e2s-training

### Step 2: Add Dataset
1. Click **"+ Add data"** (right sidebar)
2. Click **"Your Datasets"** tab
3. Find **"d2e2s-codebase"**
4. Click **"Add"**

### Step 3: Enable GPU & Run
1. Click **"⚙️ Settings"** → **Accelerator** → **"GPU T4 x2"**
2. Click **"▶️ Run All"**

---

## 📊 What Happens

The codebase already contains the dataset in `data/` folder, so:
1. ✅ Copies entire codebase (with data) to `/kaggle/working/`
2. ✅ Installs dependencies
3. ✅ Runs training (15res, 100 epochs, optimal params)
4. ✅ Shows results

**Training Time**: ~2-3 hours on GPU

---

## 📥 Download Results

After training:
```bash
kaggle kernels output uom239364u/d2e2s-training -p ./results
```

Or download from "Output" tab in notebook.

---

## 🔄 Update Code

```bash
cd Codebase
kaggle datasets version -m "Updated" -r zip
```

Then re-run notebook.

---

**Notebook**: https://www.kaggle.com/code/uom239364u/d2e2s-training
**Dataset**: https://www.kaggle.com/datasets/uom239364u/d2e2s-codebase
