# D2E2S Kaggle Training (GitHub Method)

## 🚀 2-Step Setup

### Step 1: Open Notebook
https://www.kaggle.com/code/uom239364u/d2e2s-training

### Step 2: Enable GPU & Run
1. Settings → Accelerator → **GPU T4 x2**
2. Settings → Internet → **On** (required for git clone)
3. Click **"Run All"**

---

## ✅ What Happens

1. Clones repo from GitHub (includes data)
2. Installs dependencies
3. Runs training (15res, 100 epochs)
4. Shows results

**Training Time**: ~2-3 hours

---

## 🔄 Update Code

Push changes to GitHub:
```bash
cd /Users/vishal.thenuwara/Documents/MSC/Research/D2E2S
git add .
git commit -m "Update"
git push
```

Then re-run notebook on Kaggle.

---

## 📥 Download Results

```bash
kaggle kernels output uom239364u/d2e2s-training -p ./results
```

---

**That's it! No dataset uploads, no manual linking.**
