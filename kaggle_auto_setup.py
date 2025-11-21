#!/usr/bin/env python3
"""
Automated Kaggle Setup for D2E2S Training
This script will:
1. Create a Kaggle dataset from your codebase
2. Create/update a Kaggle notebook
3. Push everything to Kaggle
"""

import os
import json
import subprocess
import sys
from pathlib import Path

# Configuration
DATASET_SLUG = "d2e2s-codebase"  # Will be: username/d2e2s-codebase
NOTEBOOK_SLUG = "d2e2s-training"  # Will be: username/d2e2s-training
CODEBASE_DIR = "Codebase"
NOTEBOOK_FILE = "D2E2S_Kaggle_Training.ipynb"

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False
    print(f"✅ Success!")
    if result.stdout:
        print(result.stdout)
    return True

def get_kaggle_username():
    """Get Kaggle username from credentials"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    with open(kaggle_json) as f:
        creds = json.load(f)
    return creds["username"]

def create_dataset_metadata():
    """Create dataset-metadata.json for the codebase"""
    username = get_kaggle_username()
    
    metadata = {
        "title": "D2E2S Codebase",
        "id": f"{username}/{DATASET_SLUG}",
        "licenses": [{"name": "MIT"}],
        "keywords": ["nlp", "sentiment-analysis", "absa", "pytorch", "gcn"]
    }
    
    # Create metadata in Codebase directory
    metadata_path = Path(CODEBASE_DIR) / "dataset-metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created dataset metadata: {metadata_path}")
    return metadata_path

def create_notebook_metadata():
    """Create kernel-metadata.json for the notebook"""
    username = get_kaggle_username()
    
    metadata = {
        "id": f"{username}/{NOTEBOOK_SLUG}",
        "title": "D2E2S Training",
        "code_file": NOTEBOOK_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{username}/{DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": []
    }
    
    with open("kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created notebook metadata: kernel-metadata.json")
    return "kernel-metadata.json"

def main():
    print("\n" + "="*60)
    print("🚀 D2E2S Kaggle Automated Setup")
    print("="*60)
    
    # Get username
    username = get_kaggle_username()
    print(f"\n👤 Kaggle Username: {username}")
    
    # Step 1: Create dataset metadata
    print("\n📦 Step 1: Preparing Dataset")
    create_dataset_metadata()
    
    # Step 2: Create/Update dataset
    print("\n📤 Step 2: Uploading Dataset to Kaggle")
    dataset_exists = subprocess.run(
        f"kaggle datasets list --user {username} | grep {DATASET_SLUG}",
        shell=True, capture_output=True
    ).returncode == 0
    
    if dataset_exists:
        print(f"Dataset '{DATASET_SLUG}' exists. Updating...")
        cmd = f"cd {CODEBASE_DIR} && kaggle datasets version -m 'Updated codebase' -r zip"
    else:
        print(f"Creating new dataset '{DATASET_SLUG}'...")
        cmd = f"cd {CODEBASE_DIR} && kaggle datasets create -r zip"
    
    if not run_command(cmd, "Uploading codebase to Kaggle"):
        print("❌ Failed to upload dataset")
        return False
    
    # Step 3: Create notebook metadata
    print("\n📓 Step 3: Preparing Notebook")
    create_notebook_metadata()
    
    # Step 4: Push notebook
    print("\n📤 Step 4: Pushing Notebook to Kaggle")
    notebook_exists = subprocess.run(
        f"kaggle kernels list --user {username} | grep {NOTEBOOK_SLUG}",
        shell=True, capture_output=True
    ).returncode == 0
    
    if notebook_exists:
        print(f"Notebook '{NOTEBOOK_SLUG}' exists. Updating...")
        cmd = "kaggle kernels push"
    else:
        print(f"Creating new notebook '{NOTEBOOK_SLUG}'...")
        cmd = "kaggle kernels push"
    
    if not run_command(cmd, "Pushing notebook to Kaggle"):
        print("❌ Failed to push notebook")
        return False
    
    # Success!
    print("\n" + "="*60)
    print("✅ SUCCESS! Everything is set up on Kaggle")
    print("="*60)
    print(f"\n📊 Dataset URL: https://www.kaggle.com/datasets/{username}/{DATASET_SLUG}")
    print(f"📓 Notebook URL: https://www.kaggle.com/code/{username}/{NOTEBOOK_SLUG}")
    print("\n🎯 Next Steps:")
    print("1. Go to your notebook URL")
    print("2. Click 'Run All' to start training")
    print("3. Monitor progress and download results when done")
    print("\n" + "="*60)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
