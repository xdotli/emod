#!/usr/bin/env python3
"""
Simple script to upload IEMOCAP datasets to Modal volume
"""

import modal
import os
from pathlib import Path

# Get paths to the datasets
current_dir = Path(__file__).parent.absolute()
repo_root = current_dir.parent

IEMOCAP_FINAL_PATH = repo_root / "IEMOCAP_Final.csv"
IEMOCAP_FILTERED_PATH = repo_root / "IEMOCAP_Filtered.csv"

print(f"IEMOCAP_Final.csv path: {IEMOCAP_FINAL_PATH}")
print(f"IEMOCAP_Filtered.csv path: {IEMOCAP_FILTERED_PATH}")

# Check if files exist locally
if not IEMOCAP_FINAL_PATH.exists():
    print(f"WARNING: IEMOCAP_Final.csv not found at {IEMOCAP_FINAL_PATH}")

if not IEMOCAP_FILTERED_PATH.exists():
    print(f"WARNING: IEMOCAP_Filtered.csv not found at {IEMOCAP_FILTERED_PATH}")

# Create persistent volume for datasets
data_volume = modal.Volume.from_name("emod-data-vol", create_if_missing=True)
DATA_VOLUME_PATH = "/root/data"

# Create Modal app
app = modal.App("emod-dataset-uploader")

@app.function(
    volumes={DATA_VOLUME_PATH: data_volume},
    mounts=[
        modal.Mount.from_local_file(IEMOCAP_FINAL_PATH, "/root/IEMOCAP_Final.csv"),
        modal.Mount.from_local_file(IEMOCAP_FILTERED_PATH, "/root/IEMOCAP_Filtered.csv")
    ]
)
def copy_datasets_to_volume():
    """Copy mounted datasets to the volume"""
    import os
    import shutil
    
    # Make sure the data directory exists
    os.makedirs(DATA_VOLUME_PATH, exist_ok=True)
    
    # List of datasets to copy from mounted files to volume
    datasets = [
        ("/root/IEMOCAP_Final.csv", f"{DATA_VOLUME_PATH}/IEMOCAP_Final.csv"),
        ("/root/IEMOCAP_Filtered.csv", f"{DATA_VOLUME_PATH}/IEMOCAP_Filtered.csv")
    ]
    
    results = []
    
    for src, dst in datasets:
        if os.path.exists(src):
            try:
                print(f"Copying {src} to {dst}")
                shutil.copyfile(src, dst)
                size = os.path.getsize(dst)
                print(f"✓ Successfully copied to volume. Size: {size/1024/1024:.2f} MB")
                results.append({"file": os.path.basename(src), "status": "success", "size": size})
            except Exception as e:
                print(f"✗ Error copying {src}: {e}")
                results.append({"file": os.path.basename(src), "status": "error", "error": str(e)})
        else:
            print(f"✗ Source file not found: {src}")
            results.append({"file": os.path.basename(src), "status": "missing"})
    
    # Commit the changes to the volume
    print("Committing changes to volume...")
    data_volume.commit()
    
    # List files in the volume
    print("Files in volume:")
    for file in os.listdir(DATA_VOLUME_PATH):
        size = os.path.getsize(os.path.join(DATA_VOLUME_PATH, file))
        print(f"  - {file} ({size/1024/1024:.2f} MB)")
    
    return results

@app.local_entrypoint()
def main():
    print("Uploading datasets to Modal volume...")
    results = copy_datasets_to_volume.remote()
    
    print("\nUpload results:")
    for result in results:
        if result["status"] == "success":
            print(f"✓ {result['file']}: Successfully uploaded ({result['size']/1024/1024:.2f} MB)")
        else:
            print(f"✗ {result['file']}: Failed to upload. Status: {result['status']}")
    
    print("\nAll operations complete.") 