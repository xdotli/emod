#!/usr/bin/env python3
"""
Simple script to debug Modal API functionality
"""

import modal
import os
from pathlib import Path

# Create volume references
data_volume = modal.Volume.from_name("emod-data-vol", create_if_missing=True)
results_volume = modal.Volume.from_name("emod-results-vol", create_if_missing=True)

# Create app
app = modal.App("emod-debug")

@app.function(volumes={"/root/data": data_volume})
def upload_datasets():
    import os
    import shutil
    from pathlib import Path
    
    print("Inside upload_datasets function")
    
    # Check volume mount
    print(f"Volume mounted at: /root/data")
    print(f"Contents: {os.listdir('/root/data')}")
    
    # Create test file
    with open("/root/data/test.txt", "w") as f:
        f.write("Test file")
    
    print("Created test file")
    data_volume.commit()
    
    return {"status": "success"}

@app.local_entrypoint()
def main():
    print("Starting Modal debug script...")
    
    print("Created volume references")
    
    # Run the function
    print("Calling upload_datasets function...")
    result = upload_datasets.remote()
    print(f"Result: {result}")

if __name__ == "__main__":
    main() 