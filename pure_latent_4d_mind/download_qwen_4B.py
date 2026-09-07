import os
from huggingface_hub import snapshot_download

# Define model paths
REPO_ID = "Qwen/Qwen3.5-4B"
LOCAL_DIR = "../Qwen3.5-4B"

print(f"Starting download of {REPO_ID} to {LOCAL_DIR}...")
try:
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4
    )
    print("Download completed successfully!")
except Exception as e:
    print(f"Error during download: {e}")
