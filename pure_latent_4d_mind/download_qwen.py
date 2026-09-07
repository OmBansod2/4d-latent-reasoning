import os
from huggingface_hub import snapshot_download

# Disable symlinks to prevent lock issues in macOS
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

print("Downloading Qwen 3.8 27B model...")
try:
    path = snapshot_download(
        repo_id="lmstudio-community/Qwen3.8-27B-MLX-4bit",
        local_dir="../Qwen3.8-27B",
        local_dir_use_symlinks=False,
        max_workers=8
    )
    print(f"\nDownload completed successfully! Saved to: {path}")
except Exception as e:
    print(f"\nError downloading model: {e}")
