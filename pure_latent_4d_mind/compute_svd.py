import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from datasets import load_dataset
import numpy as np
import gc

# -----------------------------------------------------------------------------
# Configuration & Hyperparameters
# -----------------------------------------------------------------------------
TEACHER_MODEL_ID = "../Qwen3.8-27B"
DATASET_ID = "bespokelabs/Bespoke-Stratos-17k"
LATENT_DIM = 256

def main():
    model_name = TEACHER_MODEL_ID
    print(f"Loading teacher model: {model_name}...")
    model, tokenizer = load(model_name)
    core_model = model.language_model.model if hasattr(model, "language_model") else model.model
    vocab_size = core_model.embed_tokens.weight.shape[0]
    total_layers = len(core_model.layers)
    
    # Run a dummy pass to get the exact hidden dimension (bypassing 4-bit config quirks)
    dummy_input = mx.array([[0]])
    dummy_h = core_model(dummy_input)
    embed_dim = dummy_h.shape[-1]
    
    print(f"Teacher detected: vocab={vocab_size}, embed_dim={embed_dim}, layers={total_layers}")
    print(f"Active memory after model load: {mx.get_active_memory() / 1e9:.2f} GB")
    
    target_layers = [
        int(total_layers * 0.25),
        int(total_layers * 0.50),
        int(total_layers * 0.75),
        total_layers - 1
    ]
    print(f"Intercepting macro-stages at layers: {target_layers}")
    
    intercepted_states = {}
    
    class HookedLayer(nn.Module):
        def __init__(self, original_layer, idx):
            super().__init__()
            self.layer = original_layer
            self.idx = idx
            
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                if name == "layer":
                    raise
                return getattr(self.layer, name)
            
        def __call__(self, x, *args, **kwargs):
            intercepted_states[self.idx] = x
            return self.layer(x, *args, **kwargs)
            
    for i in target_layers:
        core_model.layers[i] = HookedLayer(core_model.layers[i], i)
        
    C = mx.zeros((embed_dim, embed_dim))
    
    print("Loading Bespoke-Stratos-17k dataset...")
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train", streaming=True)
    
    print("Processing first 100 prompts for variance calibration...")
    prompt_count = 0
    
    for row in dataset:
        if prompt_count >= 100:
            break
        
        # FIX: Dataset schema is {"system": str, "conversations": [{"from":..., "value":...}]}
        system_prompt = row.get("system", "")
        conversations = row.get("conversations", [])
        user_prompt = ""
        for turn in conversations:
            if turn.get("from") == "user":
                user_prompt = turn.get("value", "")
                break  # Take only the first user turn
        
        if not user_prompt:
            continue  # Skip rows with no user prompt
        
        # Apply chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(text)
        
        input_ids = mx.array([tokens])
        
        intercepted_states.clear()
        _ = model(input_ids)
        
        for i in target_layers:
            h = intercepted_states[i]
            # h shape: (1, seq_len, embed_dim)
            # Cast to float32 to prevent float16 overflow during covariance accumulation
            h = h.astype(mx.float32).reshape(-1, embed_dim)
            # Accumulate covariance: h.T @ h
            C = C + (h.T @ h)
            
        # Critical MLX Invariant: evaluate C to prevent OOM
        mx.eval(C)
        intercepted_states.clear()
        gc.collect()
        
        prompt_count += 1
        if prompt_count % 10 == 0:
            print(f"Processed {prompt_count}/100 prompts. Active memory: {mx.get_active_memory() / 1e9:.2f} GB")
            
    print("Computing SVD...")
    C_np = np.array(C)
    eigenvalues, eigenvectors = np.linalg.eigh(C_np)
    
    # Extract top 256 eigenvectors (eigh returns ascending order)
    top_indices = np.argsort(eigenvalues)[::-1][:256]
    top_eigenvectors = eigenvectors[:, top_indices] # Shape: (embed_dim, 256)
    
    total_var = np.sum(np.abs(eigenvalues))  # Use abs for numerical safety
    top_var = np.sum(np.abs(eigenvalues[top_indices]))
    variance_retained = top_var / total_var
    print(f"Variance retained by 256D projection: {variance_retained * 100:.2f}%")
    
    # Save meta information for downstream scripts
    np.savez("svd_projection.npz",
             projection_matrix=top_eigenvectors.astype(np.float32),
             vocab_size=vocab_size,
             embed_dim=embed_dim)
    print("Saved svd_projection.npz successfully.")

if __name__ == "__main__":
    main()
