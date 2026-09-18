import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from datasets import load_dataset
import numpy as np
import gc
import argparse

def main():
    # These were previously hardcoded to a mid-run resume point (skip 13000 / chunk 13),
    # which silently skipped the first 13k examples for anyone following the README.
    parser = argparse.ArgumentParser(description="JIT teacher trajectory extractor")
    parser.add_argument("--skip", type=int, default=0, help="Examples to skip (resume point)")
    parser.add_argument("--chunk-idx", type=int, default=0, help="Chunk index to start numbering from")
    parser.add_argument("--max-prompts", type=int, default=17000, help="Stop after this many examples")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Examples per output chunk")
    args = parser.parse_args()

    print("Loading projection matrix...")
    try:
        data = np.load("svd_projection.npz")
        W_proj_np = data["projection_matrix"]
        vocab_size = data["vocab_size"].item()
        embed_dim = data["embed_dim"].item()
        W_proj = mx.array(W_proj_np.astype(np.float32))
    except FileNotFoundError:
        print("Error: svd_projection.npz not found. Run compute_svd.py first.")
        return
    
    # -----------------------------------------------------------------------------
    # Configuration & Hyperparameters
    # -----------------------------------------------------------------------------
    model_name = "../Qwen3.8-27B"
    print(f"Loading teacher model: {model_name}...")
    model, tokenizer = load(model_name)
    
    core_model = model.language_model.model if hasattr(model, "language_model") else model.model
    vocab_size = core_model.embed_tokens.weight.shape[0]
    total_layers = len(core_model.layers)
    
    target_layers = [
        int(total_layers * 0.25),
        int(total_layers * 0.50),
        int(total_layers * 0.75),
        total_layers - 1
    ]
    print(f"Intercepting macro-stages at layers: {target_layers}")
    print(f"Active memory after model load: {mx.get_active_memory() / 1e9:.2f} GB")
    
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
        
    extracted_targets = {i: [] for i in target_layers}
    saved_tokens = []
    
    def flush_chunk():
        nonlocal extracted_targets, saved_tokens, chunk_idx
        import pickle
        print(f"Saving extracted targets chunk {chunk_idx}...")
        output_data = {
            "layer_1": extracted_targets[target_layers[0]],
            "layer_2": extracted_targets[target_layers[1]],
            "layer_3": extracted_targets[target_layers[2]],
            "layer_4": extracted_targets[target_layers[3]],
            "tokens": saved_tokens,
            "vocab_size": vocab_size,
            "embed_dim": embed_dim
        }
        with open(f"teacher_256D_targets_chunk_{chunk_idx}.pkl", "wb") as f:
            pickle.dump(output_data, f)
        # Clear memory for next chunk
        extracted_targets = {i: [] for i in target_layers}
        saved_tokens = []
        chunk_idx += 1
        gc.collect()
    
    max_prompts = args.max_prompts
    chunk_size = args.chunk_size
    chunk_idx = args.chunk_idx
    
    print(f"Loading Bespoke-Stratos-17k dataset...")
    dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train", streaming=True)
    if args.skip > 0:
        dataset = dataset.skip(args.skip)
    
    print(f"Processing up to {max_prompts} prompts for JIT target extraction in chunks of {chunk_size}...")
    prompt_count = args.skip
    
    for row in dataset:
        if prompt_count >= max_prompts:
            break
        
        system_prompt = row.get("system", "")
        conversations = row.get("conversations", [])
        user_prompt = ""
        for turn in conversations:
            if turn.get("from") == "user":
                user_prompt = turn.get("value", "")
                break  # Take only the first user turn
        
        if not user_prompt:
            continue  # Skip rows with no user prompt
            
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
            h = intercepted_states[i] # (1, seq_len, embed_dim)
            h = h.astype(mx.float16)
            h_proj = mx.matmul(h, W_proj)
            mx.eval(h_proj)
            extracted_targets[i].append(np.array(h_proj))
            
        saved_tokens.append(np.array(tokens))
        intercepted_states.clear()
        
        prompt_count += 1
        if prompt_count % 50 == 0:
            gc.collect()
            print(f"Extracted {prompt_count}/{max_prompts}. Active memory: {mx.get_active_memory() / 1e9:.2f} GB")
            
        # Save chunk and clear RAM
        if prompt_count % chunk_size == 0 or prompt_count == max_prompts:
            flush_chunk()
    
    # Bespoke-Stratos-17k holds ~16.7k usable rows, so the loop above exhausts the
    # dataset before prompt_count reaches a chunk_size multiple. Without this final
    # flush the trailing partial chunk was silently discarded.
    if saved_tokens:
        print(f"Flushing final partial chunk ({len(saved_tokens)} examples)...")
        flush_chunk()
             
    print(f"Extraction complete. Saved {prompt_count} prompts. Process terminates to release all RAM.")

if __name__ == "__main__":
    main()
