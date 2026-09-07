import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from train_hybrid_4d import Hybrid4DQwen
import time

def main():
    print("Loading Base Student Model Qwen3.5-4B...")
    base_model, tokenizer = load("../Qwen3.5-4B")
    
    print("Initializing Hybrid 4D Recurrent Architecture...")
    model = Hybrid4DQwen(base_model)
    mx.eval(model.parameters())
    
    print("Testing forward pass with dummy tensor...")
    x = mx.array([[101, 102, 103, 104, 105]]) # Batch size 1, seq len 5
    
    t0 = time.perf_counter()
    logits, vq_losses, router_losses, extracted_states = model(x, padding_mask=None)
    mx.eval(logits)
    t1 = time.perf_counter()
    
    print(f"Forward pass completed in {t1 - t0:.3f}s")
    print(f"Logits shape: {logits.shape}")
    print(f"Number of VQ losses: {len(vq_losses)}")
    print(f"Number of router losses: {len(router_losses)}")
    print(f"Number of extracted states: {len(extracted_states)}")
    for i, state in enumerate(extracted_states):
        print(f"State {i+1} shape: {state.shape}")
        
    print("Test passed successfully!")

if __name__ == "__main__":
    main()
