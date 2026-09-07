import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import time
from train_hybrid_4d import Hybrid4DQwen

def test_inference():
    print("Loading tokenizer and Base Model Qwen3.5-4B...")
    base_model, tokenizer = load("../Qwen3.5-4B")
    
    print("Initializing Hybrid 4D Student Architecture...")
    model = Hybrid4DQwen(base_model)
    
    weights_path = "hybrid_student_4d.safetensors"
    print(f"Loading trained Hybrid 4D adapter weights from {weights_path}...")
    try:
        model.load_weights(weights_path, strict=False)
    except Exception as e:
        print(f"Warning/Error loading weights: {e}")
        
    mx.eval(model.parameters())
    
    prompt = "Below is a mathematical reasoning problem. Solve it step-by-step:\n\nQuestion: If 3 cats catch 3 mice in 3 minutes, how many cats are needed to catch 100 mice in 100 minutes?\n\nAnswer:"
    
    tokens = mx.array([tokenizer.encode(prompt)])
    
    print("\n--- Hybrid 4D Inference Started ---\n")
    print(prompt, end="", flush=True)
    
    # Stateless autoregressive inference (O(N^2) because cache=None in Hybrid4DQwen)
    max_tokens = 50
    
    for i in range(max_tokens):
        t0 = time.perf_counter()
        
        logits, _, _, _ = model(tokens, padding_mask=None)
        
        next_token_logits = logits[:, -1, :]
        next_token = mx.argmax(next_token_logits, axis=-1).item()
        
        decoded_token = tokenizer.decode([next_token])
        print(decoded_token, end="", flush=True)
        
        tokens = mx.concatenate([tokens, mx.array([[next_token]])], axis=1)
        mx.eval(tokens)
        
        if next_token in [tokenizer.eos_token_id, 151643, 151645]: 
            break

    print("\n\n--- Inference Complete ---")

if __name__ == "__main__":
    test_inference()
