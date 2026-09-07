import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import time
from train_4d_mind import PureLatent4DMind

def test_inference():
    print("Loading tokenizer from Qwen 3.8 27B...")
    _, tokenizer = load("../Qwen3.8-27B")
    
    vocab_size = 248320 
    
    print("Initializing 4D Student Model Architecture...")
    model = PureLatent4DMind(vocab_size)
    
    print("Loading trained 4D weights from student_4d_mind.safetensors...")
    try:
        model.load_weights("student_4d_mind.safetensors")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
        
    mx.eval(model.parameters())
    
    prompt = "Below is a mathematical reasoning problem. Solve it step-by-step:\n\nQuestion: If 3 cats catch 3 mice in 3 minutes, how many cats are needed to catch 100 mice in 100 minutes?\n\nAnswer:"
    
    tokens = mx.array([tokenizer.encode(prompt)])
    
    print("\n--- 4D Latent Inference Started ---\n")
    print(prompt, end="", flush=True)
    
    # Simple stateless autoregressive inference
    max_tokens = 50
    
    for i in range(max_tokens):
        t0 = time.perf_counter()
        
        seq_len = tokens.shape[1]
        causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        
        logits, _, _, _ = model(tokens, targets=None, padding_mask=None, causal_mask=causal_mask)
        next_token_logits = logits[:, -1, :]
        next_token = mx.argmax(next_token_logits, axis=-1).item()
        
        decoded_token = tokenizer.decode([next_token])
        print(decoded_token, end="", flush=True)
        
        tokens = mx.concatenate([tokens, mx.array([[next_token]])], axis=1)
        mx.eval(tokens)
        
        if next_token in [151643, 151645]: 
            break

    print("\n\n--- Inference Complete ---")

if __name__ == "__main__":
    test_inference()
