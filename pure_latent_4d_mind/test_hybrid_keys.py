import mlx.core as mx
try:
    weights = mx.load("hybrid_student_4d.safetensors")
    print("Keys in safetensors:", list(weights.keys())[:10])
except Exception as e:
    print("Failed to load:", e)
