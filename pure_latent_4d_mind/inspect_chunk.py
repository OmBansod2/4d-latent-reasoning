import pickle
import sys
import numpy as np

# Take the bound from the file itself rather than hardcoding 151936, which is the
# Qwen3 vocabulary. Both the teacher and the student here are qwen3_5 (vocab 248320),
# so a hardcoded 151936 reports tens of thousands of perfectly valid tokens as corrupt.
path = sys.argv[1] if len(sys.argv) > 1 else "teacher_256D_targets_chunk_10.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

tokens = data["tokens"]
vocab_size = data["vocab_size"]
print("Vocab size recorded in file:", vocab_size)
print("Max len:", max(len(t) for t in tokens))
print("Min len:", min(len(t) for t in tokens))

empty = [i for i, t in enumerate(tokens) if len(t) <= 1]
print("Empty or len<=1 samples:", empty)

for i, t in enumerate(tokens):
    if len(t) > 0 and max(t) >= vocab_size:
        print(f"Sample {i} has out of bounds token: {max(t)} (vocab {vocab_size})")
        
