import pickle
import numpy as np
with open("teacher_256D_targets_chunk_10.pkl", "rb") as f:
    data = pickle.load(f)

tokens = data["tokens"]
print("Max len:", max(len(t) for t in tokens))
print("Min len:", min(len(t) for t in tokens))

empty = [i for i, t in enumerate(tokens) if len(t) <= 1]
print("Empty or len<=1 samples:", empty)

for i, t in enumerate(tokens):
    if len(t) > 0 and max(t) >= 151936:
        print(f"Sample {i} has out of bounds token: {max(t)}")
        
