import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import json

def inspect_model():
    print("Loading model...")
    model, tokenizer = load("../Qwen3.5-4B")
    
    print("Model Class:", type(model))
    print("Has language_model?", hasattr(model, "language_model"))
    
    if hasattr(model, "language_model"):
        core = model.language_model.model
    elif hasattr(model, "model"):
        core = model.model
    else:
        core = model
        
    print("Core Class:", type(core))
    print("Has layers?", hasattr(core, "layers"))
    if hasattr(core, "layers"):
        print("Number of layers:", len(core.layers))
        print("First layer class:", type(core.layers[0]))
        
        # Check components of the first layer
        layer = core.layers[0]
        print("Layer attributes:", [k for k in layer.__dict__.keys() if not k.startswith('_')])
        
    print("Done.")

if __name__ == "__main__":
    inspect_model()
