import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
import mlx.optimizers as optim
import math
import numpy as np
import pickle
import time
import os
import types

# Import our custom 4D routing components from scratch script
from train_4d_mind import GroupedVectorQuantizer, get_batches, clip_grad_norm

class Hybrid4DQwen(nn.Module):
    def __init__(self, base_model, teacher_dim=256):
        super().__init__()
        self.base = base_model
        
        # We need to extract the underlying sequential transformer model
        if hasattr(self.base, "language_model"):
            self.core = self.base.language_model.model
            self.lm_head = getattr(self.base.language_model, "lm_head", None)
            self.tied_embeddings = getattr(self.base.language_model.args, "tie_word_embeddings", False)
        elif hasattr(self.base, "model"):
            self.core = self.base.model
            self.lm_head = getattr(self.base, "lm_head", None)
            self.tied_embeddings = getattr(self.base.args, "tie_word_embeddings", False)
        else:
            self.core = self.base
            self.lm_head = None
            self.tied_embeddings = False
            
        self.dim = self.core.embed_tokens.weight.shape[1]
        
        # 4D Router with learnable temperature for smoother early training
        self.router = nn.Linear(self.dim, 4)
        self.router_temperature = mx.array([1.0])  # Starts at 1.0, can be annealed
        
        # FiLM Modulators for the 4D coordinate (bounded with tanh)
        self.film_gamma = nn.Linear(4, self.dim)
        self.film_beta = nn.Linear(4, self.dim)
        
        # Zero-initialize so Zone 2 starts as an identity pass
        self.film_gamma.weight = mx.zeros_like(self.film_gamma.weight)
        if self.film_gamma.bias is not None:
            self.film_gamma.bias = mx.zeros_like(self.film_gamma.bias)
        self.film_beta.weight = mx.zeros_like(self.film_beta.weight)
        if self.film_beta.bias is not None:
            self.film_beta.bias = mx.zeros_like(self.film_beta.bias)
        
        # Create a VQ module for each of the 4 target matching points
        self.vqs = [GroupedVectorQuantizer(4, 2048, self.dim) for _ in range(4)]
        
        # Adapter to map Student Dim to Teacher SVD Targets (256D)
        self.adapter = nn.Linear(self.dim, teacher_dim)
        
    def __call__(self, x, padding_mask=None):
        h = self.core.embed_tokens(x)
        
        fa_mask = None
        ssm_mask = None
        if h.shape[1] > 1:
            fa_mask = create_attention_mask(h, None)
            ssm_mask = create_ssm_mask(h, None)
            
        vq_losses = []
        router_losses = []
        extracted_states = []
        
        layers = self.core.layers
        total = len(layers)
        # Zone 2 must be pure linear_attention (GDN) layers to avoid KV-cache
        # issues during recurrent looping. Layers 16-18 are all linear_attention.
        zone_1 = layers[0:16]    # Parse (16 layers, up through the full_attention at 15)
        zone_2 = layers[16:19]   # Recurrent Core (3 pure GDN layers)
        zone_3 = layers[19:total]  # Unembed (remaining layers)
        
        def run_zone(h_in, zone_layers, is_zone_2=False):
            for i, layer in enumerate(zone_layers):
                mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
                if is_zone_2 and i == 0:
                    try:
                        res = layer(h_in, mask=mask, cache=None, state=None)
                    except TypeError:
                        # Qwen layers do not accept 'state', they are stateless when cache=None
                        res = layer(h_in, mask=mask, cache=None)
                else:
                    res = layer(h_in, mask=mask, cache=None)
                h_in = res[0] if isinstance(res, tuple) else res
            return h_in
            
        def process_distill_point(h_in, vq_idx):
            # 4D Routing logic with temperature-scaled sigmoid
            # Clamp temperature to prevent collapse (too low → step function)
            # or drift (too high → uniform random)
            safe_temp = mx.maximum(self.router_temperature, mx.array([0.1]))
            safe_temp = mx.minimum(safe_temp, mx.array([10.0]))
            coord = mx.sigmoid(self.router(h_in) / safe_temp)
            
            # Router Entropy Loss (encourage spread)
            ent = -coord * mx.log(coord + 1e-6) - (1 - coord) * mx.log(1 - coord + 1e-6)
            router_losses.append(mx.sum(ent, axis=-1))
            
            # FiLM Modulation (bounded with tanh to prevent activation explosion)
            gamma = 1.0 + 0.1 * mx.tanh(self.film_gamma(coord))
            beta = 0.1 * mx.tanh(self.film_beta(coord))
            h_modulated = h_in * gamma + beta
            
            # Vector Quantization
            vq_module = self.vqs[vq_idx]
            z_q, quantized, _ = vq_module(h_modulated)
            
            # VQ Commitment Loss
            v_loss = mx.mean(
                mx.square(mx.stop_gradient(quantized) - h_modulated) +
                0.25 * mx.square(quantized - mx.stop_gradient(h_modulated)),
                axis=-1
            )
            vq_losses.append(v_loss)
            
            # Save adapted state for distillation against Teacher targets
            h_256 = self.adapter(z_q)
            extracted_states.append(h_256)
            
            return z_q
            
        # --- PASS 1: Zone 1 (Parse) ---
        # Base model is frozen, no gradients should flow into it
        h = run_zone(mx.stop_gradient(h), zone_1)
        h_anchor = h # Save the output of Zone 1 to anchor the loop
        
        # We match our 4 extraction targets at specific loop iterations
        target_mapping = {1: 0, 3: 1, 5: 2, 6: 3}
        
        # --- PASS 2: Zone 2 (Loop 6 times) ---
        for loop_idx in range(1, 7):
            # Anchor Injection
            h = h + h_anchor
            
            # Modulate, Quantize, and Extract State only at specific targets
            if loop_idx in target_mapping:
                vq_idx = target_mapping[loop_idx]
                h = process_distill_point(h, vq_idx=vq_idx)
                
            # Execute Recurrent Block (exactly 3 layers, layers[16:19])
            # Prevent backprop through frozen base model custom kernels (RoPE/Attention)
            h = run_zone(mx.stop_gradient(h), zone_2, is_zone_2=True)
            
        # --- PASS 3: Zone 3 (Unembed) ---
        h = run_zone(mx.stop_gradient(h), zone_3)
        
        h_final = self.core.norm(h)
        
        if self.lm_head is not None:
            logits = self.lm_head(h_final)
        elif self.tied_embeddings:
            logits = self.core.embed_tokens.as_linear(h_final)
        else:
            logits = None
            
        return logits, vq_losses, router_losses, extracted_states

def multi_task_loss(model, x, targets, y, padding_mask):
    logits, vq_losses, router_losses, extracted_states = model(x, padding_mask=padding_mask)
    
    # 1. Distillation Loss (N-to-1 Mapping to 6 SVD targets)
    traj_losses = []
    # targets shape: (6, B, L, 256)
    for i in range(len(extracted_states)):
        h_256 = extracted_states[i]
        target = targets[i]
        
        # Mean Squared Error
        mse = mx.mean(mx.square(h_256 - target), axis=-1)
        
        # Cosine Distance
        dot = mx.sum(h_256 * target, axis=-1)
        norm_s = mx.sqrt(mx.sum(mx.square(h_256), axis=-1) + 1e-6)
        norm_t = mx.sqrt(mx.sum(mx.square(target), axis=-1) + 1e-6)
        cos_sim = dot / (norm_s * norm_t)
        cosine_loss = 1.0 - cos_sim
        
        traj_loss_val = mse + cosine_loss
        if padding_mask is not None:
            traj_loss_val = traj_loss_val * padding_mask
            traj_losses.append(mx.sum(traj_loss_val) / (mx.sum(padding_mask) + 1e-6))
        else:
            traj_losses.append(mx.mean(traj_loss_val))
            
    # 2. VQ Commitment Loss
    vq_loss_list = []
    for vl in vq_losses:
        if padding_mask is not None:
            vl = vl * padding_mask
            vq_loss_list.append(mx.sum(vl) / (mx.sum(padding_mask) + 1e-6))
        else:
            vq_loss_list.append(mx.mean(vl))
            
    # 3. Router Entropy Loss
    router_loss_list = []
    for rl in router_losses:
        if padding_mask is not None:
            rl = rl * padding_mask
            router_loss_list.append(mx.sum(rl) / (mx.sum(padding_mask) + 1e-6))
        else:
            router_loss_list.append(mx.mean(rl))

    total_traj = sum(traj_losses) / len(traj_losses)
    total_vq = sum(vq_loss_list) / len(vq_loss_list)
    total_router = sum(router_loss_list) / len(router_loss_list)
    
    # Cross-Entropy for final language modeling
    if padding_mask is not None and logits is not None:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_y = y.reshape(-1)
        ce = nn.losses.cross_entropy(flat_logits, flat_y)
        ce = ce.reshape(y.shape) * padding_mask
        masked_ce = mx.sum(ce) / (mx.sum(padding_mask) + 1e-6)
    else:
        masked_ce = mx.array(0.0)

    # Hybrid Loss (High weight on trajectory to force 4D reasoning)
    loss = masked_ce + 5.0 * total_traj + 1.0 * total_vq + total_router
    return loss, (masked_ce, total_traj, total_vq, total_router)

def main():
    import glob
    import re
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-chunk", type=int, default=0, help="Chunk index to resume from")
    args = parser.parse_args()
    
    print("Loading Base Student Model Qwen3.5-4B...")
    base_model, tokenizer = load("../Qwen3.5-4B")
    
    base_model.freeze()
    if hasattr(base_model, "lm_head"):
        base_model.lm_head.unfreeze()
    
    print("Initializing Hybrid 4D Distillation Architecture...")
    model = Hybrid4DQwen(base_model)
    mx.eval(model.parameters())
    
    # Unfreeze 4D modules
    model.router.unfreeze()
    model.film_gamma.unfreeze()
    model.film_beta.unfreeze()
    for vq in model.vqs:
        vq.unfreeze()
    model.adapter.unfreeze()
    
    if args.start_chunk > 0 and os.path.exists("hybrid_student_4d.safetensors"):
        print(f"Resuming from chunk {args.start_chunk}! Loading hybrid_student_4d.safetensors...")
        model.load_weights("hybrid_student_4d.safetensors", strict=False)
    
    param_count = sum(v.size for _, v in mlx.utils.tree_flatten(model.parameters()))
    trainable_count = sum(v.size for _, v in mlx.utils.tree_flatten(model.trainable_parameters()))
    print(f"Total Parameters: {param_count / 1e6:.2f}M | Trainable: {trainable_count / 1e6:.2f}M")
    
    optimizer = optim.AdamW(learning_rate=3e-4)
    loss_and_grad_fn = nn.value_and_grad(model, multi_task_loss)
    
    batch_size = 2
    
    chunk_pattern = "teacher_256D_targets_chunk_*.pkl"
    chunk_files_all = glob.glob(chunk_pattern)
    def get_chunk_idx(f):
        m = re.search(r"chunk_(\d+)\.pkl", f)
        return int(m.group(1)) if m else -1
    
    available_chunks = sorted([f for f in chunk_files_all if get_chunk_idx(f) >= 0], key=get_chunk_idx)
    
    # Filter to start_chunk
    available_chunks = [f for f in available_chunks if get_chunk_idx(f) >= args.start_chunk]
    
    if not available_chunks:
        print(f"Error: No teacher target chunk files found for start_chunk {args.start_chunk}.")
        return
        
    print(f"Starting Hybrid 4D Training over {len(available_chunks)} chunks...")
    
    for chunk_file in available_chunks:
        print(f"\n{'#'*110}")
        print(f"### BEGINNING TRAINING ON CHUNK {get_chunk_idx(chunk_file)} ({chunk_file})")
        print(f"{'#'*110}")
        
        step = 0
        for x, y, padding_mask, targets, _ in get_batches(chunk_file, batch_size=batch_size):
            t0 = time.perf_counter()
            
            # targets are yielded as [T1, T2, T3, T4] which are (B, L, 256)
            # We need to scale them by 0.01
            targets = mx.stack(targets) * 0.01
            
            (loss_val, (ce, traj, vq_l, router_l)), grads = loss_and_grad_fn(model, x, targets, y, padding_mask)
            
            grads, grad_norm = clip_grad_norm(grads, max_norm=1.0)
            
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            t1 = time.perf_counter()
            if step % 25 == 0:
                print(f"[Chunk {get_chunk_idx(chunk_file):02d}] Step {step:04d} | "
                      f"Loss: {loss_val.item():.4f} (CE: {ce.item():.4f}, Traj: {traj.item():.4f}, VQ: {vq_l.item():.4f}, Router: {router_l.item():.4f}) | "
                      f"GradNorm: {grad_norm:.2f} | Time: {t1-t0:.3f}s", flush=True)
            step += 1
            
        print(f"Completed Chunk {get_chunk_idx(chunk_file)}")
        
        # Save intermediate weights
        mx.eval(model.trainable_parameters())
        trainable_weights = dict(mlx.utils.tree_flatten(model.trainable_parameters()))
        mx.save_safetensors(f"hybrid_student_4d_chunk_{get_chunk_idx(chunk_file)}.safetensors", trainable_weights)
        mx.save_safetensors("hybrid_student_4d.safetensors", trainable_weights)
        
    print("Training complete! Adapter weights saved.")

if __name__ == "__main__":
    main()
