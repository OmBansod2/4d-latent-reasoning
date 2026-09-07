import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import gc
import mlx.utils

# ─── FiLM-Modulated SwiGLU FFN ───────────────────────────────────────────────
class FiLMSwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.film_gamma = nn.Linear(4, dim)
        self.film_beta = nn.Linear(4, dim)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.inner_norm = nn.RMSNorm(hidden_dim)
        
    def __call__(self, x, coord):
        # FIX: Bound gamma with tanh to strictly prevent recurrent explosion in BPTT
        gamma = 1.0 + 0.1 * mx.tanh(self.film_gamma(coord))
        beta = 0.1 * mx.tanh(self.film_beta(coord))
        x = x * gamma + beta
        # SwiGLU: w2(norm(silu(w1(x)) * w3(x))) prevents quadratic explosion
        inner = nn.silu(self.w1(x)) * self.w3(x)
        return self.w2(self.inner_norm(inner))


# ─── 4-Head Grouped Vector Quantizer with EMA + Dead Code Replacement ────────
class GroupedVectorQuantizer(nn.Module):
    """4-head Grouped VQ with EMA codebook updates (decay=0.99).
    Prevents codebook collapse during recurrent loops via:
    1. EMA updates instead of direct gradient optimization on codebooks
    2. Dead code replacement — reassigns unused codes to batch mean + noise
    3. Utilization tracking for diagnostics
    Combinatorial space: 2048^4 ≈ 1.76e13 unique states."""
    
    def __init__(self, num_heads, num_embeddings, embedding_dim, ema_decay=0.99):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.num_embeddings = num_embeddings
        self.ema_decay = ema_decay
        
        # Codebook embeddings (H, K, d) — NOT optimized by gradient; updated by EMA
        scale = 1.0 / (self.head_dim ** 0.5)
        self.codebooks = scale * mx.random.normal([num_heads, num_embeddings, self.head_dim])
        
        # EMA tracking buffers (not trainable parameters)
        # Running sum of assigned vectors per code
        self._ema_cluster_sum = mx.zeros([num_heads, num_embeddings, self.head_dim])
        # Running count of assignments per code
        self._ema_cluster_count = mx.ones([num_heads, num_embeddings]) * 1e-5
        # Total calls for dead code detection
        self._total_calls = 0
        # Track utilization for logging
        self._last_utilization = 0.0
        
    def _compute_distances(self, x_flat):
        """Compute L2 distances between input vectors and codebook entries.
        x_flat: (N, H, d) -> distances: (N, H, K)"""
        # Transpose for batched matmul: (H, N, d) @ (H, d, K) -> (H, N, K)
        x_t = mx.transpose(x_flat, (1, 0, 2))       # (H, N, d)
        c_t = mx.transpose(self.codebooks, (0, 2, 1)) # (H, d, K)
        dot = mx.matmul(x_t, c_t)                     # (H, N, K)
        dot = mx.transpose(dot, (1, 0, 2))            # (N, H, K)
        
        x_sq = mx.sum(x_flat ** 2, axis=-1, keepdims=True)  # (N, H, 1)
        c_sq = mx.sum(self.codebooks ** 2, axis=-1)           # (H, K)
        c_sq = mx.expand_dims(c_sq, 0)                        # (1, H, K)
        
        return x_sq + c_sq - 2 * dot  # (N, H, K)
        
    def _ema_update(self, x_flat, encoding_indices):
        """Update codebook entries using Exponential Moving Average.
        Vectorized: uses one-hot encoding + matmul for efficient batch updates."""
        N = x_flat.shape[0]
        
        new_counts = mx.zeros_like(self._ema_cluster_count)   # (H, K)
        new_sums = mx.zeros_like(self._ema_cluster_sum)       # (H, K, d)
        
        for head_idx in range(self.num_heads):
            head_indices = encoding_indices[:, head_idx]  # (N,)
            head_vectors = x_flat[:, head_idx, :]          # (N, d)
            
            # One-hot encode assignments: (N, K)
            one_hot = mx.zeros([N, self.num_embeddings])
            one_hot = one_hot.at[mx.arange(N), head_indices].add(1.0)
            
            # Per-code counts and sums via matmul
            new_counts = new_counts.at[head_idx].add(mx.sum(one_hot, axis=0))
            new_sums = new_sums.at[head_idx].add(mx.matmul(one_hot.T, head_vectors))
        
        # Full EMA decay + new observation
        self._ema_cluster_count = self.ema_decay * self._ema_cluster_count + (1 - self.ema_decay) * new_counts
        self._ema_cluster_sum = self.ema_decay * self._ema_cluster_sum + (1 - self.ema_decay) * new_sums
        
        # Update codebooks: codebook[h,k] = ema_sum[h,k] / ema_count[h,k]
        counts_expanded = mx.expand_dims(self._ema_cluster_count, -1)  # (H, K, 1)
        self.codebooks = self._ema_cluster_sum / (counts_expanded + 1e-6)
        
    def _replace_dead_codes(self, x_flat, encoding_indices):
        """Replace dead codebook entries by sampling random individual vectors
        from the current batch (+ noise), scattering replacements across the
        actual data distribution instead of clumping at the centroid."""
        DEAD_THRESHOLD = 1.0
        
        for head_idx in range(self.num_heads):
            head_counts = self._ema_cluster_count[head_idx]  # (K,)
            dead_mask = head_counts < DEAD_THRESHOLD
            num_dead = int(mx.sum(dead_mask.astype(mx.float32)).item())
            
            if num_dead > 0:
                head_vectors = x_flat[:, head_idx, :]  # (N, d)
                N = head_vectors.shape[0]
                
                # Sample random individual vectors from the batch for each dead code
                # This scatters replacements across the data manifold
                sample_indices = mx.random.randint(0, N, [self.num_embeddings])
                sampled = head_vectors[sample_indices]  # (K, d)
                noise = 0.01 * mx.random.normal([self.num_embeddings, self.head_dim])
                replacement = sampled + noise  # (K, d)
                
                # Only replace where dead
                dead_mask_f = dead_mask.astype(mx.float32)
                dead_mask_expanded = mx.expand_dims(dead_mask_f, -1)  # (K, 1)
                self.codebooks = self.codebooks.at[head_idx].add(
                    dead_mask_expanded * (replacement - self.codebooks[head_idx])
                )
                # Reset EMA counts and sums for replaced codes so subsequent EMA updates preserve them
                self._ema_cluster_count = self._ema_cluster_count.at[head_idx].add(
                    dead_mask_f * (1.0 - self._ema_cluster_count[head_idx])
                )
                self._ema_cluster_sum = self._ema_cluster_sum.at[head_idx].add(
                    dead_mask_expanded * (replacement - self._ema_cluster_sum[head_idx])
                )
        
    def __call__(self, x, training=True):
        B, L, D = x.shape
        # Split into groups: (B, L, num_heads, head_dim)
        x_grouped = x.reshape(B, L, self.num_heads, self.head_dim)
        # Flatten batch and seq: (B*L, num_heads, head_dim)
        x_flat = x_grouped.reshape(-1, self.num_heads, self.head_dim)
        N = x_flat.shape[0]  # B*L
        
        # Compute distances and find nearest codes
        distances = self._compute_distances(x_flat)  # (N, H, K)
        encoding_indices = mx.stop_gradient(mx.argmin(distances, axis=-1))  # (N, H)
        
        # Gather quantized vectors from each head's codebook
        quantized_parts = []
        for head_idx in range(self.num_heads):
            head_indices = encoding_indices[:, head_idx]  # (N,)
            head_codebook = self.codebooks[head_idx]       # (K, d)
            quantized_parts.append(head_codebook[head_indices])  # (N, d)
        
        # Stack and reshape: (N, H, d) -> (B, L, H, d) -> (B, L, D)
        quantized = mx.stack(quantized_parts, axis=1)  # (N, H, d)
        quantized = quantized.reshape(B, L, self.num_heads, self.head_dim)
        quantized_flat = quantized.reshape(B, L, D)
        
        # Straight-Through Estimator (STE) — gradients flow through x
        z_q = x + mx.stop_gradient(quantized_flat - x)
        
        # EMA codebook update (only during training)
        if training:
            self._total_calls += 1
            self._ema_update(mx.stop_gradient(x_flat), encoding_indices)
            
            # Dead code replacement every 100 forward passes
            if self._total_calls % 100 == 0:
                self._replace_dead_codes(mx.stop_gradient(x_flat), encoding_indices)
            
            # Track per-head active codebook utilization (% of codes with active assignments)
            # (Diagnostic metrics removed from inner loop to prevent synchronous MLX graph breaks. Computed in logging loop instead)
        
        return z_q, quantized_flat, encoding_indices


# ─── 4D Topology-Aware Attention ─────────────────────────────────────────────
class TopologyAwareAttention(nn.Module):
    """Self-attention with a learned 4D distance bias that penalizes
    attention between topologically distant tokens."""
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        # FIX: Learned distance scale as a proper Linear(1,1) so MLX tracks it
        # This creates a single trainable scalar via a 1->1 linear (no bias)
        self.distance_proj = nn.Linear(1, 1, bias=False)
        
    def __call__(self, x, coord, causal_mask=None):
        B, L, D = x.shape
        
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # Standard scaled dot-product attention
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, L, L)
        
        # 4D Distance Bias
        # coord: (B, L, 4)
        coord_i = mx.expand_dims(coord, 2)  # (B, L, 1, 4)
        coord_j = mx.expand_dims(coord, 1)  # (B, 1, L, 4)
        diff = coord_i - coord_j             # (B, L, L, 4)
        dist = mx.sqrt(mx.sum(diff ** 2, axis=-1) + 1e-8)  # (B, L, L)
        
        # Apply learned scaling via the distance_proj (acts as a trainable scalar)
        # dist: (B, L, L) -> (B, L, L, 1) -> project -> (B, L, L, 1) -> squeeze
        dist_input = mx.expand_dims(dist, -1)  # (B, L, L, 1)
        bias = -mx.abs(self.distance_proj(dist_input)).squeeze(-1)  # (B, L, L)
        bias = mx.expand_dims(bias, 1)  # (B, 1, L, L) — broadcast over heads
        
        scores = scores + bias
        
        if causal_mask is not None:
            scores = scores + causal_mask
            
        probs = mx.softmax(scores, axis=-1)
        out = (probs @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)


# ─── Pure Latent 4D Mind Student Model ───────────────────────────────────────
class PureLatent4DMind(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = 1024
        self.num_heads = 16
        
        # Tied embeddings (embed_tokens.weight == lm_head.weight)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.dim)
        # FIX: Standard transformer embedding scale to prevent forward-pass explosion
        self.embed_tokens.weight = mx.random.normal([self.vocab_size, self.dim]) * 0.02
        
        # 4D Gumbel-Sigmoid Router
        self.router = nn.Linear(self.dim, 4)
        
        # ─── 2-Layer Recurrent Block ─────────────────────────────────
        # Layer 1
        self.norm1_1 = nn.RMSNorm(self.dim)
        self.attn_1 = TopologyAwareAttention(self.dim, self.num_heads)
        self.attn_1.out_proj.weight = self.attn_1.out_proj.weight * 0.01
        self.norm2_1 = nn.RMSNorm(self.dim)
        self.film_ffn_1 = FiLMSwiGLU(self.dim, 4096)
        self.film_ffn_1.w2.weight = self.film_ffn_1.w2.weight * 0.01
        self.vq_1 = GroupedVectorQuantizer(4, 2048, self.dim)
        
        # Layer 2
        self.norm1_2 = nn.RMSNorm(self.dim)
        self.attn_2 = TopologyAwareAttention(self.dim, self.num_heads)
        self.attn_2.out_proj.weight = self.attn_2.out_proj.weight * 0.01
        self.norm2_2 = nn.RMSNorm(self.dim)
        self.film_ffn_2 = FiLMSwiGLU(self.dim, 4096)
        self.film_ffn_2.w2.weight = self.film_ffn_2.w2.weight * 0.01
        self.vq_2 = GroupedVectorQuantizer(4, 2048, self.dim)
        
        # Detachable Adapter to bridge student 1024D -> 256D teacher targets
        self.adapter = nn.Linear(self.dim, 256)
        
        # Final norm before LM head to ensure bounded logits
        self.final_norm = nn.RMSNorm(self.dim)
        
    def __call__(self, x, targets=None, padding_mask=None, causal_mask=None):
        # x: (B, seq_len) token IDs
        h_emb = self.embed_tokens(x)
        h = h_emb
        
        traj_losses = []
        vq_losses = []
        router_losses = []
        
        for loop_idx in range(1, 7):  # Exactly 6 BPTT loops
            # Residual injection of embedding signal
            h = h + h_emb
            
            # 4D Router: continuous coordinate via Sigmoid
            coord = mx.sigmoid(self.router(h))  # (B, L, 4)
            
            # Router Entropy Loss: maximize variance to prevent dead vertices
            # Binary entropy: -p*log(p) - (1-p)*log(1-p)
            ent = -coord * mx.log(coord + 1e-6) - (1 - coord) * mx.log(1 - coord + 1e-6)
            ent = mx.sum(ent, axis=-1)  # (B, L)
            if padding_mask is not None:
                ent = ent * padding_mask
                router_losses.append(-mx.sum(ent) / (mx.sum(padding_mask) + 1e-6))
            else:
                router_losses.append(-mx.mean(ent))
            
            # ─── Layer 1 ─────────────────────────────────────────────
            h_norm = self.norm1_1(h)
            h = h + self.attn_1(h_norm, coord, causal_mask)
            h_norm2 = self.norm2_1(h)
            h = h + self.film_ffn_1(h_norm2, coord)
            z_q, quantized, _ = self.vq_1(h)
            
            # VQ Commitment Loss for Layer 1
            vq1_loss = mx.mean(
                mx.square(mx.stop_gradient(quantized) - h) +
                0.25 * mx.square(quantized - mx.stop_gradient(h)),
                axis=-1
            )
            if padding_mask is not None:
                vq1_loss = vq1_loss * padding_mask
                vq_losses.append(mx.sum(vq1_loss) / (mx.sum(padding_mask) + 1e-6))
            else:
                vq_losses.append(mx.mean(vq1_loss))
            h = z_q
            
            # ─── Layer 2 ─────────────────────────────────────────────
            h_norm_2 = self.norm1_2(h)
            h = h + self.attn_2(h_norm_2, coord, causal_mask)
            h_norm2_2 = self.norm2_2(h)
            h = h + self.film_ffn_2(h_norm2_2, coord)
            z_q2, quantized2, _ = self.vq_2(h)
            
            # VQ Commitment Loss for Layer 2
            vq2_loss = mx.mean(
                mx.square(mx.stop_gradient(quantized2) - h) +
                0.25 * mx.square(quantized2 - mx.stop_gradient(h)),
                axis=-1
            )
            if padding_mask is not None:
                vq2_loss = vq2_loss * padding_mask
                vq_losses.append(mx.sum(vq2_loss) / (mx.sum(padding_mask) + 1e-6))
            else:
                vq_losses.append(mx.mean(vq2_loss))
            h = z_q2
            
            # ─── Target Matching at loops [2, 4, 5, 6] ──────────────
            if targets is not None and loop_idx in [2, 4, 5, 6]:
                target_map = {2: 0, 4: 1, 5: 2, 6: 3}
                target_idx = target_map[loop_idx]
                # FIX: Scale massive SVD targets down by 100x to prevent BPTT gradient explosions
                target = targets[target_idx] * 0.01
                
                h_256 = self.adapter(h)  # (B, L, 256)
                
                # Pure MSE trajectory loss (MSE on 256D student adapter vs projected teacher state)
                mse = mx.mean(mx.square(h_256 - target), axis=-1)
                traj_loss_val = mse
                if padding_mask is not None:
                    traj_loss_val = traj_loss_val * padding_mask
                    traj_losses.append(mx.sum(traj_loss_val) / (mx.sum(padding_mask) + 1e-6))
                else:
                    traj_losses.append(mx.mean(traj_loss_val))
                    
        # LM Head with tied weights
        h_final = self.final_norm(h)
        logits = mx.matmul(h_final, self.embed_tokens.weight.T)
        
        return logits, traj_losses, vq_losses, router_losses


# ─── Multi-Task Loss Function ────────────────────────────────────────────────
def multi_task_loss(model, x, targets, y, padding_mask, causal_mask):
    logits, traj_losses, vq_losses, router_losses = model(x, targets, padding_mask, causal_mask)
    
    # Masked Cross-Entropy
    ce_loss = nn.losses.cross_entropy(logits, y)
    if padding_mask is not None:
        ce_loss = ce_loss * padding_mask
        masked_ce = mx.sum(ce_loss) / (mx.sum(padding_mask) + 1e-6)
    else:
        masked_ce = mx.mean(ce_loss)
        
    total_traj = sum(traj_losses) / len(traj_losses) if traj_losses else mx.array(0.0)
    total_vq = sum(vq_losses) / len(vq_losses) if vq_losses else mx.array(0.0)
    total_router = sum(router_losses) / len(router_losses) if router_losses else mx.array(0.0)
    
    # Calibrated Multi-Task Loss
    loss = masked_ce + 5.0 * total_traj + 1.0 * total_vq + total_router
    return loss, (masked_ce, total_traj, total_vq, total_router)


# ─── Data Loading ────────────────────────────────────────────────────────────
def get_batches(pkl_file, batch_size=2):
    """Load teacher targets from .pkl and yield padded batches.
    NOTE: Loads entire file into RAM (~few GB for 1000 prompts). Acceptable for this scale."""
    import pickle
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    tokens = data["tokens"]
    vocab_size = data["vocab_size"]
    l1, l2, l3, l4 = data["layer_1"], data["layer_2"], data["layer_3"], data["layer_4"]
    
    num_samples = len(tokens)
    
    # Filter out corrupted sequences to prevent out-of-bounds MLX embedding crashes
    valid_indices = []
    for i in range(num_samples):
        if len(tokens[i]) > 1 and max(tokens[i]) < vocab_size:
            valid_indices.append(i)
            
    if len(valid_indices) < num_samples:
        print(f"Filtered {num_samples - len(valid_indices)} corrupted sequences from {pkl_file}")
        
    indices = np.array(valid_indices)
    num_samples = len(indices)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        batch_idx = indices[start_idx:start_idx + batch_size]
        if len(batch_idx) < batch_size:
            continue  # Skip incomplete final batch
        
        batch_tokens = [tokens[i] for i in batch_idx]
        batch_l1 = [l1[i][0] for i in batch_idx]
        batch_l2 = [l2[i][0] for i in batch_idx]
        batch_l3 = [l3[i][0] for i in batch_idx]
        batch_l4 = [l4[i][0] for i in batch_idx]
        
        max_len = max(len(t) for t in batch_tokens)
        seq_len = max_len - 1  # Input is tokens[:-1], target is tokens[1:]
        
        x_padded, y_padded, mask_padded = [], [], []
        T1_padded, T2_padded, T3_padded, T4_padded = [], [], [], []
        
        for i in range(len(batch_tokens)):
            toks = batch_tokens[i]
            x_i = toks[:-1]
            y_i = toks[1:]
            length = len(x_i)
            
            x_padded.append(np.pad(x_i, (0, seq_len - length)))
            y_padded.append(np.pad(y_i, (0, seq_len - length)))
            mask_padded.append(np.pad(np.ones(length), (0, seq_len - length)))
            
            # Teacher targets: align with input tokens (drop last to match seq_len)
            T1_padded.append(np.pad(batch_l1[i][:-1], ((0, seq_len - length), (0, 0))))
            T2_padded.append(np.pad(batch_l2[i][:-1], ((0, seq_len - length), (0, 0))))
            T3_padded.append(np.pad(batch_l3[i][:-1], ((0, seq_len - length), (0, 0))))
            T4_padded.append(np.pad(batch_l4[i][:-1], ((0, seq_len - length), (0, 0))))
            
        yield (
            mx.array(np.stack(x_padded).astype(np.int32)),
            mx.array(np.stack(y_padded).astype(np.int32)),
            mx.array(np.stack(mask_padded).astype(np.float32)),
            [
                mx.array(np.stack(T1_padded).astype(np.float32)),
                mx.array(np.stack(T2_padded).astype(np.float32)),
                mx.array(np.stack(T3_padded).astype(np.float32)),
                mx.array(np.stack(T4_padded).astype(np.float32)),
            ],
            vocab_size,
        )


# ─── Gradient Clipping Utility ───────────────────────────────────────────────
def clip_grad_norm(grads, max_norm=1.0):
    """Clip gradient norms to prevent explosions during 6-loop BPTT."""
    # Pre-filter all non-finite values to strictly 0.0 to prevent any float leaks
    grads = mlx.utils.tree_map(
        lambda g: mx.where(mx.isfinite(g), g, mx.array(0.0)),
        grads
    )
    flat_grads = mlx.utils.tree_flatten(grads)
    total_norm_sq = sum(mx.sum(g ** 2).item() for _, g in flat_grads)
    total_norm = total_norm_sq ** 0.5
    
    if total_norm > max_norm or not np.isfinite(total_norm):
        scale = max_norm / (total_norm + 1e-6)
        if not np.isfinite(scale):
            scale = 0.0
        grads = mlx.utils.tree_map(lambda g: g * scale, grads)
    return grads, total_norm


# ─── Main Training Loop ─────────────────────────────────────────────────────
def main():
    import sys, os, glob, re, argparse
    
    parser = argparse.ArgumentParser(description="Pure Latent 4D Mind Multi-Chunk Training")
    parser.add_argument("--start-chunk", type=int, default=1, help="First chunk index to train (default: 1)")
    parser.add_argument("--end-chunk", type=int, default=15, help="Last chunk index to train (default: 15)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    args = parser.parse_args()

    # Discover and sort available chunks
    chunk_pattern = "teacher_256D_targets_chunk_*.pkl"
    chunk_files_all = glob.glob(chunk_pattern)
    def get_chunk_idx(f):
        m = re.search(r"chunk_(\d+)\.pkl", f)
        return int(m.group(1)) if m else -1
    
    available_chunks = sorted([f for f in chunk_files_all if get_chunk_idx(f) >= 0], key=get_chunk_idx)
    if not available_chunks:
        # Fallback to single file if named teacher_256D_targets.pkl
        if os.path.exists("teacher_256D_targets.pkl"):
            available_chunks = ["teacher_256D_targets.pkl"]
        else:
            print("Error: No teacher target chunk files found.")
            return

    # Filter chunks based on start and end
    selected_chunks = [
        f for f in available_chunks
        if args.start_chunk <= get_chunk_idx(f) <= args.end_chunk
    ]
    if not selected_chunks and len(available_chunks) > 0 and "teacher_256D_targets.pkl" not in available_chunks[0]:
        print(f"No chunks match range [{args.start_chunk}, {args.end_chunk}]. Available: {[get_chunk_idx(f) for f in available_chunks]}")
        return

    print("=" * 110)
    print(f"PURE LATENT 4D MIND: MULTI-CHUNK TRAINING PIPELINE")
    print(f"Chunks to train: {[get_chunk_idx(f) for f in selected_chunks]} (Total: {len(selected_chunks)} chunks)")
    print(f"Batch size: {args.batch_size} | Learning rate: {args.lr}")
    print("=" * 110)

    # Initialize model using metadata from first available chunk
    import pickle
    first_chunk = selected_chunks[0]
    with open(first_chunk, "rb") as f:
        meta_data = pickle.load(f)
    vocab_size = meta_data["vocab_size"]
    del meta_data

    print(f"Initializing 4D Latent Student Model (dynamic vocab={vocab_size})...")
    model = PureLatent4DMind(vocab_size)
    mx.eval(model.parameters())

    # Resume from existing checkpoint if available
    checkpoint_file = "student_4d_mind.safetensors"
    if os.path.exists(checkpoint_file):
        print(f"Loading existing checkpoint from {checkpoint_file}...")
        try:
            model.load_weights(checkpoint_file)
            mx.eval(model.parameters())
            print("✓ Resumed successfully from checkpoint!")
        except Exception as e:
            print(f"Warning: Could not load checkpoint: {e}")

    num_params = sum(x.size for _, x in mlx.utils.tree_flatten(model.parameters()))
    print(f"Student parameter count: {num_params / 1e6:.2f}M")
    print(f"Active memory after init: {mx.get_active_memory() / 1e9:.2f} GB")

    optimizer = optim.AdamW(learning_rate=args.lr)
    loss_and_grad_fn = nn.value_and_grad(model, multi_task_loss)

    overall_step = 0
    t_pipeline_start = time.perf_counter()

    for chunk_file in selected_chunks:
        c_idx = get_chunk_idx(chunk_file)
        print("\n" + "#" * 110)
        print(f"### BEGINNING TRAINING ON CHUNK {c_idx} ({chunk_file})")
        print("#" * 110)

        chunk_step = 0
        chunk_loss = 0.0
        t_chunk_start = time.perf_counter()

        for x, y, padding_mask, targets, _ in get_batches(chunk_file, batch_size=args.batch_size):
            seq_len = x.shape[1]
            causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)

            t0 = time.perf_counter()

            # Forward + Backward
            (loss, (ce, traj, vq, router)), grads = loss_and_grad_fn(
                model, x, targets, y, padding_mask, causal_mask
            )

            # Periodic Equilibrium Check: Monitor relative gradient magnitudes
            check_equilibrium = (chunk_step % 25 == 0 or chunk_step == 0)
            t_gnorm, v_gnorm, eq_ratio = 0.0, 0.0, 0.0
            if check_equilibrium:
                # Removed extra value_and_grad passes to prevent OOM Kernel Panics
                # Only use the overall gradient norm which is already computed
                pass

            # Gradient clipping to prevent BPTT explosions
            grads, grad_norm = clip_grad_norm(grads, max_norm=1.0)

            optimizer.update(model, grads)

            # Critical MLX Invariant: evaluate everything to flush the lazy graph
            mx.eval(
                model.parameters(), 
                optimizer.state, 
                loss, 
                grads,
                model.vq_1._ema_cluster_sum,
                model.vq_1._ema_cluster_count,
                model.vq_2._ema_cluster_sum,
                model.vq_2._ema_cluster_count
            )

            t1 = time.perf_counter()

            loss_val = loss.item()
            chunk_loss += loss_val
            chunk_step += 1
            overall_step += 1

            if check_equilibrium:
                def get_util(vq):
                    active = mx.sum(vq._ema_cluster_count >= 0.1, axis=-1)
                    return [float(active[h].item()) / vq.num_embeddings * 100.0 for h in range(vq.num_heads)]
                
                vq1_util = get_util(model.vq_1)
                vq2_util = get_util(model.vq_2)
                h_utils = [(u1 + u2) / 2.0 for u1, u2 in zip(vq1_util, vq2_util)]
                vq_util = sum(h_utils) / len(h_utils)
                h_str = ", ".join(f"H{i}:{u:.1f}%" for i, u in enumerate(h_utils))

                print(
                    f"[Chunk {c_idx:02d}] Step {chunk_step:04d} | "
                    f"Loss: {loss_val:.4f} (CE:{ce.item():.2f}, Traj:{traj.item():.4f}, VQ:{vq.item():.5f}) | "
                    f"||∇Overall||:{grad_norm:.2f} | "
                    f"VQ Heads: [{h_str}] (Avg:{vq_util:.1f}%) | "
                    f"Mem:{mx.get_active_memory() / 1e9:.2f}GB | "
                    f"{t1-t0:.2f}s",
                    flush=True
                )

            gc.collect()

        c_time = time.perf_counter() - t_chunk_start
        avg_loss = chunk_loss / max(chunk_step, 1)
        print(f"\n{'='*110}")
        print(f"✓ Completed Chunk {c_idx} in {c_time/60:.1f}m ({chunk_step} steps) | Avg Loss: {avg_loss:.4f}")
        print(f"{'='*110}")

        # Checkpoint weights after each completed chunk
        model.save_weights("student_4d_mind.safetensors")
        snapshot_name = f"student_4d_mind_chunk_{c_idx}.safetensors"
        model.save_weights(snapshot_name)
        print(f"Saved weights checkpoint: student_4d_mind.safetensors & {snapshot_name}\n", flush=True)

    p_time = time.perf_counter() - t_pipeline_start
    print("\n" + "=" * 110)
    print(f"ALL CHUNKS COMPLETE! Total time: {p_time/3600:.2f}h | Total Steps: {overall_step}")
    print(f"Final model weights saved to: student_4d_mind.safetensors")
    print("=" * 110)

if __name__ == "__main__":
    main()
