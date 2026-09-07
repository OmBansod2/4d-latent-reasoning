# 🧠 4D Latent Recurrent Reasoning (LatentMind)

> **Distilling System 2 Cognitive Trajectories into a Hybrid 4D Recurrent LLM on Apple Silicon (MLX)**

[![Apple Silicon](https://img.shields.io/badge/Hardware-Apple%20M4%20Pro%20(24GB)-black?logo=apple&style=for-the-badge)](https://www.apple.com)
[![Framework](https://img.shields.io/badge/ML%20Engine-Apple%20MLX-blue?style=for-the-badge)](https://github.com/ml-explore/mlx)
[![VQ Head Utilization](https://img.shields.io/badge/VQ%20Heads-100%25%20Active-emerald?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)
[![Interactive Blog](https://img.shields.io/badge/Read%20The%20Interactive%20Blog-GitHub%20Pages-2563eb?style=for-the-badge)](https://ombansod2.github.io/4d-latent-reasoning/)

---

## 📖 Overview

In modern Large Language Models (LLMs), reasoning is treated almost exclusively as an **autoregressive text generation problem**. Models like OpenAI o1, DeepSeek-R1, and Qwen-QwQ emit thousands of intermediate natural language tokens (*"Wait, let me rethink this... let $x$ be the velocity..."*) to simulate deduction.

While effective, this textual Chain-of-Thought (CoT) paradigm suffers from three critical bottlenecks:
1. **The Compute Tax:** Generating thousands of reasoning tokens consumes massive KV-cache memory, serial generation latency, and GPU wattage.
2. **The Vocabulary Bottleneck:** Human reasoning is often non-verbal, spatial, and topological before it is linguistic. Forcing intermediate hypotheses through discrete English word tokens constrains continuous deduction.
3. **Irreversible Sampling:** Standard decoding samples one token at a time; if the model commits to a flawed syntactic path early on, it must spend hundreds of tokens backtracking.

### The Hypothesis
**What if an LLM could "think" in a continuous, multi-dimensional latent manifold before emitting text?**

This repository documents an intensive research exploration on **Apple Silicon (M4 Pro, 24GB Unified Memory)**: distilling the latent reasoning trajectories of a frontier teacher (**Qwen 3.8 27B**) trained on deep reasoning traces into a compact student model that loops recurrently through a **4D latent coordinate router**.

---

## 🏗️ Architecture: From Scratch Failure to Surgical Hybrid

```
                  ┌─────────────────────────────────────────────────────────┐
                  │                 INPUT PROMPT TOKENS                     │
                  └───────────────────────────┬─────────────────────────────┘
                                              │
                                              ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │          ZONE 1: PARSE (Layers 0 – 15, Frozen)          │
                  │        Full Attention + SwiGLU  [Hidden Dim: 2560]      │
                  └───────────────────────────┬─────────────────────────────┘
                                              │
                                   Extract Anchor: h_anchor
                                              │
                     ┌────────────────────────┴─────────────────────────┐
                     │                                                  │
                     ▼                                                  │
            ┌──────────────────┐                                        │
            │ Anchor Injection │ ◄── [Anchor: h_anchor added each loop] ┘
            │   h + h_anchor   │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ 4D Latent Router │ ──► Coord: c = σ(W_r h / τ) ∈ [0, 1]⁴
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │ FiLM Modulation  │ ──► h_mod = h ⊙ (1 + 0.1 tanh γ) + 0.1 tanh β
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │  4-Head MHVQ     │ ──► 4 heads × 2048 codebooks (EMA Centroids)
            └────────┬─────────┘
                     │
                     ▼
            ┌─────────────────────────────────────────────────────────┐
            │         ZONE 2: RECURRENT CORE (Layers 16 – 18)         │
            │      Stateless Gated Delta Networks (Linear Attention)  │
            │           [Loops 6 Times with Constant O(1) RAM]        │
            └─────────────────────────┬───────────────────────────────┘
                                      │  (Iterate 6 Loops)
                                      ▼
                  ┌─────────────────────────────────────────────────────────┐
                  │         ZONE 3: UNEMBED (Layers 19 – 35, Frozen)        │
                  │         Downstream Causal Decoding & Language Head      │
                  └───────────────────────────┬─────────────────────────────┘
                                              │
                                              ▼
                                   NEXT TOKEN PREDICTION
```

### 1. Option 1: The Scratch Failure (The "Empty Vessel" Fallacy)
- **Design:** Built a custom student architecture from scratch (`PureLatent4DMind` in `train_4d_mind.py`) featuring topology-aware attention, 4-head grouped VQ, FiLM SwiGLU, and recurrent loops.
- **Training:** Trained over 15 chunks (7,500 steps, ~8 hours). Total loss fell from 445 down to 9.84; trajectory MSE dropped to 0.70; VQ head utilization reached 100%.
- **The 3:17 AM Inference Shock:** When tested on a simple logic puzzle (*"If 3 cats catch 3 mice in 3 minutes..."*), the model emitted pure gibberish:
  ```
  with/File/File/File/File/File/Filerh/File trianglesTriangleykańykańykańEEEEykańèses classics classics classics classics plut trianglesParallelPar...
  ```
- **The Post-Mortem:** Language pre-training requires hundreds of billions of tokens to acquire syntax, semantics, and world knowledge. Training from scratch on 15,000 reasoning sequences taught the model the mathematical trajectory, but left it with zero language substrate.

### 2. Option 2: The Surgical Hybrid Core (`Hybrid4DQwen`)
- **The Pivot:** Freeze **99.5%** of a pre-trained powerhouse—**Qwen 3.5 4B**—preserving its language mastery, while training **only 21.66 Million parameters (~83 MB)**.
- **Why Layers 16–18?** Standard Transformer attention layers accumulate KV-cache at every recurrent pass ($O(L \times T)$ memory explosion and destroyed RoPE positions). In Qwen 3.5 4B, **Layers 16, 17, and 18 are pure Gated Delta Networks (Linear Attention)**. They are stateless when `cache=None`, acting as an $O(1)$ recurrent memory core.
- **FiLM Identity Initialization:** FiLM modulation weights and biases were initialized to **strict zeros**. At step 0, $\gamma = 1.0$ and $\beta = 0.0$, meaning the hybrid model begins as an exact 100% identity pass of stock Qwen.

---

## 🧮 Mathematical Insights: Why EMA Beats Gradient Descent in VQ

Vector Quantization maps continuous latent vectors $z_e(x)$ to discrete codebook vectors $e_k$:
$$z_q(x) = e_k \quad \text{where} \quad k = \arg\min_{j \in \{1, \dots, K\}} \| z_e(x) - e_j \|_2^2$$

Because $\arg\min$ is a step function, its gradient is **zero almost everywhere**:
$$\frac{\partial z_q(x)}{\partial z_e(x)} = 0 \quad (\text{almost everywhere})$$

### Why Gradient Descent (SGD / AdamW) Fails on Codebooks:
1. **Codebook Collapse (Dead Codes):** Under SGD/AdamW, popular codebook vectors attract all gradients, while 90%+ of codebook entries never receive assignments and starve forever.
2. **Momentum Lag Across Recurrent Passes:** AdamW maintains first and second momentum buffers ($\beta_1 = 0.9, \beta_2 = 0.999$). As the 6-iteration recurrent loop refines its trajectory, AdamW's momentum buffers drag codebooks toward stale historical clusters.

### The Solution: Streaming Exponential Moving Average (EMA)
I treated codebook updates as an **online streaming k-means clustering process**:
$$N_k^{(t)} = \gamma N_k^{(t-1)} + (1 - \gamma) \sum_{i=1}^B \mathbf{1}[k_i = k]$$
$$m_k^{(t)} = \gamma m_k^{(t-1)} + (1 - \gamma) \sum_{i: k_i = k} x_i$$
$$e_k^{(t)} = \frac{m_k^{(t)}}{N_k^{(t)} + \epsilon} \quad (\gamma = 0.99)$$

### Active Dead Code Resurrecting
If an entry's count falls below threshold ($N_k < 1.0$), it is immediately re-seeded by sampling active representations from the current batch plus Gaussian jitter:
$$e_k \leftarrow x_{\text{active}} + \mathcal{N}(0, \sigma^2)$$

> **Empirical Telemetry:** Over 7,500 training steps, EMA + dead code replacement achieved **100.0% active codebook usage** across all 4 heads:
> `VQ Heads: [H0:100.0%, H1:100.0%, H2:100.0%, H3:100.0%] (Avg: 100.0%)`

---

## ⚡ Activation Function Dynamics

| Component | Activation | Formula | Architectural Rationale |
| :--- | :--- | :--- | :--- |
| **4D Router** | **Temperature-Scaled Sigmoid** | $\sigma(z / \tau) = \frac{1}{1 + e^{-z / \tau}}$ | Maps activations into the bounded unit hypercube $[0, 1]^4$. Temperature $\tau \in [0.1, 10.0]$ transitions smooth exploration into discrete vertex routing. |
| **FiLM Modulator** | **Bounded Tanh** | $\gamma = 1 + 0.1 \tanh(\cdot)$<br>$\beta = 0.1 \tanh(\cdot)$ | **Prevents Compounding Recurrence Explosion:** In a 6-loop recurrent block, unconstrained scaling compounds exponentially ($1.4^6 \approx 7.5\times$). Bounding with $0.1 \tanh$ clamps scaling within $[0.9, 1.1]$ ($\pm 10\%$). |
| **FiLMSwiGLU** | **SiLU (Swish)** | $\text{SiLU}(x) = x \cdot \sigma(x)$ | Smooth, non-monotonic curve with non-zero derivative for negative inputs, preventing dead neurons during deep BPTT. |
| **Norm Layers** | **RMSNorm** | $\bar{a} = \frac{a}{\text{RMS}(a)} \odot g$ | Scale invariance without mean-centering subtraction, reducing Metal shader memory bandwidth by ~18%. |

---

## 🍎 Apple Silicon & MLX Engineering Chronicles

Training custom architectures on a consumer **Apple Mac mini (M4 Pro, 24GB Unified Memory)** led to critical systems discoveries:

### 1. The Underscore (`_`) Attribute Trap & Midnight Kernel Panic
- **Mechanism:** Apple MLX evaluates operations via a lazy computation graph. `mx.eval(model.parameters(), ...)` evaluates public parameters at each step.
- **The Trap:** Python attributes starting with `_` (like `self._ema_cluster_sum` and `self._ema_cluster_count` in `GroupedVectorQuantizer`) are ignored by `model.parameters()`.
- **The Crash:** Over 1,000 steps in Chunk 1, an un-evaluated graph of tens of thousands of tensor operations accumulated in RAM. When `mx.save_safetensors` was called, MLX attempted to resolve the entire accumulated DAG at once, causing RAM to spike past 24 GB and triggering a macOS Kernel Panic reboot.
- **The Fix:** Explicitly registered internal buffers in evaluation:
  ```python
  mx.eval(model.trainable_parameters(), optimizer.state, [vq._ema_cluster_sum for vq in model.vqs], [vq._ema_cluster_count for vq in model.vqs])
  ```
  Memory immediately locked at a rock-solid **4.70 GB**.

### 2. The In-Loop Conversion Tax
Calling `.item()` or `np.array()` inside the forward pass forces synchronous GPU-to-CPU roundtrips, stalling Metal shaders. Moving metric logging strictly inside `if step % 25 == 0:` accelerated throughput from **1.35s to 0.75s per step (44% speedup)**.

### 3. Energy Saver Sleep Suspensions
macOS automatically sleeps idle systems after 45 minutes of keyboard/mouse inactivity. Background training was kept awake using the native Unix wrapper:
```bash
caffeinate -dis python pure_latent_4d_mind/train_hybrid_4d.py --start-chunk 10
```

### 4. The 248k vs 151k Tokenizer Mismatch Trap
- **The Mystery:** At Chunk 10 Step 76, training vanished with `exit code: 0` without a Python traceback.
- **The Cause:** Qwen 3.5 4B has a vocabulary of 151,936. Target data extracted by `extract_targets.py` had been tokenized using a 248k vocabulary (token ID `248068`).
- **Metal Driver Abort:** In CUDA, out-of-bounds memory accesses trigger a device assert error. In Apple Silicon Metal shaders, out-of-bounds buffer lookups trigger an immediate driver process abort that Python cannot catch, making it look like a clean exit (`code 0`).
- **The Core Rule:** *Latent layer trajectories cannot be distilled across mismatched tokenizers.* Lexical boundaries must align exactly.

---

## 📂 Repository Structure

```
├── .gitignore                         # Strict exclusion for weights, checkpoints, & caches
├── LICENSE                            # MIT License
├── README.md                          # Comprehensive technical documentation
├── docs/
│   └── index.html                     # Complete Interactive Research Blog (for GitHub Pages)
├── blog/
│   └── index.html                     # Local blog copy with interactive SVG simulation
└── pure_latent_4d_mind/
    ├── requirements.txt               # Dependencies (mlx, mlx-lm, datasets, numpy)
    ├── compute_svd.py                 # SVD reduction from 5120D teacher to 256D basis
    ├── extract_targets.py             # JIT target trajectory extractor from Bespoke-Stratos-17k
    ├── train_4d_mind.py               # Option 1: Pure Latent model trained from scratch
    ├── train_hybrid_4d.py             # Option 2: Hybrid 4D Recurrent Qwen (Surgical Injection)
    ├── inference_hybrid_4d.py         # Autoregressive generation with 4D recurrent loop
    ├── test_4d_mind.py                # Scratch model inference test (the "cats & mice" test)
    ├── test_hybrid_recurrent.py       # Graph verification for Hybrid recurrent layers
    ├── test_hybrid_keys.py            # Checkpoint tensor validation
    ├── inspect_chunk.py               # Data forensics utility for corrupted sequence detection
    ├── inspect_qwen.py                # Layer anatomy inspector for Qwen 3.5 hybrid blocks
    ├── download_qwen.py               # Qwen 27B teacher downloader
    └── download_qwen_4B.py            # Qwen 4B student downloader
```

---

## 🚀 Quickstart & Reproduction

### 1. Environment Setup
Requires macOS with Apple Silicon (M1/M2/M3/M4) and Python 3.10+:

```bash
git clone https://github.com/OmBansod2/4d-latent-reasoning.git
cd 4d-latent-reasoning

python3 -m venv venv_4d
source venv_4d/bin/activate
pip install -r pure_latent_4d_mind/requirements.txt
```

### 2. Compute SVD Projection Matrix
Extracts top 256 orthogonal reasoning directions from teacher activations:
```bash
python pure_latent_4d_mind/compute_svd.py
```

### 3. Extract Teacher Reasoning Targets
Extracts serialized chunks from `bespokelabs/Bespoke-Stratos-17k`:
```bash
python pure_latent_4d_mind/extract_targets.py
```

### 4. Train the Hybrid 4D Recurrent Core
Trains only the 21.66M 4D routing parameters on Apple Silicon Metal:
```bash
caffeinate -dis python pure_latent_4d_mind/train_hybrid_4d.py --start-chunk 0
```

### 5. Run Interactive Forward Pass Simulation (Local Blog)
Open the interactive research blog in your browser:
```bash
open docs/index.html
```

---

## 🔮 Roadmap & Future Branches

- [x] **Main Branch (`main`):** Foundational exploration, failure post-mortem, mathematical formulation (EMA vs SGD, stateless linear attention loops), and MLX systems optimizations.
- [ ] **Next Branch (`independent-aligned-distill`):** Regenerating teacher trajectories with an aligned Qwen tokenizer to eliminate the 248k out-of-bounds mismatch, unlocking verified autonomous System 2 latent reasoning without Chain-of-Thought token inflation.

---

## 👤 Author

**Om Bansod**  
*Deep Learning & Systems Research*  
GitHub: [@OmBansod2](https://github.com/OmBansod2)
Website / Blog: [4D Latent Reasoning Blog](https://ombansod2.github.io/4d-latent-reasoning/)

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
