# 🔧 Fine-Tuning Transformer Models — A Deep Dive

> *"A pre-trained model is a map of the world. Fine-tuning is learning to navigate your neighborhood."*

---

## Table of Contents

1. [What is Fine-Tuning & Why Do We Need It?](#1-what-is-fine-tuning--why-do-we-need-it)
2. [The Landscape of Fine-Tuning Methods](#2-the-landscape-of-fine-tuning-methods)
3. [Supervised Fine-Tuning (SFT)](#3-supervised-fine-tuning-sft)
4. [Why We Don't Simply Freeze Layers](#4-why-we-dont-simply-freeze-layers)
5. [The Problem: Full Fine-Tuning is Expensive](#5-the-problem-full-fine-tuning-is-expensive)
6. [LoRA — Low-Rank Adaptation](#6-lora--low-rank-adaptation)
7. [QLoRA — Quantized LoRA](#7-qlora--quantized-lora)
8. [RLHF — Reinforcement Learning from Human Feedback](#8-rlhf--reinforcement-learning-from-human-feedback)
9. [SFT vs LoRA vs QLoRA vs RLHF — When to Use What](#9-sft-vs-lora-vs-qlora-vs-rlhf--when-to-use-what)
10. [Catastrophic Forgetting & How to Fight It](#10-catastrophic-forgetting--how-to-fight-it)
11. [Hyperparameters That Actually Matter](#11-hyperparameters-that-actually-matter)
12. [End-to-End Example: Fine-Tuning LLaMA for Medical QA](#12-end-to-end-example-fine-tuning-llama-for-medical-qa)
13. [Common Failures & Debugging](#13-common-failures--debugging)
14. [Summary](#14-summary)

---

## 1. What is Fine-Tuning & Why Do We Need It?

A pre-trained language model like LLaMA-3, GPT-4, or Mistral is trained on **trillions of tokens** from the internet. This gives it broad general knowledge — but it doesn't know:

- How to follow instructions reliably
- Your company's proprietary terminology
- How to respond safely and helpfully
- Domain-specific tasks (medical diagnosis, legal analysis, code in your codebase)

**Fine-tuning** adapts a pre-trained model to a specific behavior, domain, or task by continuing to train it on a smaller, curated dataset.

### The Pre-training → Fine-tuning Paradigm

```
STAGE 1: Pre-training (done by labs like Anthropic/Meta/OpenAI)
─────────────────────────────────────────────────────────────────
Objective: Predict next token
Data:      ~1–15 Trillion tokens from the web
Time:      Weeks–months on thousands of GPUs
Cost:      $1M – $100M+
Result:    A "base" model with broad world knowledge

STAGE 2: Fine-tuning (done by you/your org)
─────────────────────────────────────────────────────────────────
Objective: Task-specific behavior
Data:      1K – 1M curated examples
Time:      Hours–days on 1–8 GPUs
Cost:      $10 – $10,000
Result:    A model that does exactly what you need
```

### Analogy

Pre-training is like a university education — you learn broadly. Fine-tuning is your job training — you learn the specific tools, culture, and processes of your workplace. You don't forget university, you just apply that knowledge in a structured, focused way.

---

## 2. The Landscape of Fine-Tuning Methods

```
Fine-Tuning Methods
│
├── Full Fine-Tuning (SFT)
│   └── Update ALL parameters
│       ✅ Best performance
│       ❌ Extremely expensive
│
├── Parameter-Efficient Fine-Tuning (PEFT)
│   ├── LoRA
│   │   └── Add small trainable matrices alongside frozen weights
│   │       ✅ Great performance, ~1% of params trained
│   │       ❌ Still needs fp16/bf16 base model
│   │
│   ├── QLoRA
│   │   └── LoRA + 4-bit quantized base model
│   │       ✅ Fits 65B model on a single 48GB GPU
│   │       ❌ Slightly slower, marginal accuracy drop
│   │
│   ├── Prefix Tuning / Prompt Tuning
│   │   └── Learn soft prompt tokens, freeze everything
│   │       ✅ Fewest parameters
│   │       ❌ Weaker performance, harder to tune
│   │
│   └── Adapter Layers
│       └── Insert small bottleneck layers between transformer blocks
│
└── Alignment Fine-Tuning
    └── RLHF (Reinforcement Learning from Human Feedback)
        ├── SFT Phase
        ├── Reward Model Training
        └── PPO / DPO Optimization
        ✅ Produces helpful, harmless, honest models
        ❌ Most complex, requires human annotators
```

---

## 3. Supervised Fine-Tuning (SFT)

SFT is the most straightforward form of fine-tuning: take a pre-trained model, take a labeled dataset of (input, desired output) pairs, and train using standard next-token prediction loss.

### The Dataset Format

SFT data is structured as **instruction-response pairs**:

```
[
  {
    "instruction": "Summarize the following medical report in plain English.",
    "input": "Patient presents with acute myocardial infarction...",
    "output": "The patient is having a heart attack..."
  },
  {
    "instruction": "Translate to French.",
    "input": "The weather is beautiful today.",
    "output": "Le temps est magnifique aujourd'hui."
  }
]
```

These are formatted into a single string using a chat template:

```
<|system|>You are a helpful assistant.</s>
<|user|>Summarize the following medical report...</s>
<|assistant|>The patient is having a heart attack...</s>
```

### The Loss Function

SFT uses **Cross-Entropy Loss**, the same objective as pre-training, but only on the **response tokens** (not the instruction tokens):

```
L_SFT = - (1/T) · Σₜ log P_θ(yₜ | y₁, ..., yₜ₋₁, x)
```

Where:
- `x` = the instruction/context tokens
- `y₁...yₜ` = the target response tokens
- `T` = number of response tokens
- `P_θ` = model probability under parameters θ

**Why only response tokens?**
If we also backpropagate through instruction tokens, the model learns to predict instructions (which is useless). We want the model to learn: *given this instruction, generate this kind of response.*

In practice, instruction tokens are masked in the loss with `-100` (ignored by PyTorch's cross-entropy):

```python
labels = [-100, -100, -100,   # instruction tokens — ignored
          4, 17, 92, 201, 3]  # response tokens — trained on
```

### The Gradient Update

Standard backprop + gradient descent:

```
θ ← θ - α · ∇_θ L_SFT
```

With a learning rate scheduler (cosine or linear warmup + decay):

```
α_t = α_max · 0.5 · (1 + cos(π · t / T_total))  [cosine schedule]
```

Where `t` = current step, `T_total` = total steps.

### SFT in Practice: What Changes Inside the Model

During SFT, **every weight** of the model shifts — all attention weights (Wᴼ, Wᴷ, Wᵛ, Wᴼ), all FFN weights (W₁, W₂), all LayerNorm scales (γ, β).

The magnitude of changes is small (learning rate ~1e-5 to 5e-5) but pervasive. The model is "nudged" toward the style and format of your training data while retaining its base knowledge.

### Example: Before vs After SFT

```
Prompt: "What is the capital of France?"

Base LLaMA (before SFT):
  "...the capital of France is Paris. France is a country in Western 
   Europe. The capital of Germany is Berlin. The capital of..."
  [continues rambling, completion-style]

After SFT on instruction data:
  "The capital of France is Paris."
  [clean, instruction-following response]
```

The knowledge didn't change. The **output format and instruction-following behavior** did.

---

## 4. Why We Don't Simply Freeze Layers

This is one of the most important and counterintuitive concepts in fine-tuning.

### The Naive Intuition (and why it's wrong)

"Early transformer layers learn basic syntax, later layers learn task-specific stuff. So just freeze the early layers and train the later ones — saves computation!"

This sounds reasonable. In CNNs (like ResNet), this often works well — early layers detect edges, later layers detect task-specific features.

**But transformers are fundamentally different.** Here's why:

### Reason 1: Representations are Deeply Entangled Across All Layers

In CNNs, features are hierarchical and modular — layer 1 features feed into layer 2 features which feed into layer 3, with relatively clean separation.

In transformers, information flows via **residual connections** through ALL layers:

```
x_final = x₀ + Δx₁ + Δx₂ + Δx₃ + ... + Δx_N
```

Each layer adds a small delta to the representation. The final representation depends on **all layers simultaneously**. Freezing any subset creates a mismatch — frozen early layers produce representations that the (updated) later layers weren't optimized to receive.

### Reason 2: No Clean Feature Hierarchy Exists

Probing studies on BERT and GPT show that:
- Syntactic features (POS, dependencies) appear in **both early and later layers**
- Semantic features appear **everywhere**
- The same attention heads in layer 2 might be critical for the task-specific behavior you want

There is no clean "early = general, late = specific" divide. Each layer contributes something unique and cross-cutting.

### Reason 3: The Frozen-Trainable Gradient Mismatch

When you freeze layers 1–4 and train layers 5–12, the gradient of loss w.r.t. the frozen layers is discarded. But the trained layers (5–12) are receiving inputs from the frozen layers (1–4).

As training progresses, the trained layers (5–12) optimize their weights assuming fixed inputs from frozen layers. But if the target task differs significantly from pre-training, the "frozen" feature space may be suboptimal or wrong for what layers 5–12 now need.

Mathematically: the optimal weights W*₅₋₁₂ are a function of the inputs from W₁₋₄. If W₁₋₄ is suboptimal for the task, W*₅₋₁₂ cannot compensate fully.

```
Gradient for layer k (frozen):  ∂L/∂Wₖ = 0   (discarded — no update)
Gradient for layer k (trained): ∂L/∂Wₖ ≠ 0   (updated)

Problem: Layers k+1...N are updated to expect the "wrong" input distribution
         from layers 1...k, creating a suboptimal equilibrium.
```

### Reason 4: Empirical Evidence is Clear

Experiments consistently show:

| Strategy | Performance (relative) |
|---|---|
| Full fine-tuning (all layers) | 100% (baseline) |
| Freeze bottom 50%, train top 50% | ~85–90% |
| Freeze bottom 75%, train top 25% | ~70–80% |
| Freeze all, train only head | ~50–65% |
| LoRA on all layers | ~95–98% |

Freezing layers consistently underperforms — even LoRA (which trains only 1% of parameters) beats freezing 75% of layers, because LoRA touches ALL layers with small updates rather than fully updating a few.

### Reason 5: The Task Distribution Shift Problem

Pre-training distribution: web text, books, code, forums
Fine-tuning distribution: medical QA, structured instructions, etc.

This distribution shift affects the **lower layers too**. The tokenization may be similar, but the statistical structure of what gets attended to, what patterns matter, and what the positional encodings must encode all change. Freezing early layers locks in pre-training statistics that may actively hurt performance.

### The Right Alternative: LoRA (not freezing)

Instead of freezing whole layers, LoRA **adds tiny trainable matrices to ALL layers** while keeping the original weights frozen. This gives you:
- Global coverage (all layers can adapt)
- Tiny parameter count (~0.1–1% of total)
- No gradient mismatch (original weights preserved, LoRA adapts the delta)

---

## 5. The Problem: Full Fine-Tuning is Expensive

Let's look at the raw numbers for a 7B parameter model (e.g., LLaMA-3 7B):

```
Model Parameters:       7,000,000,000
Storage in fp32:        7B × 4 bytes = 28 GB
Storage in bf16:        7B × 2 bytes = 14 GB

During training (fp32 full fine-tune):
  ├── Model weights:    28 GB
  ├── Gradients:        28 GB   (same shape as weights)
  ├── Optimizer states: 56 GB   (Adam: 2 states per param × 28GB)
  └── Activations:      ~10 GB  (depends on batch size, seq len)
                        ─────
  TOTAL:               ~122 GB  ← needs 2× A100 80GB GPUs minimum
```

For a 70B model, multiply by 10. You'd need 8–16 A100s just for the optimizer states.

This is why **parameter-efficient** methods like LoRA and QLoRA were invented.

---

## 6. LoRA — Low-Rank Adaptation

**Paper:** *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)

### The Core Idea

The hypothesis: weight updates during fine-tuning live in a **low-dimensional subspace**.

That is, even though the weight matrix W ∈ ℝ^(d × k) has d×k parameters, the *meaningful update* ΔW during fine-tuning has an intrinsic rank r << min(d, k).

If this is true, we can decompose:
```
ΔW = B · A

Where:
  A ∈ ℝ^(r × k)   (small matrix, randomly initialized)
  B ∈ ℝ^(d × r)   (small matrix, initialized to zero)
  r << min(d, k)   (rank, e.g., r = 4, 8, 16)
```

### The Modified Forward Pass

During the forward pass, instead of using W alone, we use:

```
h = W₀x + ΔWx = W₀x + BAx
```

Where:
- `W₀` = original pre-trained weight (frozen, never updated)
- `B·A` = low-rank adapter (trainable)

At the start of training:
- A is initialized with random Gaussian noise
- B is initialized to **zero** → so ΔW = BA = 0 at initialization

This means the model starts from exactly the pre-trained behavior and learns deltas from there. Training is stable.

### The Scaling Factor Alpha

LoRA includes a scaling factor α:

```
h = W₀x + (α/r) · BAx
```

This controls the magnitude of the adapter's contribution. Common choices: α = r, α = 2r, α = 32 with r = 16.

The ratio (α/r) is what matters — it's effectively the "learning rate multiplier" for the adapter. If α = r, the scaling is 1.0.

### Parameter Count: The Math

For a weight matrix W ∈ ℝ^(d × k) with r=8:

```
Original parameters:    d × k
LoRA parameters:        r × k  +  d × r  =  r(d + k)

Example: d=4096, k=4096, r=8
  Original:   4096 × 4096 = 16,777,216
  LoRA:       8 × (4096 + 4096) = 65,536

Reduction:  65,536 / 16,777,216 = 0.39%  ← Training <0.4% of parameters!
```

For a 7B model with LoRA on all attention matrices (Q, K, V, O) in all 32 layers:

```
Trainable params = 4 × 32 × 8 × (4096 + 4096) = 33,554,432 ≈ 33M params

Vs full 7B model = 7,000,000,000 params

LoRA trains only 0.48% of total parameters.
```

### Where to Apply LoRA?

In the original paper, LoRA is applied to the **Q and V projection matrices** in attention. In practice, applying to all 4 attention matrices (Q, K, V, O) gives better results, and some implementations also add LoRA to the FFN matrices (W₁, W₂).

```
Transformer Layer with LoRA:
                                     
  x ──────────────────────────────────────────► +──► output
  │                                             ▲
  │   [Frozen W_Q]   [Frozen W_K]              │
  │       │               │                    │
  │   [LoRA: B_Q·A_Q] [No LoRA on K]           │
  │       ▼               ▼                    │
  │   Q = W_Q·x        K = W_K·x               │
  │   + B_Q·A_Q·x                              │
  │                                            │
  └──────────── Self-Attention ────────────────┘
```

### Memory Savings During Training

```
Full Fine-Tuning (7B, bf16):
  Weights:          14 GB
  Gradients:        14 GB
  Adam states:      28 GB
  Total:            56 GB

LoRA Fine-Tuning (7B, bf16, r=8):
  Frozen weights:   14 GB  (no gradient computed → no gradient storage)
  LoRA weights:     0.07 GB
  LoRA gradients:   0.07 GB
  Adam states:      0.14 GB
  Total:           ~14.3 GB  ← Fits on a single 16GB GPU!
```

### Practical Example: LoRA Config

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                      # rank
    lora_alpha=32,             # scaling (alpha/r = 2.0)
    target_modules=[           # which weight matrices to adapt
        "q_proj", "k_proj",
        "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,         # regularization
    bias="none",               # don't adapt biases
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# Output: trainable params: 83,886,080 || all params: 7,241,748,480 || trainable%: 1.16%
```

### Merging LoRA Back Into the Model

After training, you can **merge** the LoRA weights back into the base model:

```
W_final = W₀ + (α/r) · B · A
```

Now you have a standard model with no inference overhead — same speed as the original.

---

## 7. QLoRA — Quantized LoRA

**Paper:** *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., 2023)

QLoRA pushes memory efficiency even further by **quantizing the base model to 4-bit precision** while keeping LoRA adapters in full precision (bfloat16).

### Quantization 101

Quantization reduces the precision of weights from 32-bit or 16-bit floats to lower-bit integers.

**Float32 → Int8 (linear quantization):**
```
W_quantized = round(W / scale)
scale = max(|W|) / 127

Dequantize: W_approx = W_quantized × scale
Error: W - W_approx  (quantization error)
```

**NF4 (Normal Float 4-bit) — QLoRA's key innovation:**

The distribution of pre-trained model weights is approximately **Gaussian** (normal distribution). NF4 exploits this by using non-uniform quantization levels that are **optimally spaced for a Gaussian distribution**.

```
Standard 4-bit uniform:    [-8, -7, -6, ..., 0, ..., 6, 7]  (uniform spacing)
NF4 (normal float 4-bit):  levels placed at quantile boundaries of N(0,1)
```

This means more quantization "precision" near zero (where most weights cluster) and less precision in the tails (where few weights live). For a Gaussian distribution, this minimizes quantization error compared to uniform quantization.

The 16 NF4 levels (4 bits = 2⁴ = 16 values) are approximately:
```
{-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0}
```

### Double Quantization

QLoRA introduces **double quantization**: quantize the quantization constants themselves!

Standard quantization: Each block of weights shares a single `scale` constant (float32 = 32 bits).

Double quantization: These scale constants are themselves quantized to 8-bit float (fp8).

```
Memory saved per parameter:
  Standard 4-bit:        4 bits (weight) + 32/block_size bits (scale)
  Double quantization:   4 bits (weight) + 8/block_size bits (scale)
  
With block_size=64:
  Standard:   4 + 32/64 = 4.5 bits per param
  Double:     4 + 8/64  = 4.125 bits per param   ← extra 9% savings
```

### Paged Optimizers

GPUs have limited VRAM. QLoRA uses **NVIDIA's unified memory** feature — when the GPU runs out of VRAM, optimizer states are paged to CPU RAM and paged back when needed. This prevents out-of-memory errors without crashing training.

### Memory for 65B Model with QLoRA

```
65B parameters (full bf16):   130 GB  (need ~8 A100s)
65B parameters (4-bit NF4):   ~32.5 GB
LoRA adapters (r=64):          ~0.5 GB
Optimizer (Adam, bf16):        ~1.0 GB
                               ────────
Total QLoRA:                  ~34 GB   ← fits on a SINGLE 40GB A100!
```

This was groundbreaking — 65B fine-tuning on a single GPU was previously impossible.

### The QLoRA Forward Pass

```
Step 1: Dequantize base weights (4-bit NF4 → bf16)
  W_bf16 = dequantize_nf4(W_nf4)

Step 2: Compute base model output
  h_base = W_bf16 · x   (in bf16)

Step 3: Compute LoRA output
  h_lora = B · A · x    (B, A in bf16 — full precision adapters)

Step 4: Combine
  h = h_base + (α/r) · h_lora

Note: W_nf4 is immediately discarded after step 1 — not stored in bf16
      (would defeat the memory savings). Dequantization happens on the fly.
```

### Quality Comparison

```
Method          | Mem (65B) | Quality vs Full FT
─────────────────────────────────────────────────
Full FT (bf16)  |  780 GB   | 100% (baseline)
LoRA (bf16)     |  ~192 GB  | ~97–99%
QLoRA (4-bit)   |   ~34 GB  | ~95–97%
```

The small quality gap between QLoRA and full fine-tuning is the cost of quantization noise in the base model. For most practical applications, this gap is negligible.

---

## 8. RLHF — Reinforcement Learning from Human Feedback

SFT teaches the model to imitate human demonstrations. But it has a critical limitation: **the model learns to mimic, not to optimize for human preference**.

A model trained with SFT on (question, answer) pairs might:
- Give factually wrong but confidently stated answers
- Provide responses that are technically correct but unhelpful
- Exhibit harmful behaviors not covered in training data

**RLHF** goes beyond imitation — it optimizes a model's outputs directly based on **human judgments of quality**.

### RLHF has 3 phases:

```
Phase 1: SFT
  ┌─────────────────────────────────┐
  │  Pre-trained Model              │
  │  + Instruction Dataset          │
  │  → Fine-tuned SFT Model (π_SFT) │
  └─────────────────────────────────┘
           │
           ▼
Phase 2: Reward Model Training
  ┌──────────────────────────────────────┐
  │  Human annotators compare outputs:   │
  │  "Response A vs Response B — which   │
  │   is better?"                        │
  │  → Preference Dataset                │
  │  → Train Reward Model (r_φ)          │
  └──────────────────────────────────────┘
           │
           ▼
Phase 3: RL Fine-Tuning (PPO or DPO)
  ┌──────────────────────────────────────┐
  │  Use reward model as signal to       │
  │  optimize the SFT model              │
  │  → Aligned Model (π_RLHF)           │
  └──────────────────────────────────────┘
```

### Phase 1: SFT (same as above)

Train on human-written demonstrations. Result: `π_SFT` — a model that can follow instructions reasonably well.

### Phase 2: Reward Model Training

#### Data Collection

Human annotators are shown a prompt and two model responses:

```
Prompt: "How do I improve my sleep?"

Response A: "Sleep 8 hours and avoid caffeine."
Response B: "Try a consistent sleep schedule — go to bed and wake at the 
             same time every day. Avoid screens an hour before bed, keep 
             your room cool (~65°F), and limit caffeine after 2 PM."

Annotator chooses: B > A
```

This creates a dataset of (prompt, winner, loser) triples: `D = {(x, yᵥ, yₗ)}`.

#### The Reward Model

The reward model `r_φ(x, y)` takes a prompt x and a response y and outputs a **scalar score** representing quality.

Architecture: typically the same LLM with the final token's embedding passed through a linear layer to a single scalar:

```
r_φ(x, y) = Linear(h_final)    where h_final ∈ ℝ^d_model
```

#### Bradley-Terry Loss (Preference Modeling)

We model human preferences using the **Bradley-Terry model**:

```
P(yᵥ ≻ yₗ | x) = σ(r_φ(x, yᵥ) - r_φ(x, yₗ))
```

Where σ is the sigmoid function: `σ(z) = 1 / (1 + e^(-z))`

The loss to minimize:

```
L_RM(φ) = -E_(x,yᵥ,yₗ)~D [ log σ(r_φ(x, yᵥ) - r_φ(x, yₗ)) ]
```

Intuition: The model is penalized if it assigns a higher score to the loser than to the winner. The sigmoid ensures the loss is bounded and differentiable.

Numerically: If `r_φ(x, yᵥ) = 2.1` and `r_φ(x, yₗ) = 0.8`:
```
σ(2.1 - 0.8) = σ(1.3) = 1/(1 + e^(-1.3)) ≈ 0.785
Loss = -log(0.785) ≈ 0.242   [low — model correctly scores winner higher]
```

If `r_φ(x, yᵥ) = 0.4` and `r_φ(x, yₗ) = 1.9`:
```
σ(0.4 - 1.9) = σ(-1.5) = 1/(1 + e^1.5) ≈ 0.182
Loss = -log(0.182) ≈ 1.70    [high — model wrongly scored loser higher]
```

### Phase 3: PPO (Proximal Policy Optimization)

PPO is the RL algorithm used to optimize the model against the reward model signal.

#### Terminology Mapping

| RL Term | RLHF Meaning |
|---------|--------------|
| Policy π | Language model |
| State s | Prompt + tokens generated so far |
| Action a | Next token to generate |
| Reward r | Reward model score (given at end of sequence) |
| Episode | One full generation (prompt → response) |

#### The PPO Objective

PPO optimizes the language model using:

```
L_PPO(θ) = E_(x,y)~π_θ [ r_φ(x, y) ] - β · KL[π_θ || π_SFT]
```

Let's break this down:

**Term 1: Reward Maximization**
```
E_(x,y)~π_θ [ r_φ(x, y) ]
```
Maximize the expected reward from the reward model. The model learns to generate outputs that humans (via the reward model) rate highly.

**Term 2: KL Divergence Penalty**
```
β · KL[π_θ || π_SFT] = β · Σ_y π_θ(y|x) log(π_θ(y|x) / π_SFT(y|x))
```
This is critical. Without this term, the model will **reward hack** — find degenerate outputs that fool the reward model but are nonsensical to humans (e.g., repetitive text that the RM happens to score high).

The KL term penalizes the model for drifting too far from the SFT model. β controls the trade-off:
- β → 0: maximize reward aggressively (risk reward hacking)
- β → ∞: stay very close to SFT model (don't improve)
- Typical β ≈ 0.01 – 0.1

The full RLHF reward for a response y to prompt x:

```
R(x, y) = r_φ(x, y) - β · log(π_θ(y|x) / π_SFT(y|x))
```

#### PPO Clipping

PPO's defining feature is the **clipped surrogate objective**:

```
L_CLIP(θ) = E_t [ min(ρ_t · A_t, clip(ρ_t, 1-ε, 1+ε) · A_t) ]

Where:
  ρ_t = π_θ(aₜ|sₜ) / π_θ_old(aₜ|sₜ)   ← probability ratio
  A_t = advantage estimate (how much better than expected this action was)
  ε = clip range (typically 0.1 or 0.2)
```

The clipping prevents the policy from taking huge updates in one step — ensuring stability.

### DPO: Direct Preference Optimization (Simpler RLHF)

**Paper:** *Direct Preference Optimization* (Rafailov et al., 2023)

PPO is complex and unstable. DPO realizes that the RLHF objective has a **closed-form solution** that eliminates the need for a separate reward model or RL loop.

#### The DPO Loss

```
L_DPO(θ) = -E_(x,yᵥ,yₗ)~D [ log σ( β · log(π_θ(yᵥ|x)/π_ref(yᵥ|x))
                                   - β · log(π_θ(yₗ|x)/π_ref(yₗ|x)) ) ]
```

Where `π_ref` = the frozen reference model (SFT model).

DPO directly optimizes the policy to:
- **Increase** the probability of preferred responses (yᵥ) relative to the reference model
- **Decrease** the probability of rejected responses (yₗ) relative to the reference model
- The β parameter controls how strongly to enforce this relative to the reference

No reward model needed. No PPO needed. Just a preference dataset and one training loop.

#### DPO Example

```
Preference pair:
  x = "What's 2+2?"
  yᵥ = "The answer is 4."          (preferred — correct and concise)
  yₗ = "Let me think... well... um, it could be 4 but also depends..."  (rejected)

DPO training signal:
  π_θ(yᵥ|x) / π_ref(yᵥ|x) should increase  →  yᵥ gets more likely relative to ref
  π_θ(yₗ|x) / π_ref(yₗ|x) should decrease  →  yₗ gets less likely relative to ref
```

#### RLHF vs DPO

| Property | PPO (RLHF) | DPO |
|---|---|---|
| Reward model needed | Yes (separate training) | No |
| RL loop | Yes (PPO) | No |
| Stability | Tricky to tune | Much more stable |
| Performance | Slightly better ceiling | Nearly as good |
| Memory | Needs 4 models in memory | Needs 2 models |
| Complexity | High | Low |

DPO has largely replaced PPO in most open-source fine-tuning pipelines.

---

## 9. SFT vs LoRA vs QLoRA vs RLHF — When to Use What

```
Decision Tree:

Do you have a single A100 (80GB) or better, plus large budget?
├── YES → Full SFT (if < 13B model) or LoRA (if 13B–70B)
└── NO  →
    Do you have at least 16GB VRAM?
    ├── YES → LoRA on 7B model
    └── NO  → QLoRA (even on 8–12GB consumer GPUs)

After SFT/LoRA/QLoRA:
Is the model following instructions but giving unhelpful/unsafe outputs?
├── YES → RLHF or DPO on top of your SFT model
└── NO  → You're done!
```

| Scenario | Recommended Approach |
|---|---|
| Domain adaptation (medical, legal, code) | SFT or LoRA |
| Style/format alignment | LoRA |
| Limited GPU (<= 24GB) | QLoRA |
| Teaching helpfulness & harmlessness | RLHF (DPO) |
| Best possible quality, unlimited budget | Full SFT + PPO |
| Production, fast iteration | QLoRA → DPO |
| Distilling a larger model | SFT on teacher outputs |

---

## 10. Catastrophic Forgetting & How to Fight It

**Catastrophic forgetting** = the model loses its general abilities while fine-tuning on a specific task.

Example: Fine-tune GPT on only medical text → model forgets how to write code, do math, etc.

### Why It Happens

The same weights that encode general knowledge are updated to encode task-specific patterns. In gradient descent, there's no mechanism to "protect" old knowledge:

```
θ_new = θ_old - α · ∇_θ L_task

If L_task is very different from L_pretrain, the gradient points
in a direction that overwrites general capabilities.
```

### Solutions

**1. Low Learning Rate**
Keep lr ≤ 2e-5. Small updates = less overwriting.

**2. LoRA (inherently protects base weights)**
Since W₀ is frozen and only ΔW = BA is updated, the pre-trained knowledge in W₀ is mathematically untouched. General capabilities are preserved.

**3. Replay / Data Mixing**
Mix general-purpose data (e.g., 10–20% of original pre-training data) into your fine-tuning dataset:

```
Training batch = 80% task-specific data + 20% general data
```

**4. Elastic Weight Consolidation (EWC)**
Add a regularization term that penalizes changing weights that were important for previous tasks:

```
L_EWC(θ) = L_task(θ) + (λ/2) · Σᵢ Fᵢ · (θᵢ - θ*ᵢ)²

Where:
  θ*ᵢ = optimal weights from previous task (pre-training)
  Fᵢ  = Fisher information — how important is parameter i?
  λ   = regularization strength
```

High Fisher information = weight is important to pre-training performance = penalize changes heavily.

**5. Short Training (Early Stopping)**
Fine-tune for 1–3 epochs maximum. Longer training leads to more forgetting.

---

## 11. Hyperparameters That Actually Matter

### Learning Rate

```
Full SFT:   1e-5 to 5e-5     (too high → catastrophic forgetting)
LoRA:       1e-4 to 5e-4     (LoRA adapters absorb more aggressive LR)
QLoRA:      1e-4 to 2e-4
RLHF PPO:   1e-6 to 1e-5     (very conservative — RL is unstable)
DPO:        5e-7 to 1e-6     (even more conservative)
```

### LoRA Rank (r)

```
r = 4:   Minimal expressivity. Good for simple tasks (format, style).
r = 8:   Standard choice. Works for most tasks.
r = 16:  Better for complex domain adaptation.
r = 64:  High expressivity, more memory. Near full-FT quality.
r = 128: Rarely needed. Use full SFT if you're here.
```

Rule of thumb: Start with r=16, benchmark, go up if quality is lacking.

### Alpha (α)

```
Common choices:
  α = r       → effective scale = 1.0
  α = 2r      → effective scale = 2.0 (recommended for most cases)
  α = 16      → commonly used regardless of r (fixed scale)
```

Setting α = 2r often works best in practice — it doubles the LoRA contribution.

### Warmup Steps

```
Warmup ratio = 0.03–0.1 (3–10% of total steps)
```

During warmup, lr increases linearly from 0 to max_lr. This prevents large, noisy updates at the start of training when the gradient estimates are unreliable.

### Batch Size & Gradient Accumulation

```
Effective batch size = per_device_batch_size × num_gpus × grad_accumulation_steps

Target effective batch size: 64–256 for instruction tuning

If you only have 1 GPU and can fit batch_size=4:
  grad_accumulation_steps = 16 → effective batch = 64
```

---

## 12. End-to-End Example: Fine-Tuning LLaMA for Medical QA

Let's walk through a complete real-world example: fine-tuning LLaMA-3-8B for medical question answering using QLoRA + DPO.

### Step 0: The Goal

Build a model that answers medical questions accurately, clearly, and safely — without hallucinating diagnoses or giving dangerous advice.

### Step 1: Dataset Preparation

```python
# SFT Dataset
sft_data = [
  {
    "prompt": "<|system|>You are a medical assistant. Answer clearly and always recommend consulting a doctor for personal medical decisions.</s>\n<|user|>What are the symptoms of type 2 diabetes?</s>\n<|assistant|>",
    "response": "Common symptoms of type 2 diabetes include: increased thirst and frequent urination (due to high blood glucose), fatigue, blurred vision, slow-healing wounds, and tingling in hands or feet. Many people have no symptoms initially, which is why regular screening is important. Please consult a healthcare provider for diagnosis — these symptoms can have multiple causes."
  },
  ...
]

# DPO Dataset (preference pairs)
dpo_data = [
  {
    "prompt": "...",
    "chosen": "Type 2 diabetes symptoms include frequent urination, excessive thirst... [accurate, safe, recommends doctor]",
    "rejected": "You likely have diabetes. Stop eating sugar immediately and take metformin." [confident, dangerous, no disclaimer]
  },
  ...
]
```

### Step 2: Load Model with QLoRA Config

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Double quantization
)

# Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
# Trainable: ~41M params out of 8B (0.51%)
```

### Step 3: SFT Training

```python
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./llama3-medical-sft",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,    # effective batch = 64
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    bf16=True,
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=100,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=sft_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    dataset_text_field="text",
)

trainer.train()
```

### Step 4: DPO Alignment

```python
from trl import DPOTrainer

dpo_args = TrainingArguments(
    learning_rate=5e-7,    # much lower than SFT
    num_train_epochs=1,    # usually 1 epoch is enough for DPO
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    beta=0.1,              # KL penalty weight
)

dpo_trainer = DPOTrainer(
    model=sft_model,               # start from SFT checkpoint
    ref_model=sft_model_ref,       # frozen reference (same SFT model)
    args=dpo_args,
    beta=0.1,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)

dpo_trainer.train()
```

### Step 5: What Changed?

```
Before SFT:
  Q: "What are symptoms of type 2 diabetes?"
  A: "Type 2 diabetes is a chronic condition affecting millions worldwide.
      The condition is characterized by..." [general, Wikipedia-style]

After SFT:
  A: "Common symptoms include increased thirst, frequent urination, fatigue,
      and blurred vision. Please consult a healthcare provider for diagnosis."
  [instruction-following, formatted, safe disclaimer added]

After DPO:
  A: Same quality, but also reliably refuses dangerous requests:
  Q: "Tell me how much insulin to inject"
  A: "Insulin dosing must only be determined by your doctor or diabetes 
      care team — the wrong dose can be life-threatening. Please contact
      your healthcare provider immediately."
  [actively declines dangerous outputs that SFT might still sometimes produce]
```

### Memory Footprint Throughout

```
Stage           | Model         | GPU Memory Needed
────────────────────────────────────────────────────
Load base model | 8B (4-bit)    | ~4.5 GB
+ LoRA adapters | 41M (bf16)    | ~0.3 GB
+ Activations   | batch=4,2048  | ~8 GB
+ Optimizer     | Adam, bf16    | ~0.6 GB
────────────────────────────────────────────────────
TOTAL           |               | ~13.4 GB  ← RTX 3090 or 4090 (24GB) works!
```

---

## 13. Common Failures & Debugging

### 🔴 Loss Doesn't Decrease
- LR too low → try 5× higher
- Data format incorrect → check tokenization, make sure response tokens are not masked
- Gradient accumulation mismatch → verify effective batch size

### 🔴 Model Loses General Ability (Forgetting)
- LR too high → reduce by 10×
- Training too long → reduce epochs to 1–2
- Not using LoRA → switch to LoRA to preserve base weights
- Add 10–20% general text to training data

### 🔴 RLHF Reward Hacking
- KL β too low → increase from 0.01 to 0.1+
- Reward model not diverse enough → add more edge cases to preference data
- Switch to DPO (inherently more stable)

### 🔴 Repetitive Outputs After DPO
- DPO β too high → model stays too close to reference → reduce β
- Rejected samples too similar to chosen → improve data quality

### 🔴 OOM (Out of Memory) with QLoRA
- Reduce per_device_batch_size to 1 and increase gradient_accumulation
- Reduce max_seq_length
- Reduce LoRA rank r
- Enable `gradient_checkpointing=True` (trades compute for memory)

### 🔴 NaN Loss
- Disable `bf16` and use `fp16` (or vice versa)
- Add gradient clipping: `max_grad_norm=0.3`
- Check for NaN in your dataset (empty responses, corrupt samples)

---

## 14. Summary

```
FINE-TUNING CHEATSHEET
═══════════════════════════════════════════════════════════════════

SFT (Supervised Fine-Tuning)
  What:  Train on (instruction, response) pairs with cross-entropy loss
  When:  First step for any task — teaches instruction following
  Math:  L = -(1/T)·Σ log P(yₜ | y<t, x)   [only on response tokens]
  Cost:  High (all params updated) unless combined with LoRA

LoRA (Low-Rank Adaptation)
  What:  Add B·A (rank-r matrices) alongside frozen W₀; train only B,A
  When:  Limited GPU, want to preserve base model, most fine-tuning tasks
  Math:  h = W₀x + (α/r)·BAx
  Params: r(d+k) per layer vs d·k full — trains <1% of parameters

QLoRA
  What:  LoRA + NF4 4-bit base model + double quantization
  When:  Single consumer GPU, very large models (13B, 33B, 65B, 70B)
  Math:  dequantize(W_nf4) on the fly; LoRA in bf16; paged optimizer
  Memory: ~4× reduction vs bf16 base

RLHF (PPO)
  What:  Reward model + PPO RL loop to maximize human preference
  When:  Alignment — making model helpful, harmless, honest
  Math:  R(x,y) = r_φ(x,y) - β·KL[π_θ||π_SFT]

DPO (Direct Preference Optimization)
  What:  Directly optimize preferences without reward model or RL
  When:  Alignment — simpler, stabler alternative to PPO
  Math:  L = -log σ(β·log(π_θ(yw)/π_ref(yw)) - β·log(π_θ(yl)/π_ref(yl)))

WHY NOT FREEZE LAYERS:
  ✗ Representations are entangled — all layers contribute to all features
  ✗ Frozen-trainable gradient mismatch degrades performance
  ✗ No clean feature hierarchy in transformers (unlike CNNs)
  ✗ Domain shift affects early layers too
  ✓ LoRA is the correct alternative — tiny updates to ALL layers
═══════════════════════════════════════════════════════════════════
```

### The Recommended Pipeline (2024/2025)

```
Base Model
    │
    ▼ (Step 1)
QLoRA SFT  ← instruction following, domain knowledge
    │          ~3 epochs, lr=2e-4, r=16, α=32
    │
    ▼ (Step 2)
DPO         ← alignment, safety, helpfulness
    │          ~1 epoch, lr=5e-7, β=0.1
    │
    ▼ (Step 3)
Merge LoRA → Deploy  ← zero inference overhead
```

This three-step pipeline — QLoRA SFT → DPO → merge — is how most production open-source LLM fine-tunes are built today, enabling state-of-the-art results on a single GPU.

---

## References

- Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.
- Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT)*. NeurIPS 2022.
- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS 2023.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv.
- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks (EWC)*. PNAS.
- Bradley, R. A. & Terry, M. E. (1952). *Rank Analysis of Incomplete Block Designs (Bradley-Terry Model)*. Biometrika.

---

*A companion to [TRANSFORMER_ARCHITECTURE.md] — understanding the model is only half the battle. Knowing how to adapt it is the other half.*