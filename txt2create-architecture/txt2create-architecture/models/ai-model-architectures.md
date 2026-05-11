# AI Model Architectures - Deep Technical Documentation

## Overview

This document provides in-depth technical details of all AI models used in TXT2CREATE, including their architectures, how they process data, optimization techniques, and GPU serving strategies.

---

## TABLE OF CONTENTS

1. [Stable Diffusion Architecture](#1-stable-diffusion-architecture)
2. [VAE (Variational Autoencoder)](#2-vae-variational-autoencoder)
3. [Text-to-Video Pipeline (Stable Diffusion + Motion)](#3-text-to-video-pipeline)
4. [LLM Architecture (Llama 3)](#4-llm-architecture-llama-3)
5. [Audio Generation Models](#5-audio-generation-models)
6. [GAN Architecture for Avatar Generation](#6-gan-architecture-for-avatar-generation)
7. [Video Captioning Models](#7-video-captioning-models)
8. [Audio-to-Text (Whisper)](#8-audio-to-text-whisper)
9. [GPU Optimization Techniques](#9-gpu-optimization-techniques)
10. [Model Serving Architecture](#10-model-serving-architecture)

---

## 1. STABLE DIFFUSION ARCHITECTURE

### 1.1 High-Level Overview

Stable Diffusion is a **latent diffusion model** that generates images from text prompts by iteratively denoising random noise in a compressed latent space.

```
Text Prompt → CLIP Text Encoder → Text Embeddings (77×768)
                                          ↓
Random Noise (latent) ← ← ← ← U-Net Denoising (T steps)
    ↓                              ↑
VAE Decoder                        │
    ↓                         Guidance from
Final Image (RGB)             text embeddings
```

### 1.2 Detailed Architecture

#### Component 1: CLIP Text Encoder

```
Architecture: Transformer-based (12 layers)
Input: Text prompt (max 77 tokens)
Output: Text embeddings (77 × 768 dimensions)

Process:
1. Tokenization
   "A cat in a garden" → [49406, 320, 2368, 530, 320, 2010, 49407]
   (BPE tokenizer, vocabulary size: 49,408)

2. Token Embedding
   Each token ID → 768-dim vector

3. Positional Encoding
   Add position information (learned embeddings)

4. Transformer Layers (12 layers)
   For each layer:
     - Multi-head self-attention (12 heads, 64 dim each)
     - LayerNorm
     - Feed-forward network (768 → 3072 → 768)
     - Residual connections

5. Final Output
   Shape: [batch_size, 77, 768]

Example:
   Input: "A photorealistic cat"
   Output: [1, 77, 768] tensor
   → First few tokens are meaningful
   → Padding tokens for unused positions
```

**PyTorch Implementation:**
```python
import torch
from transformers import CLIPTokenizer, CLIPTextModel

# Load CLIP text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# Encode text
prompt = "A photorealistic fluffy orange cat in a vibrant garden"
tokens = tokenizer(
    prompt,
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt"
)

# Forward pass
with torch.no_grad():
    text_embeddings = text_encoder(tokens.input_ids)[0]
    # Shape: [1, 77, 768]

# These embeddings guide the diffusion process
```

---

#### Component 2: U-Net Denoiser

```
Architecture: Convolutional U-Net with attention layers
Input: Noisy latent (4 × 64 × 64) + timestep + text embeddings
Output: Predicted noise

U-Net Structure:
┌─────────────────────────────────────────────────────────┐
│                    INPUT (4×64×64)                       │
└───────────────────────┬─────────────────────────────────┘
                        ↓
    ┌──────────────────────────────────────┐
    │  Timestep Embedding (320-dim)        │
    │  sin/cos encoding of step t          │
    └──────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              ENCODER (Downsampling)                      │
├─────────────────────────────────────────────────────────┤
│  ResBlock 1 (4→320)    + SelfAttention + CrossAttention │
│  ResBlock 2 (320→320)  + SelfAttention + CrossAttention │
│  Downsample (320, 64×64 → 32×32)                        │
│                                                          │
│  ResBlock 3 (320→640)  + SelfAttention + CrossAttention │
│  ResBlock 4 (640→640)  + SelfAttention + CrossAttention │
│  Downsample (640, 32×32 → 16×16)                        │
│                                                          │
│  ResBlock 5 (640→1280) + SelfAttention + CrossAttention │
│  ResBlock 6 (1280→1280)+ SelfAttention + CrossAttention │
│  Downsample (1280, 16×16 → 8×8)                         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              BOTTLENECK (8×8)                            │
├─────────────────────────────────────────────────────────┤
│  ResBlock (1280)       + SelfAttention + CrossAttention │
│  ResBlock (1280)       + SelfAttention + CrossAttention │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              DECODER (Upsampling)                        │
├─────────────────────────────────────────────────────────┤
│  ResBlock (1280→1280) + SelfAttention + CrossAttention  │
│  Upsample (1280, 8×8 → 16×16)                           │
│  Skip connection from encoder ───────────────┐          │
│                                               ↓          │
│  ResBlock (1280→640)  + SelfAttention + CrossAttention  │
│  Upsample (640, 16×16 → 32×32)                          │
│  Skip connection from encoder ───────────────┐          │
│                                               ↓          │
│  ResBlock (640→320)   + SelfAttention + CrossAttention  │
│  Upsample (320, 32×32 → 64×64)                          │
│  Skip connection from encoder ───────────────┐          │
│                                               ↓          │
│  ResBlock (320→320)   + SelfAttention + CrossAttention  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              OUTPUT (4×64×64)                            │
│              Predicted noise                             │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**

**A. ResBlock (Residual Block)**
```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=320):
        super().__init__()

        # First conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, in_channels)

        # Time embedding projection
        self.time_emb = nn.Linear(time_emb_dim, out_channels)

        # Second conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, time_emb):
        # x: [B, C_in, H, W]
        # time_emb: [B, 320]

        h = self.norm1(x)
        h = F.silu(h)  # SiLU activation
        h = self.conv1(h)

        # Add time embedding
        time_emb = self.time_emb(F.silu(time_emb))
        h = h + time_emb[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        # Residual connection
        return h + self.skip(x)
```

**B. Cross-Attention (Text Conditioning)**
```python
class CrossAttention(nn.Module):
    def __init__(self, embed_dim=320, context_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query from image features
        self.q_proj = nn.Linear(embed_dim, embed_dim)

        # Key and Value from text embeddings
        self.k_proj = nn.Linear(context_dim, embed_dim)
        self.v_proj = nn.Linear(context_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, context):
        # x: image features [B, H*W, embed_dim]
        # context: text embeddings [B, 77, 768]

        B, N, C = x.shape

        # Compute Q, K, V
        q = self.q_proj(x)  # [B, H*W, embed_dim]
        k = self.k_proj(context)  # [B, 77, embed_dim]
        v = self.v_proj(context)  # [B, 77, embed_dim]

        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, C)

        return self.out_proj(out)

# This is how text controls image generation!
# Attention weights determine which words influence which image regions
```

---

#### Component 3: Diffusion Process (Denoising)

**Forward Diffusion (Training only):**
```
Clean image x₀ → Add noise gradually → Pure noise xₜ

At timestep t:
xₜ = √(αₜ) · x₀ + √(1 - αₜ) · ε
where ε ~ N(0, I) is Gaussian noise
```

**Reverse Diffusion (Inference - what we use):**
```
Pure noise x_T → Denoise step-by-step → Clean image x₀

DDPM Sampling (50 steps):
for t = T, T-1, ..., 1:
    1. U-Net predicts noise: ε_θ(xₜ, t, text_emb)
    2. Remove predicted noise:
       xₜ₋₁ = (xₜ - √(1-αₜ) · ε_θ) / √(αₜ) + σₜ · z
       where z ~ N(0, I) is added noise (except last step)
```

**Classifier-Free Guidance (CFG):**
```python
# Generate with guidance scale (typically 7.5)
def generate_with_cfg(latent, text_emb, cfg_scale=7.5):
    for t in range(T, 0, -1):
        # Unconditional prediction (no text)
        noise_uncond = unet(latent, t, empty_text_emb)

        # Conditional prediction (with text)
        noise_cond = unet(latent, t, text_emb)

        # Guided noise prediction
        noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)

        # Denoise one step
        latent = denoise_step(latent, noise_pred, t)

    return latent

# Higher CFG scale = stronger text guidance
# Too high (>15) = oversaturated, artifacts
# Too low (<3) = weak text following
```

---

#### Component 4: VAE Decoder

```
Architecture: Convolutional decoder with upsampling
Input: Latent (4 × 64 × 64)
Output: RGB image (3 × 512 × 512) for SD 2.1
        RGB image (3 × 1024 × 1024) for SD XL

Decoding Process:
┌────────────────────────────────┐
│ Input: Latent (4×64×64)        │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│ Conv (4 → 512 channels)        │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│ ResBlock (512)                 │
│ ResBlock (512)                 │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│ Upsample 2x (64×64 → 128×128)  │
│ ResBlock (512 → 512)           │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│ Upsample 2x (128×128 → 256×256)│
│ ResBlock (512 → 256)           │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│ Upsample 2x (256×256 → 512×512)│
│ ResBlock (256 → 128)           │
└───────────┬────────────────────┘
            ↓
┌────────────────────────────────┐
│ GroupNorm + SiLU               │
│ Conv (128 → 3)                 │
│ Output: (3×512×512)            │
└────────────────────────────────┘
```

**Latent Space Compression:**
```
Compression ratio: 8x in each spatial dimension
- Image 512×512 → Latent 64×64 (spatial)
- RGB (3 channels) → Latent (4 channels)
- Total compression: 512/64 × 512/64 × 3/4 = 48x

Benefits:
✓ 48x less memory
✓ 48x faster diffusion (fewer pixels to process)
✓ Semantic compression (preserves important features)
```

---

### 1.3 Complete Stable Diffusion Pipeline

```python
from diffusers import StableDiffusionPipeline
import torch

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# Enable optimizations
pipe.enable_attention_slicing()  # Reduce memory
pipe.enable_vae_slicing()        # Process VAE in slices

# Generate image
prompt = "A photorealistic fluffy orange cat in a vibrant garden, golden hour lighting, 8k, highly detailed"
negative_prompt = "blurry, low quality, cartoon, anime"

with torch.autocast("cuda"):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,      # Denoising steps
        guidance_scale=7.5,           # CFG scale
        width=1024,
        height=1024,
        generator=torch.Generator("cuda").manual_seed(42)
    ).images[0]

image.save("cat_in_garden.png")
```

**Step-by-Step Execution:**
```
Step 1: Text encoding (CLIP)
  Time: ~50ms
  Memory: 500 MB

Step 2: Generate random latent
  Shape: [1, 4, 128, 128] for 1024×1024
  Time: <1ms

Step 3: Denoising loop (30 steps)
  For each step:
    - U-Net forward pass: ~800ms per step
    - Noise prediction
    - Denoise latent
  Total: 30 × 800ms = 24 seconds

Step 4: VAE decode
  Time: ~2 seconds
  Output: [1, 3, 1024, 1024]

Total time: ~27 seconds on A100 GPU
Peak memory: 8-10 GB
```

---

## 2. VAE (VARIATIONAL AUTOENCODER)

### 2.1 Architecture Details

**VAE for Stable Diffusion:**
```
Encoder:
  Input: RGB image (3 × 512 × 512)
  ↓
  Conv layers with downsampling
  ↓
  Output: Latent distribution parameters
    - Mean μ: (4 × 64 × 64)
    - Log variance log(σ²): (4 × 64 × 64)

Latent Sampling:
  z = μ + σ · ε, where ε ~ N(0, I)
  (Reparameterization trick for backprop)

Decoder:
  Input: Latent z (4 × 64 × 64)
  ↓
  Conv layers with upsampling
  ↓
  Output: Reconstructed image (3 × 512 × 512)
```

### 2.2 VAE for Video Compression

**Temporal VAE Extension:**
```python
class TemporalVAE(nn.Module):
    """VAE with temporal compression for video"""

    def __init__(self):
        super().__init__()

        # Spatial encoder (same as image VAE)
        self.spatial_encoder = VAEEncoder()

        # Temporal encoder (3D convolutions)
        self.temporal_encoder = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.GroupNorm(2, 8),
            nn.SiLU(),
            nn.Conv3d(8, 4, kernel_size=(3,1,1), padding=(1,0,0))
        )

        # Temporal decoder
        self.temporal_decoder = nn.Sequential(
            nn.Conv3d(4, 8, kernel_size=(3,1,1), padding=(1,0,0)),
            nn.GroupNorm(2, 8),
            nn.SiLU(),
            nn.Conv3d(8, 4, kernel_size=(3,1,1), padding=(1,0,0))
        )

        # Spatial decoder
        self.spatial_decoder = VAEDecoder()

    def encode_video(self, video_frames):
        """
        video_frames: [B, T, 3, H, W] - batch of video sequences
        Returns: [B, T, 4, H//8, W//8]
        """
        B, T, C, H, W = video_frames.shape

        # Encode each frame spatially
        latents = []
        for t in range(T):
            frame_latent = self.spatial_encoder(video_frames[:, t])
            latents.append(frame_latent)

        latents = torch.stack(latents, dim=1)  # [B, T, 4, H//8, W//8]

        # Temporal compression
        latents = latents.permute(0, 2, 1, 3, 4)  # [B, 4, T, H//8, W//8]
        compressed = self.temporal_encoder(latents)

        return compressed

    def decode_video(self, compressed_latents):
        """
        compressed_latents: [B, 4, T, H//8, W//8]
        Returns: [B, T, 3, H, W]
        """
        # Temporal decompression
        latents = self.temporal_decoder(compressed_latents)
        latents = latents.permute(0, 2, 1, 3, 4)  # [B, T, 4, H//8, W//8]

        # Decode each frame
        B, T = latents.shape[:2]
        frames = []
        for t in range(T):
            frame = self.spatial_decoder(latents[:, t])
            frames.append(frame)

        return torch.stack(frames, dim=1)  # [B, T, 3, H, W]

# Usage for video compression
vae = TemporalVAE().cuda()

# Compress video (240 frames)
video = load_video()  # [1, 240, 3, 720, 1280]
compressed = vae.encode_video(video)  # [1, 4, 240, 90, 160]

# Compression ratio:
# Original: 240 × 3 × 720 × 1280 = 665M values
# Compressed: 4 × 240 × 90 × 160 = 13.8M values
# Ratio: 48x compression

# Decode back
reconstructed = vae.decode_video(compressed)
```

---

## 3. TEXT-TO-VIDEO PIPELINE

### 3.1 Complete Architecture

**Multi-Stage Video Generation:**
```
Stage 1: Keyframe Generation (Stable Diffusion)
Stage 2: Frame Interpolation (FILM/RIFE)
Stage 3: Temporal Refinement (VAE + Consistency)
Stage 4: Audio Sync (optional)
```

### 3.2 Stage 1: Keyframe Generation with Stable Diffusion

```python
class VideoKeyframeGenerator:
    """Generate video keyframes using Stable Diffusion"""

    def __init__(self):
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16
        ).to("cuda")

        # For temporal consistency
        self.use_same_seed = True
        self.seed_increment = 1

    def decompose_prompt(self, prompt, duration_frames):
        """
        Use LLM to decompose prompt into keyframe prompts

        Example:
        Input: "A sunset over mountains" (240 frames, 8 seconds)
        Output: [
            (0, "Mountains at dusk, purple sky, sun visible"),
            (80, "Sun touching horizon, orange and red colors"),
            (160, "Sun half-set, deep orange glow"),
            (240, "Stars appearing, dark blue sky")
        ]
        """
        # Use Llama 3 with chain-of-thought
        keyframe_prompts = llm_decompose(prompt, num_keyframes=4)

        # Distribute across timeline
        frames_per_keyframe = duration_frames // len(keyframe_prompts)
        keyframes = [
            (i * frames_per_keyframe, prompt)
            for i, prompt in enumerate(keyframe_prompts)
        ]

        return keyframes

    def generate_keyframes(self, prompt, duration=8, fps=30):
        """
        Generate keyframes for video

        Args:
            prompt: Text description of video
            duration: Video duration in seconds
            fps: Frames per second

        Returns:
            List of (frame_idx, image) tuples
        """
        total_frames = duration * fps

        # Decompose into keyframe prompts
        keyframe_specs = self.decompose_prompt(prompt, total_frames)

        # Generate each keyframe
        keyframes = []
        base_seed = 42

        for frame_idx, kf_prompt in keyframe_specs:
            # Use sequential seeds for consistency
            seed = base_seed + (frame_idx // 10)

            # Generate with same noise pattern for consistency
            generator = torch.Generator("cuda").manual_seed(seed)

            image = self.sd_pipeline(
                prompt=kf_prompt,
                num_inference_steps=40,  # Higher quality
                guidance_scale=7.5,
                generator=generator,
                width=1280,
                height=720
            ).images[0]

            keyframes.append((frame_idx, image))

        return keyframes

# Usage
gen = VideoKeyframeGenerator()
keyframes = gen.generate_keyframes(
    "A cat walking through a garden and stopping to smell flowers",
    duration=10,
    fps=30
)
# Returns 4 keyframes at frames 0, 75, 150, 225 (for 300 total frames)
```

### 3.3 Stage 2: Frame Interpolation (FILM)

**FILM (Frame Interpolation for Large Motion):**
```
Architecture: U-Net with optical flow estimation

Input: Two keyframes (I₀, I₁) + target time t ∈ [0, 1]
Output: Interpolated frame Iₜ

Process:
┌──────────────────────────────────────────┐
│ Keyframe 0 (t=0)     Keyframe 1 (t=1)    │
│     I₀                     I₁             │
└──────┬────────────────────┬───────────────┘
       │                    │
       ▼                    ▼
┌──────────────────────────────────────────┐
│   Feature Extraction (Shared Encoder)    │
│   - Conv layers                           │
│   - Extract multi-scale features          │
│   F₀ = [f₀₁, f₀₂, f₀₃, f₀₄]              │
│   F₁ = [f₁₁, f₁₂, f₁₃, f₁₄]              │
└──────┬────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│   Optical Flow Estimation                │
│   - Estimate motion from I₀ to I₁        │
│   - Forward flow: v₀→₁                    │
│   - Backward flow: v₁→₀                   │
└──────┬────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│   Flow Refinement Pyramid                │
│   - Refine flows at multiple scales      │
│   - Coarse to fine (4 levels)             │
└──────┬────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│   Frame Synthesis                         │
│   - Warp I₀ forward by t × v₀→₁          │
│   - Warp I₁ backward by (1-t) × v₁→₀     │
│   - Blend warped frames                   │
│   Iₜ = α(t)·warp(I₀, t·v₀→₁) +           │
│        (1-α(t))·warp(I₁, (1-t)·v₁→₀)     │
└──────┬────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│   Output: Interpolated frame Iₜ          │
└──────────────────────────────────────────┘
```

**Implementation:**
```python
import torch
import torch.nn.functional as F
from film_net import FILMModel  # Pre-trained FILM

class VideoInterpolator:
    def __init__(self):
        self.film_model = FILMModel.load_pretrained().cuda()
        self.film_model.eval()

    def interpolate_frames(self, frame0, frame1, num_intermediates=7):
        """
        Interpolate between two frames

        Args:
            frame0: First keyframe [3, H, W]
            frame1: Second keyframe [3, H, W]
            num_intermediates: Number of frames to generate between

        Returns:
            List of interpolated frames (length: num_intermediates + 2)
        """
        frames = [frame0]

        # Generate intermediate frames
        for i in range(1, num_intermediates + 1):
            t = i / (num_intermediates + 1)  # time in [0, 1]

            with torch.no_grad():
                # FILM takes both frames and time
                interpolated = self.film_model(
                    frame0.unsqueeze(0),
                    frame1.unsqueeze(0),
                    torch.tensor([t]).cuda()
                ).squeeze(0)

            frames.append(interpolated)

        frames.append(frame1)
        return frames

    def generate_full_video(self, keyframes, target_fps=30):
        """
        Generate full video from keyframes

        Args:
            keyframes: List of (frame_idx, image) tuples
            target_fps: Target frames per second

        Returns:
            Full video tensor [T, 3, H, W]
        """
        all_frames = []

        for i in range(len(keyframes) - 1):
            frame_idx0, img0 = keyframes[i]
            frame_idx1, img1 = keyframes[i + 1]

            # Convert PIL to tensor
            tensor0 = torch.from_numpy(np.array(img0)).permute(2, 0, 1).float() / 255.0
            tensor1 = torch.from_numpy(np.array(img1)).permute(2, 0, 1).float() / 255.0
            tensor0, tensor1 = tensor0.cuda(), tensor1.cuda()

            # Calculate how many frames between keyframes
            num_between = frame_idx1 - frame_idx0 - 1

            # Interpolate
            interpolated = self.interpolate_frames(
                tensor0, tensor1, num_intermediates=num_between
            )

            # Add all except last (to avoid duplication)
            all_frames.extend(interpolated[:-1])

        # Add final keyframe
        all_frames.append(tensor1)

        return torch.stack(all_frames)

# Complete video generation
keyframe_gen = VideoKeyframeGenerator()
interpolator = VideoInterpolator()

# Step 1: Generate keyframes
keyframes = keyframe_gen.generate_keyframes(
    "A sunset over mountains",
    duration=8,
    fps=30
)

# Step 2: Interpolate to full FPS
full_video = interpolator.generate_full_video(keyframes, target_fps=30)
# Shape: [240, 3, 720, 1280] - 8 seconds at 30 FPS
```

### 3.4 Stage 3: Temporal Consistency Refinement

**Problem:** Interpolated frames may have flickering or inconsistencies

**Solution:** Temporal consistency post-processing

```python
class TemporalConsistencyRefiner:
    """Enforce temporal consistency across video frames"""

    def __init__(self):
        # Load temporal VAE for consistency
        self.temporal_vae = TemporalVAE().cuda()
        self.temporal_vae.load_pretrained("temporal_vae_weights.pt")

    def apply_temporal_smoothing(self, video, window_size=5):
        """
        Apply temporal smoothing using sliding window

        Args:
            video: [T, 3, H, W]
            window_size: Number of frames to consider for smoothing

        Returns:
            Smoothed video [T, 3, H, W]
        """
        T = video.shape[0]
        smoothed = []

        for t in range(T):
            # Get window of frames
            start = max(0, t - window_size // 2)
            end = min(T, t + window_size // 2 + 1)
            window = video[start:end]

            # Encode to latent
            with torch.no_grad():
                latent = self.temporal_vae.encode_video(window.unsqueeze(0))

                # Decode back (smooths inconsistencies)
                smoothed_window = self.temporal_vae.decode_video(latent)

            # Take center frame
            center_idx = t - start
            smoothed.append(smoothed_window[0, center_idx])

        return torch.stack(smoothed)

    def detect_and_fix_flicker(self, video, threshold=0.1):
        """
        Detect and fix flickering frames

        Flickering detection: Large change between consecutive frames
        """
        T = video.shape[0]

        for t in range(1, T - 1):
            # Compute frame differences
            diff_prev = torch.abs(video[t] - video[t-1]).mean()
            diff_next = torch.abs(video[t] - video[t+1]).mean()

            # If current frame is outlier
            if diff_prev > threshold and diff_next > threshold:
                # Replace with blend of neighbors
                video[t] = 0.5 * (video[t-1] + video[t+1])

        return video

# Usage in video pipeline
refiner = TemporalConsistencyRefiner()

# After interpolation
smoothed_video = refiner.apply_temporal_smoothing(full_video, window_size=5)
final_video = refiner.detect_and_fix_flicker(smoothed_video)
```

---

*Continue in next file due to length...*
