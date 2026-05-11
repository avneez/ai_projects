# AI Model Architectures - Part 2

*Continuation from [Part 1](./ai-model-architectures.md)*

---

## 4. LLM ARCHITECTURE (LLAMA 3)

### 4.1 Transformer Architecture

**Llama 3 (8B parameters):**
```
Architecture: Decoder-only Transformer
Layers: 32
Hidden dim: 4096
Attention heads: 32
Intermediate size (FFN): 14336
Vocabulary: 128,256 tokens
Context length: 8,192 tokens

Model Structure:
┌────────────────────────────────────────┐
│ Input: Token IDs                       │
│ "Generate a detailed prompt for: cat" │
│ → [Generate, a, detailed, prompt, ...]│
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ Token Embedding (128256 → 4096)        │
│ + Rotary Position Embedding (RoPE)     │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ Transformer Layer 1 of 32              │
├────────────────────────────────────────┤
│  RMSNorm                                │
│  ↓                                     │
│  Multi-Head Attention (32 heads)       │
│  - Query, Key, Value projections       │
│  - Grouped-Query Attention (GQA)       │
│  - 8 KV heads, 32 Q heads              │
│  ↓                                     │
│  Residual connection                   │
│  ↓                                     │
│  RMSNorm                                │
│  ↓                                     │
│  Feed-Forward Network (SwiGLU)         │
│  - Linear (4096 → 14336)                │
│  - SwiGLU activation                    │
│  - Linear (14336 → 4096)                │
│  ↓                                     │
│  Residual connection                   │
└───────────┬────────────────────────────┘
            ↓
        (Repeat 31 more times)
            ↓
┌────────────────────────────────────────┐
│ Final RMSNorm                           │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ LM Head (4096 → 128256)                 │
│ Output: Next token logits              │
└────────────────────────────────────────┘
```

### 4.2 Key Components

**A. Grouped-Query Attention (GQA)**
```python
class GroupedQueryAttention(nn.Module):
    """
    Llama 3's efficient attention mechanism
    - 32 query heads
    - 8 key-value heads
    - Each KV head is shared across 4 query heads
    """

    def __init__(self, hidden_size=4096, num_heads=32, num_kv_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        # Query projection (full 32 heads)
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)

        # Key/Value projections (only 8 heads)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)

        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, position_ids):
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = self.q_proj(hidden_states)  # [B, L, 32*128]
        k = self.k_proj(hidden_states)  # [B, L, 8*128]
        v = self.v_proj(hidden_states)  # [B, L, 8*128]

        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply Rotary Position Embedding (RoPE)
        q, k = apply_rotary_pos_emb(q, k, position_ids)

        # Repeat KV heads to match Q heads (8 → 32)
        k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=2)

        # Transpose for attention: [B, num_heads, L, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention computation
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask (for autoregressive generation)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        return self.o_proj(attn_output)
```

**B. SwiGLU Feed-Forward Network**
```python
class SwiGLU_FFN(nn.Module):
    """
    Swish-Gated Linear Unit
    More expressive than standard ReLU FFN
    """

    def __init__(self, hidden_size=4096, intermediate_size=14336):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU: swish(gate(x)) ⊙ up(x)
        gate = F.silu(self.gate_proj(x))  # SiLU = Swish
        up = self.up_proj(x)
        return self.down_proj(gate * up)
```

### 4.3 Usage for Prompt Enhancement

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load Llama 3 8B
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

def enhance_prompt_with_cot(user_prompt):
    """
    Use Llama 3 with Chain-of-Thought to enhance image generation prompt
    """

    system_prompt = """You are an expert at writing Stable Diffusion prompts.
Given a simple description, you will think step-by-step to create a detailed,
high-quality prompt that will generate beautiful images.

Think about:
1. Subject details (appearance, colors, textures)
2. Environment and setting
3. Lighting and atmosphere
4. Art style and quality modifiers
5. Camera angle and composition"""

    user_message = f"""Create a detailed Stable Diffusion prompt for: "{user_prompt}"

Think step-by-step, then provide the final enhanced prompt."""

    # Format chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to("cuda")

    # Generate with sampling
    outputs = model.generate(
        inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

    return response

# Example usage
user_input = "A cat in a garden"
enhanced = enhance_prompt_with_cot(user_input)

print(enhanced)
# Output example:
# "Let me think step-by-step:
#  1. Subject: A fluffy orange tabby cat with green eyes
#  2. Setting: A vibrant English garden with roses and tulips
#  3. Lighting: Golden hour sunlight, soft shadows
#  4. Style: Photorealistic, professional photography
#  5. Composition: Cat in foreground, garden background, shallow depth of field
#
#  Final prompt: A photorealistic fluffy orange tabby cat with bright green eyes
#  sitting peacefully in a vibrant English garden filled with red roses and yellow
#  tulips, golden hour lighting with soft warm shadows, professional wildlife
#  photography, shallow depth of field, 8k resolution, highly detailed fur texture,
#  sharp focus on cat's face, bokeh background"
```

### 4.4 vLLM Serving for Fast Inference

```python
from vllm import LLM, SamplingParams

# Initialize vLLM engine
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=1,  # Number of GPUs
    dtype="float16",
    max_model_len=8192,
    gpu_memory_utilization=0.9
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=512
)

# Batch inference (much faster than sequential)
prompts = [
    "Enhance: A cat in a garden",
    "Enhance: A sunset over mountains",
    "Enhance: A futuristic city"
]

outputs = llm.generate(prompts, sampling_params)

# Process results
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print("---")

# vLLM benefits:
# - PagedAttention: 3-4x higher throughput
# - Continuous batching: Process requests as they arrive
# - KV cache optimization: Reuse computations
```

---

## 5. AUDIO GENERATION MODELS

### 5.1 MusicGen Architecture

**MusicGen (Meta's music generation model):**
```
Architecture: Transformer with audio tokenization
Model size: 3.3B parameters (large version)

Audio Generation Pipeline:
┌────────────────────────────────────────┐
│ Text Prompt: "Epic orchestral music"  │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ T5 Text Encoder                        │
│ - Encode text to embeddings            │
│ - Output: [batch, seq_len, 1024]      │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ Audio Tokens (EnCodec)                 │
│ - Start with special <BOS> token      │
│ - 4 codebooks (hierarchical)           │
│ - Each codebook: 2048 tokens           │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ Transformer Decoder (24 layers)       │
│ - Autoregressive generation            │
│ - Conditioned on text embeddings       │
│ - Generates tokens for all 4 codebooks │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ EnCodec Decoder                        │
│ - Decode tokens to waveform            │
│ - Sample rate: 32kHz stereo            │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ Output: Audio waveform                 │
│ Shape: [2, num_samples]                │
└────────────────────────────────────────┘
```

### 5.2 EnCodec Audio Tokenization

**EnCodec: Neural Audio Codec**
```python
from audiocraft.models import MusicGen
import torch
import torchaudio

class MusicGenPipeline:
    def __init__(self):
        # Load MusicGen large (3.3B parameters)
        self.model = MusicGen.get_pretrained('facebook/musicgen-large')
        self.model.set_generation_params(
            duration=30,  # 30 seconds
            temperature=1.0,
            top_k=250,
            top_p=0.0
        )

    def generate_music(self, prompt, duration=30):
        """
        Generate music from text prompt

        Args:
            prompt: Text description of music
            duration: Length in seconds

        Returns:
            Waveform tensor [channels, samples]
        """
        self.model.set_generation_params(duration=duration)

        # Generate
        with torch.no_grad():
            wav = self.model.generate([prompt], progress=True)

        # wav shape: [batch, channels, samples]
        # For 30s at 32kHz: [1, 2, 960000]

        return wav[0]  # Return first batch item

    def generate_with_melody_conditioning(self, prompt, melody_path, duration=30):
        """
        Generate music conditioned on a melody

        Args:
            prompt: Text description
            melody_path: Path to melody audio file
            duration: Duration in seconds

        Returns:
            Generated audio
        """
        # Load melody
        melody, sr = torchaudio.load(melody_path)

        # Resample to 32kHz if needed
        if sr != 32000:
            resampler = torchaudio.transforms.Resample(sr, 32000)
            melody = resampler(melody)

        # Generate conditioned on melody
        wav = self.model.generate_with_chroma(
            descriptions=[prompt],
            melody_wavs=melody.unsqueeze(0),
            melody_sample_rate=32000,
            progress=True
        )

        return wav[0]

# Usage
music_gen = MusicGenPipeline()

# Generate music from text
audio = music_gen.generate_music(
    "Epic orchestral music with violins and drums, intense battle theme",
    duration=30
)

# Save
torchaudio.save("epic_music.wav", audio, 32000)
```

**EnCodec Architecture:**
```
Encoder (Audio → Tokens):
  Waveform [2, T] → Conv layers → Latent [D, T'] → Quantization → Tokens

  Quantization uses Residual Vector Quantization (RVQ):
  - 4 codebooks
  - Each codebook: 2048 tokens
  - Hierarchical: First codebook captures main structure,
                  later codebooks add details

Decoder (Tokens → Audio):
  Tokens → Dequantization → Latent → ConvTranspose → Waveform

Compression ratio:
  32kHz audio → 50 Hz tokens (640x compression)
  30 seconds = 960,000 samples → 1,500 tokens (per codebook)
```

---

## 6. GAN ARCHITECTURE FOR AVATAR GENERATION

### 6.1 StyleGAN3 for Face Generation

**StyleGAN3 Architecture:**
```
Generator:
┌────────────────────────────────────────────────────────────┐
│ Input: Random latent z ∈ R^512                             │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ Mapping Network (8-layer MLP)                              │
│ z → w ∈ R^512 (disentangled latent space)                  │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ Learned Constant (4×4×512)                                 │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ Synthesis Network (Resolution: 4×4 → 1024×1024)            │
│                                                            │
│ For each resolution level (4, 8, 16, 32, 64, 128, 256, ...│
│   ┌──────────────────────────────────────────┐            │
│   │ Modulated Convolution                    │            │
│   │ - Style modulation from w                │            │
│   │ - Conv 3×3                                 │            │
│   │ - Add noise                                │            │
│   │ - LeakyReLU                                │            │
│   └──────────────────────────────────────────┘            │
│   ┌──────────────────────────────────────────┐            │
│   │ Modulated Convolution (second)           │            │
│   └──────────────────────────────────────────┘            │
│   ┌──────────────────────────────────────────┐            │
│   │ Upsample 2x                               │            │
│   └──────────────────────────────────────────┘            │
│                                                            │
│ Final: ToRGB layer → [3, 1024, 1024]                       │
└────────────────────────────────────────────────────────────┘

Discriminator:
┌────────────────────────────────────────────────────────────┐
│ Input: Real or Generated image [3, 1024, 1024]             │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ FromRGB → Convolutions with downsampling                   │
│ 1024×1024 → 512×512 → 256×256 → ... → 4×4                  │
│                                                            │
│ Each level:                                                │
│ - Conv 3×3                                                  │
│ - LeakyReLU                                                │
│ - Downsample                                               │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ Output: Real/Fake score (scalar)                           │
└────────────────────────────────────────────────────────────┘
```

### 6.2 Text-Controlled Face Generation

```python
import torch
from stylegan3 import Generator
from clip import load as load_clip

class TextControlledFaceGenerator:
    """
    Generate faces conditioned on text descriptions using StyleGAN3 + CLIP
    """

    def __init__(self):
        # Load StyleGAN3 generator (pre-trained on FFHQ)
        self.generator = Generator(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=1024,
            img_channels=3
        ).cuda()
        self.generator.load_state_dict(torch.load('stylegan3-ffhq-1024x1024.pt'))
        self.generator.eval()

        # Load CLIP for text guidance
        self.clip_model, self.clip_preprocess = load_clip("ViT-B/32", device="cuda")

    def text_to_face(self, text_prompt, num_optimization_steps=300):
        """
        Generate face from text description using CLIP-guided optimization

        Process:
        1. Start with random latent z
        2. Generate image with StyleGAN3
        3. Compute CLIP similarity between image and text
        4. Optimize z to maximize similarity
        5. Return final generated image

        Args:
            text_prompt: Text description (e.g., "A young woman with blonde hair and blue eyes")
            num_optimization_steps: Number of optimization iterations

        Returns:
            Generated face image [3, 1024, 1024]
        """
        # Encode text with CLIP
        text_tokens = clip.tokenize([text_prompt]).cuda()
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Initialize random latent
        z = torch.randn(1, 512, device='cuda', requires_grad=True)

        # Optimizer
        optimizer = torch.optim.Adam([z], lr=0.1)

        # Optimization loop
        for step in range(num_optimization_steps):
            optimizer.zero_grad()

            # Generate image from current latent
            with torch.no_grad():
                w = self.generator.mapping(z, None)  # z → w

            # Forward through synthesis network
            image = self.generator.synthesis(w, noise_mode='const')

            # Resize and normalize for CLIP
            image_resized = F.interpolate(image, size=224, mode='bicubic')
            image_normalized = (image_resized + 1) / 2  # [-1, 1] → [0, 1]

            # Encode image with CLIP
            image_features = self.clip_model.encode_image(image_normalized)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Compute CLIP loss (negative cosine similarity)
            similarity = (image_features * text_features).sum()
            loss = -similarity

            # Also add identity regularization (keep face realistic)
            # (StyleGAN3 naturally generates realistic faces, so light regularization)
            latent_penalty = 0.1 * (z ** 2).sum()
            total_loss = loss + latent_penalty

            # Backprop and update
            total_loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print(f"Step {step}: Similarity = {similarity.item():.4f}")

        # Generate final image
        with torch.no_grad():
            w_final = self.generator.mapping(z, None)
            final_image = self.generator.synthesis(w_final, noise_mode='const')

        return final_image[0]  # Return [3, 1024, 1024]

# Usage
face_gen = TextControlledFaceGenerator()

# Generate face from text
face = face_gen.text_to_face(
    "A young woman with long blonde hair, blue eyes, and a gentle smile",
    num_optimization_steps=300
)

# Save
save_image(face, "generated_face.png", normalize=True, range=(-1, 1))
```

### 6.3 3D Avatar Generation from 2D Face

**PIFuHD (Pixel-Aligned Implicit Function in High Resolution):**
```
Architecture: Image-based 3D reconstruction

Input: 2D face image [3, 1024, 1024]
Output: 3D mesh with texture

Process:
┌────────────────────────────────────────┐
│ 1. Multi-View Generation               │
│    Generate 4 views using StyleGAN3:   │
│    - Front view (0°)                    │
│    - Left side (90°)                    │
│    - Right side (-90°)                  │
│    - Back view (180°)                   │
│                                        │
│    Technique: Latent space manipulation│
│    w_side = w_front + direction_vector │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ 2. Image Feature Extraction            │
│    Use CNN to extract features from    │
│    each view:                          │
│    - ResNet-50 backbone                │
│    - Multi-scale features              │
│    F_front, F_left, F_right, F_back    │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ 3. 3D Point Sampling                   │
│    For each 3D point p = (x, y, z):    │
│    - Project to each view              │
│    - Sample features at projection     │
│    - Concatenate multi-view features   │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ 4. Occupancy Prediction (MLP)          │
│    Input: Point p + Multi-view features│
│    Output: Occupancy score [0, 1]      │
│            (inside surface vs outside) │
│                                        │
│    MLP (8 layers):                     │
│    [features + xyz] → 512 → 512 → ... │
│                     → 1 (occupancy)    │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ 5. Mesh Extraction                     │
│    Use Marching Cubes algorithm:       │
│    - Sample occupancy on 256³ grid     │
│    - Extract isosurface at 0.5         │
│    - Output: Vertices + Faces          │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ 6. Texture Mapping                     │
│    Project each vertex to source image │
│    Sample RGB color                    │
│    Create UV texture map               │
└───────────┬────────────────────────────┘
            ↓
┌────────────────────────────────────────┐
│ Output: 3D Mesh (OBJ/FBX)              │
│ - Vertices: ~50k                       │
│ - Faces: ~100k                         │
│ - Texture: 2048×2048 PNG               │
└────────────────────────────────────────┘
```

**Implementation:**
```python
import torch
from pifuhd import PIFuHD
from marching_cubes import marching_cubes
import trimesh

class Avatar3DGenerator:
    def __init__(self):
        self.face_generator = TextControlledFaceGenerator()
        self.pifuhd = PIFuHD().cuda()
        self.pifuhd.load_pretrained()

    def generate_3d_avatar(self, text_description):
        """
        Complete pipeline: Text → 2D Face → 3D Mesh

        Args:
            text_description: Text prompt for face

        Returns:
            3D mesh object
        """
        # Step 1: Generate 2D face from text
        print("Generating 2D face...")
        face_front = self.face_generator.text_to_face(text_description)

        # Step 2: Generate other views (latent space rotation)
        print("Generating multi-view images...")
        views = self._generate_multiview(face_front)

        # Step 3: 3D reconstruction with PIFuHD
        print("Reconstructing 3D mesh...")
        mesh = self._reconstruct_3d(views)

        # Step 4: Apply textures
        print("Applying textures...")
        textured_mesh = self._apply_texture(mesh, views)

        return textured_mesh

    def _generate_multiview(self, front_view):
        """Generate multiple views from front view"""
        # This is simplified - actual implementation uses
        # latent space manipulation or NeRF
        views = {
            'front': front_view,
            # Generate other views using 3D-aware GAN or NeRF
        }
        return views

    def _reconstruct_3d(self, views):
        """Reconstruct 3D mesh from multiple views"""
        # Sample 3D grid
        resolution = 256
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        z = torch.linspace(-1, 1, resolution)
        grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)

        # Predict occupancy for each point
        batch_size = 10000
        occupancies = []

        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size].cuda()

            with torch.no_grad():
                occ = self.pifuhd.query_occupancy(batch, views)

            occupancies.append(occ.cpu())

        occupancies = torch.cat(occupancies).reshape(resolution, resolution, resolution)

        # Extract mesh using marching cubes
        vertices, faces = marching_cubes(
            occupancies.numpy(),
            level=0.5,
            spacing=(2.0/resolution, 2.0/resolution, 2.0/resolution)
        )

        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        return mesh

    def _apply_texture(self, mesh, views):
        """Apply texture to mesh from multi-view images"""
        # Project each vertex to nearest view
        # Sample color from image
        # Create UV texture map

        # ... texture mapping logic ...

        return mesh

# Usage
avatar_gen = Avatar3DGenerator()

# Generate 3D avatar from text
avatar_mesh = avatar_gen.generate_3d_avatar(
    "A young man with short brown hair and glasses, friendly expression"
)

# Save as FBX
avatar_mesh.export("avatar.fbx")
```

---

*Continued in part 3...*
