# AI Model Architectures - Part 3

*Continuation from [Part 2](./ai-model-architectures-part2.md)*

---

## 7. VIDEO CAPTIONING MODELS

### 7.1 BLIP-2 Architecture

**BLIP-2 (Bootstrapping Language-Image Pre-training):**
```
Architecture: Vision Transformer + Q-Former + LLM

Components:
1. Frozen Vision Encoder (ViT-G/14 from CLIP)
2. Trainable Q-Former (Querying Transformer)
3. Frozen LLM (FlanT5-XL or OPT-6.7B)

Architecture:
┌────────────────────────────────────────────────────────────┐
│ Input: Image [3, 224, 224]                                  │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ Vision Encoder (ViT-G/14) - FROZEN                         │
│ Architecture:                                              │
│ - Patch embedding: 224×224 → 16×16 patches                 │
│ - Transformer: 40 layers, hidden_dim 1408                  │
│ - Output: [257, 1408] (1 CLS + 256 patches)                │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ Q-Former (Querying Transformer) - TRAINABLE                │
│                                                            │
│ Input:                                                     │
│ - 32 learnable query tokens [32, 768]                      │
│ - Image features from ViT [257, 1408]                      │
│                                                            │
│ Process:                                                   │
│ ┌──────────────────────────────────────┐                  │
│ │ Self-Attention Layer                 │                  │
│ │ - Queries attend to each other       │                  │
│ └──────────────────────────────────────┘                  │
│ ┌──────────────────────────────────────┐                  │
│ │ Cross-Attention Layer                │                  │
│ │ - Queries attend to image features   │                  │
│ │ - Extract relevant visual information│                  │
│ └──────────────────────────────────────┘                  │
│ ┌──────────────────────────────────────┐                  │
│ │ Feed-Forward Network                 │                  │
│ └──────────────────────────────────────┘                  │
│                                                            │
│ (12 layers total)                                          │
│                                                            │
│ Output: [32, 768] query representations                    │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ Linear Projection                                          │
│ Project Q-Former outputs to LLM dimension:                 │
│ [32, 768] → [32, 2048] (for FlanT5-XL)                     │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ Language Model (FlanT5-XL) - FROZEN                        │
│                                                            │
│ Input:                                                     │
│ - Visual tokens (from Q-Former): [32, 2048]                │
│ - Text prompt tokens: "Describe this image:"               │
│                                                            │
│ Architecture: T5 Encoder-Decoder                           │
│ - Encoder: Process visual + text tokens                    │
│ - Decoder: Generate caption autoregressively               │
│                                                            │
│ Output: Generated text tokens                              │
│ "A fluffy orange cat sitting in a garden..."               │
└────────────────────────────────────────────────────────────┘
```

**Implementation:**
```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from PIL import Image

class VideoCaptioner:
    def __init__(self):
        # Load BLIP-2
        self.processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-flan-t5-xl"
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=torch.float16
        ).to("cuda")

    def caption_image(self, image):
        """
        Generate caption for single image

        Args:
            image: PIL Image

        Returns:
            Caption string
        """
        # Preprocess
        inputs = self.processor(
            images=image,
            text="Describe this image in detail:",
            return_tensors="pt"
        ).to("cuda", torch.float16)

        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_length=100,
            num_beams=5,  # Beam search for better quality
            early_stopping=True
        )

        # Decode
        caption = self.processor.decode(
            generated_ids[0],
            skip_special_tokens=True
        )

        return caption

    def caption_video_keyframes(self, video_frames):
        """
        Caption video by processing keyframes

        Args:
            video_frames: List of PIL Images (keyframes)

        Returns:
            List of (frame_idx, caption) tuples
        """
        captions = []

        for idx, frame in enumerate(video_frames):
            caption = self.caption_image(frame)
            captions.append((idx, caption))

        return captions

# Usage
captioner = VideoCaptioner()

# Extract keyframes from video (1 per second)
keyframes = extract_keyframes("video.mp4", fps=1)

# Caption each keyframe
frame_captions = captioner.caption_video_keyframes(keyframes)

# Example output:
# [(0, "A person walking on a sunny street with trees"),
#  (1, "The person approaches a brown dog on a leash"),
#  (2, "The person bends down to pet the dog"),
#  (3, "The dog wags its tail happily")]
```

### 7.2 LLM-Based Temporal Aggregation

```python
def aggregate_video_captions(frame_captions, llm_model):
    """
    Use LLM to combine per-frame captions into coherent narrative

    Args:
        frame_captions: List of (timestamp, caption) tuples
        llm_model: Llama 3 or similar

    Returns:
        Comprehensive video description
    """

    # Format frame captions for LLM
    frames_text = "\n".join([
        f"Frame {idx} ({idx}s): {caption}"
        for idx, caption in frame_captions
    ])

    prompt = f"""You are watching a video. Here are descriptions of frames sampled every second:

{frames_text}

Based on these frame descriptions, write a comprehensive summary of the video that:
1. Describes the overall narrative or sequence of events
2. Identifies the main subjects and their actions
3. Notes any important transitions or changes
4. Maintains temporal coherence

Video summary:"""

    # Generate with LLM
    response = llm_model.generate(prompt, max_tokens=300)

    return response

# Example usage with vLLM
from vllm import LLM, SamplingParams

llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct")
sampling_params = SamplingParams(temperature=0.7, max_tokens=300)

# Aggregate frame captions
summary = aggregate_video_captions(frame_captions, llm)

# Output:
# "This video shows a person taking their dog for a walk on a sunny day.
#  The video begins with the person walking alone on a tree-lined street.
#  They then encounter a brown dog, presumably their pet, and stop to interact
#  with it. The person bends down to pet the dog, which responds enthusiastically
#  by wagging its tail. The overall tone is pleasant and the weather appears
#  to be clear and sunny."
```

---

## 8. AUDIO-TO-TEXT (WHISPER)

### 8.1 Whisper Architecture

**Whisper (OpenAI's Speech Recognition Model):**
```
Architecture: Encoder-Decoder Transformer
Model sizes: tiny (39M), base (74M), small (244M), medium (769M), large (1.55B)

Audio Processing Pipeline:
┌────────────────────────────────────────────────────────────┐
│ Raw Audio Waveform                                         │
│ Shape: [num_samples] @ 16kHz                               │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ Log-Mel Spectrogram                                        │
│ Process:                                                   │
│ 1. Compute STFT (Short-Time Fourier Transform)            │
│    - Window size: 400 samples (25ms)                       │
│    - Hop length: 160 samples (10ms)                        │
│ 2. Mel filterbank (80 mel bins)                            │
│ 3. Log scale                                               │
│                                                            │
│ Output shape: [80, T] where T = audio_length / 160         │
│ Example: 30s audio → [80, 3000]                            │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ ENCODER (Audio → Embeddings)                               │
│                                                            │
│ Architecture:                                              │
│ ┌──────────────────────────────────────┐                  │
│ │ Conv1d (80 → 384, kernel=3)          │                  │
│ │ GELU activation                      │                  │
│ │ Conv1d (384 → 384, kernel=3)         │                  │
│ │ GELU activation                      │                  │
│ └──────────────────────────────────────┘                  │
│ ┌──────────────────────────────────────┐                  │
│ │ Positional Encoding (learned)        │                  │
│ └──────────────────────────────────────┘                  │
│ ┌──────────────────────────────────────┐                  │
│ │ Transformer Encoder (24 layers)      │                  │
│ │ - Multi-head attention (6 heads)     │                  │
│ │ - Feed-forward (384 → 1536 → 384)    │                  │
│ │ - LayerNorm                          │                  │
│ └──────────────────────────────────────┘                  │
│                                                            │
│ Output: [T, 384] audio embeddings                          │
└───────────────────┬────────────────────────────────────────┘
                    ↓
┌────────────────────────────────────────────────────────────┐
│ DECODER (Embeddings → Text Tokens)                         │
│                                                            │
│ Architecture:                                              │
│ ┌──────────────────────────────────────┐                  │
│ │ Token Embedding (51865 vocabulary)   │                  │
│ │ + Positional Encoding                │                  │
│ └──────────────────────────────────────┘                  │
│ ┌──────────────────────────────────────┐                  │
│ │ Transformer Decoder (24 layers)      │                  │
│ │ - Masked self-attention              │                  │
│ │ - Cross-attention to encoder outputs │                  │
│ │ - Feed-forward network               │                  │
│ └──────────────────────────────────────┘                  │
│ ┌──────────────────────────────────────┐                  │
│ │ Linear (384 → 51865)                 │                  │
│ │ Softmax → Token probabilities        │                  │
│ └──────────────────────────────────────┘                  │
│                                                            │
│ Output: Text tokens (autoregressive)                       │
│ Example: [<|startoftranscript|>, Hello, how, are, ...]    │
└────────────────────────────────────────────────────────────┘
```

### 8.2 Multi-lingual & Task Support

**Special Tokens:**
```
Whisper uses special tokens to control behavior:

<|startoftranscript|>        - Start of transcription
<|en|>                       - Language (English)
<|transcribe|>               - Task: transcription
<|translate|>                - Task: translate to English
<|notimestamps|>             - No timestamps
<|0.00|>                     - Timestamp (0.00s)
<|nospeech|>                 - No speech detected
<|endoftext|>                - End of transcription

Example output tokens:
[<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>,
 Hello, comma, how, are, you, question, <|endoftext|>]

With timestamps:
[<|startoftranscript|>, <|en|>, <|transcribe|>,
 <|0.00|>, Hello, <|0.50|>, how, are, you, <|1.20|>, <|endoftext|>]
```

### 8.3 Implementation

```python
import whisper
import torch
import numpy as np

class AudioTranscriber:
    def __init__(self, model_size="large-v3"):
        """
        Initialize Whisper model

        Args:
            model_size: tiny, base, small, medium, large, large-v2, large-v3
        """
        self.model = whisper.load_model(model_size, device="cuda")

    def transcribe_audio(self, audio_path, language=None, task="transcribe"):
        """
        Transcribe audio file

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., "en", "es", "fr") or None for auto-detect
            task: "transcribe" or "translate" (translate to English)

        Returns:
            Dict with transcription and metadata
        """
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=False,
            fp16=True  # Use FP16 for faster inference on GPU
        )

        return result

    def transcribe_with_timestamps(self, audio_path):
        """
        Transcribe with word-level timestamps

        Returns:
            List of segments with timestamps
        """
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            fp16=True
        )

        # Extract segments
        segments = []
        for segment in result['segments']:
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'words': segment.get('words', [])
            })

        return segments

    def transcribe_video_audio(self, video_path):
        """
        Extract audio from video and transcribe

        Args:
            video_path: Path to video file

        Returns:
            Transcription with timestamps
        """
        # Extract audio using ffmpeg
        audio_path = self._extract_audio(video_path)

        # Transcribe
        segments = self.transcribe_with_timestamps(audio_path)

        return segments

    def _extract_audio(self, video_path):
        """Extract audio from video using ffmpeg"""
        import subprocess
        import tempfile

        # Create temp file for audio
        audio_path = tempfile.mktemp(suffix=".wav")

        # Extract audio (16kHz mono WAV)
        subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",      # Mono
            "-vn",           # No video
            audio_path
        ], check=True, capture_output=True)

        return audio_path

# Usage
transcriber = AudioTranscriber(model_size="large-v3")

# Simple transcription
result = transcriber.transcribe_audio("speech.mp3")
print(result['text'])
# Output: "Hello, how are you today? I hope you're doing well."

# With timestamps
segments = transcriber.transcribe_with_timestamps("speech.mp3")
for seg in segments:
    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}")
# Output:
# [0.00s - 1.50s]: Hello, how are you today?
# [1.50s - 3.20s]: I hope you're doing well.

# Video transcription
video_segments = transcriber.transcribe_video_audio("video.mp4")
```

### 8.4 Audio-Video Synchronization for Captions

```python
def generate_srt_subtitles(video_path, output_path):
    """
    Generate SRT subtitle file from video

    Args:
        video_path: Path to video
        output_path: Path to save SRT file
    """
    transcriber = AudioTranscriber()

    # Transcribe video audio
    segments = transcriber.transcribe_video_audio(video_path)

    # Generate SRT format
    srt_content = []
    for idx, seg in enumerate(segments, start=1):
        # Format timestamps (SRT format: HH:MM:SS,mmm)
        start_time = format_srt_time(seg['start'])
        end_time = format_srt_time(seg['end'])

        # Add subtitle entry
        srt_content.append(f"{idx}")
        srt_content.append(f"{start_time} --> {end_time}")
        srt_content.append(seg['text'])
        srt_content.append("")  # Blank line

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(srt_content))

def format_srt_time(seconds):
    """Convert seconds to SRT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# Usage
generate_srt_subtitles("video.mp4", "subtitles.srt")

# subtitles.srt content:
# 1
# 00:00:00,000 --> 00:00:01,500
# Hello, how are you today?
#
# 2
# 00:00:01,500 --> 00:00:03,200
# I hope you're doing well.
```

---

## 9. GPU OPTIMIZATION TECHNIQUES

### 9.1 Mixed Precision Training & Inference (FP16)

**Automatic Mixed Precision (AMP):**
```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Training with mixed precision
model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Enable autocasting for forward pass
    with autocast():
        output = model(batch)
        loss = criterion(output, target)

    # Scale loss and backprop
    scaler.scale(loss).backward()

    # Unscale gradients and step optimizer
    scaler.step(optimizer)
    scaler.update()

# Inference with FP16
model.eval()
model = model.half()  # Convert to FP16

with torch.no_grad(), autocast():
    output = model(input.half())

# Benefits:
# - 2x faster computation (on Tensor Cores)
# - 50% less memory
# - Minimal accuracy loss (<1% typically)
```

### 9.2 Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint

class EfficientTransformer(nn.Module):
    """Transformer with gradient checkpointing"""

    def __init__(self, num_layers=24):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer() for _ in range(num_layers)
        ])
        self.use_checkpointing = True

    def forward(self, x):
        for layer in self.layers:
            if self.training and self.use_checkpointing:
                # Checkpoint: don't store intermediate activations
                # Recompute during backward pass
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x

# Benefits:
# - Reduce memory by ~N (number of layers)
# - Trade memory for computation (20-30% slower)
# - Enable larger batch sizes or models
```

### 9.3 Flash Attention 2

```python
from flash_attn import flash_attn_qkvpacked_func

def flash_attention_forward(q, k, v):
    """
    Flash Attention: Memory-efficient attention

    Standard attention:
    - Memory: O(N²) for attention matrix
    - Computation: 3 passes (Q@K, softmax, @V)

    Flash Attention:
    - Memory: O(N) - never materialize full attention matrix
    - Computation: Fused kernel, fewer memory accesses
    - Speed: 2-4x faster
    """

    # Combine Q, K, V for fused kernel
    qkv = torch.stack([q, k, v], dim=2)  # [batch, seqlen, 3, num_heads, head_dim]

    # Flash attention (single fused CUDA kernel)
    output = flash_attn_qkvpacked_func(
        qkv,
        dropout_p=0.0,
        causal=True  # For autoregressive models
    )

    return output

# Integration with models
class FlashAttentionLayer(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)

        # Flash attention
        attn_output = flash_attn_qkvpacked_func(qkv, causal=True)

        # Reshape and project
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        return self.out_proj(attn_output)

# Benefits:
# - 40-50% faster than standard attention
# - O(N) memory instead of O(N²)
# - Exact same results (mathematically identical)
```

### 9.4 Model Quantization

**INT8 Quantization with BitsAndBytes:**
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Quantization config
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

# Load model in INT8
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=quant_config,
    device_map="auto"
)

# Model size:
# - FP32: 32 GB
# - FP16: 16 GB
# - INT8:  8 GB
# Speed: ~80% of FP16 (still faster than FP32)
# Quality: ~98% of FP16 (minimal degradation)
```

**4-bit Quantization (GPTQ/AWQ):**
```python
# Even more aggressive quantization
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"  # Normal Float 4-bit
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B",
    quantization_config=quant_config,
    device_map="auto"
)

# Benefits:
# - 70B model fits in 40GB GPU (vs 140GB in FP16)
# - 50-60% of FP16 speed
# - 95-97% of FP16 quality
```

---

*Continued in final section...*
