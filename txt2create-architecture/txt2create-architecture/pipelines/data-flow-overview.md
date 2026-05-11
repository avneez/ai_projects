# Data Flow Overview - TXT2CREATE Pipelines

## Common Request Flow Pattern

All pipelines follow a similar pattern with variations in processing:

```
User Request → API Gateway → FastAPI → Task Queue → Pipeline Worker → AI Model → Post-Processing → Storage → Response
```

## Pipeline Index

1. [Text-to-Image Pipeline](#1-text-to-image-pipeline)
2. [Text-to-Video Pipeline](#2-text-to-video-pipeline)
3. [Text-to-Audio Pipeline](#3-text-to-audio-pipeline)
4. [Video Captioning Pipeline](#4-video-captioning-pipeline)
5. [Virtual Avatar Generation Pipeline](#5-virtual-avatar-generation-pipeline)

---

## 1. Text-to-Image Pipeline

### Data Flow

```
┌──────────────┐
│ User Input   │
│ "A cat in    │
│  a garden"   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│ FastAPI Endpoint                     │
│ POST /api/v1/generate/image          │
│ - Validate input                     │
│ - Check rate limits                  │
│ - Generate job ID                    │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ LLM Chain-of-Thought (vLLM)          │
│ - Prompt enhancement                 │
│ - Negative prompt generation         │
│ - Style parameters suggestion        │
│ Input: "A cat in a garden"           │
│ Output: "A photorealistic fluffy     │
│ orange cat sitting in a vibrant      │
│ garden with roses, professional      │
│ photography, 8k, detailed..."        │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Redis Cache Check                    │
│ Key: hash(enhanced_prompt + params)  │
│ - If HIT: Return cached image URL    │
│ - If MISS: Continue to generation    │
└──────┬───────────────────────────────┘
       │ (Cache Miss)
       ▼
┌──────────────────────────────────────┐
│ Celery Task Queue                    │
│ Task: generate_image_task            │
│ Priority: NORMAL                     │
│ ETA: 30-60 seconds                   │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Celery Worker (Image Pipeline)       │
│ 1. Dequeue task                      │
│ 2. Load parameters                   │
│ 3. Route to TorchServe               │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ TorchServe - Stable Diffusion        │
│ Model: SD XL 1.0 / SD 2.1            │
│ Hardware: 1x A100 GPU (40GB)         │
│ Process:                             │
│ 1. Text encoding (CLIP)              │
│ 2. Latent diffusion (U-Net)          │
│ 3. Denoising steps (25-50)           │
│ 4. VAE decoding to pixels            │
│ Output: 1024x1024 PNG (latent→RGB)   │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Post-Processing                      │
│ 1. Format conversion (PNG/JPEG/WEBP) │
│ 2. Compression (quality: 85)         │
│ 3. Watermark (optional)              │
│ 4. Metadata embedding (EXIF)         │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Storage Layer                        │
│ 1. S3/MinIO upload                   │
│    Path: /images/{user_id}/{job_id}  │
│ 2. PostgreSQL record                 │
│    - job_id, user_id, prompt         │
│    - s3_url, created_at, params      │
│ 3. Redis cache (24h TTL)             │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ WebSocket Notification               │
│ Event: "generation_complete"         │
│ Payload: {                           │
│   job_id, status: "completed",       │
│   image_url, thumbnail_url           │
│ }                                    │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────┐
│ User Receives│
│ Image URL    │
└──────────────┘
```

### Performance Metrics
- **Latency**: 30-60 seconds (depends on steps)
- **Throughput**: 20-30 images/minute per GPU
- **Cache Hit Rate**: ~40% for popular prompts

---

## 2. Text-to-Video Pipeline

### Data Flow

```
┌──────────────┐
│ User Input   │
│ "Sunset over│
│  mountains"  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│ FastAPI Endpoint                     │
│ POST /api/v1/generate/video          │
│ - Validate (max duration: 10s)       │
│ - Estimate GPU time & cost           │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ LLM Chain-of-Thought Pipeline        │
│ Step 1: Scene Decomposition          │
│ Input: "Sunset over mountains"       │
│ Output: {                            │
│   frames: [                          │
│     "Mountains at dusk, purple sky", │
│     "Sun touching horizon, orange",  │
│     "Stars appearing, dark blue"     │
│   ],                                 │
│   transitions: "smooth_fade",        │
│   duration: 8                        │
│ }                                    │
│                                      │
│ Step 2: Prompt Enhancement per Frame │
│ Enhance each frame prompt with       │
│ cinematic parameters                 │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Celery Task Queue (HIGH Priority)    │
│ Task: generate_video_task            │
│ ETA: 3-5 minutes                     │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Video Pipeline Worker                │
│ Strategy: Keyframe Interpolation     │
│                                      │
│ 1. Generate Keyframes (SD)           │
│    - Frame 0, 30, 60, ... 240        │
│    - TorchServe SD model             │
│                                      │
│ 2. Frame Interpolation               │
│    - FILM or RIFE model              │
│    - Generate in-between frames      │
│    - Target: 30 FPS                  │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ VAE Video Compression                │
│ Model: Custom VAE (TorchServe)       │
│ Process:                             │
│ 1. Encode frames to latent space     │
│    - 240 frames → 240 latents        │
│    - Compression ratio: 8x           │
│ 2. Temporal compression              │
│    - 3D convolutions                 │
│    - Motion vectors                  │
│ 3. Decode to video format            │
│    - H.264 codec                     │
│    - CRF 23 (quality)                │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Post-Processing                      │
│ 1. Audio addition (optional)         │
│    - BGM from text-to-audio pipeline │
│ 2. Subtitle generation               │
│ 3. Format conversion (MP4/WebM)      │
│ 4. Thumbnail extraction              │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Storage & Delivery                   │
│ 1. S3 upload (video + thumbnail)     │
│ 2. CDN distribution (CloudFlare)     │
│ 3. Database record creation          │
│ 4. WebSocket notification            │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────┐
│ User Receives│
│ Video URL    │
└──────────────┘
```

### Technical Details
- **Resolution**: 720p (1280x720) or 1080p
- **Frame Rate**: 30 FPS
- **Duration**: 3-10 seconds
- **File Size**: 5-20 MB (after compression)
- **Processing Time**: 3-5 minutes

---

## 3. Text-to-Audio Pipeline

### Data Flow

```
┌──────────────┐
│ User Input   │
│ "Epic battle│
│  music"     │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│ FastAPI Endpoint                     │
│ POST /api/v1/generate/audio          │
│ Params:                              │
│ - duration (5-60s)                   │
│ - type (music/speech/sfx)            │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ LLM Prompt Analysis                  │
│ Extract:                             │
│ - Genre, mood, tempo                 │
│ - Instruments                        │
│ - Structure (intro/verse/outro)      │
│                                      │
│ Output:                              │
│ {                                    │
│   genre: "orchestral",               │
│   mood: "epic, intense",             │
│   tempo: 140,                        │
│   instruments: ["strings", "drums"]  │
│ }                                    │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Audio Model Selection                │
│ Router decides based on type:        │
│ - Music → MusicGen / AudioLDM        │
│ - Speech → Tortoise TTS / Bark       │
│ - SFX → AudioGen                     │
└──────┬───────────────────────────────┘
       │ (Music path)
       ▼
┌──────────────────────────────────────┐
│ TorchServe - MusicGen               │
│ Model: MusicGen Large (3.3B)         │
│ Hardware: 1x A100 GPU                │
│ Process:                             │
│ 1. Text encoding                     │
│ 2. Audio token generation            │
│    - Transformer-based               │
│    - Sampling strategy: top-k        │
│ 3. EnCodec decoding                  │
│    - Tokens → waveform               │
│ Output: 32kHz stereo WAV             │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Post-Processing                      │
│ 1. Normalize audio levels            │
│ 2. Apply mastering (optional)        │
│    - Compression, EQ, limiting       │
│ 3. Format conversion                 │
│    - MP3 (320kbps)                   │
│    - WAV (lossless)                  │
│    - OGG (web optimized)             │
│ 4. Waveform visualization            │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Storage & Delivery                   │
│ - S3 upload                          │
│ - Database record                    │
│ - WebSocket notification             │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────┐
│ User Receives│
│ Audio URL    │
└──────────────┘
```

### Performance Metrics
- **Latency**: 15-45 seconds
- **Quality**: 32kHz stereo (music), 24kHz mono (speech)
- **GPU Memory**: 15-20 GB

---

## 4. Video Captioning Pipeline

### Data Flow (Reverse: Video → Text)

```
┌──────────────┐
│ User Uploads │
│ Video File   │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────┐
│ FastAPI Upload Endpoint              │
│ POST /api/v1/caption/video           │
│ 1. Stream upload to S3               │
│ 2. Video validation (format, size)   │
│ 3. Generate job ID                   │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Celery Task: process_video_caption   │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Video Pre-Processing Worker          │
│ 1. Extract metadata (FFprobe)        │
│    - Duration, resolution, FPS       │
│ 2. Keyframe extraction               │
│    - 1 frame per second              │
│    - Save as JPEG (quality 90)       │
│ 3. Audio extraction (if present)     │
│    - AAC to WAV conversion           │
└──────┬───────────────────────────────┘
       │
       ├──────────────┬─────────────────┐
       │              │                 │
       ▼              ▼                 ▼
┌─────────────┐ ┌──────────┐ ┌──────────────┐
│ Visual      │ │  Audio   │ │ Scene        │
│ Captioning  │ │ Analysis │ │ Detection    │
└─────┬───────┘ └────┬─────┘ └──────┬───────┘
       │              │               │
       ▼              ▼               ▼
┌──────────────────────────────────────┐
│ Vision-Language Model (BLIP-2/LLaVA) │
│ TorchServe deployment                │
│                                      │
│ Process each keyframe:               │
│ 1. Image encoder (ViT)               │
│ 2. Q-Former (cross-attention)        │
│ 3. LLM decoder (OPT/Flan-T5)         │
│                                      │
│ Output per frame:                    │
│ - Dense caption                      │
│ - Object detections                  │
│ - Scene attributes                   │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Temporal Aggregation (vLLM)          │
│ LLM Chain-of-Thought:                │
│                                      │
│ Input: [caption_frame_1,             │
│         caption_frame_2, ...]        │
│                                      │
│ Process:                             │
│ 1. Identify scene changes            │
│ 2. Track object/person continuity    │
│ 3. Infer actions and events          │
│ 4. Generate coherent narrative       │
│                                      │
│ Output:                              │
│ {                                    │
│   summary: "A person walking...",    │
│   scenes: [{                         │
│     timestamp: "00:00-00:05",        │
│     description: "...",              │
│     objects: ["person", "dog"]       │
│   }],                                │
│   tags: ["outdoor", "daytime"]       │
│ }                                    │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Optional: Audio Transcription        │
│ Model: Whisper (OpenAI)              │
│ - Transcribe speech                  │
│ - Merge with visual captions         │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Caption Post-Processing              │
│ 1. Grammar correction                │
│ 2. Format as SRT/VTT (subtitles)     │
│ 3. Generate title suggestion         │
│ 4. Extract hashtags                  │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Storage                              │
│ - PostgreSQL (caption data)          │
│ - S3 (SRT file)                      │
│ - WebSocket notification             │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────┐
│ User Receives│
│ Captions     │
└──────────────┘
```

### Technical Stack
- **Vision Model**: BLIP-2, LLaVA, or GIT
- **LLM**: Llama 2/3 for temporal reasoning
- **Audio**: Whisper Large v3
- **Processing Time**: 1-3 minutes for 60s video

---

## 5. Virtual Avatar Generation Pipeline

### Data Flow

```
┌──────────────────┐
│ User Input       │
│ - Text prompt    │
│ - Style (3D/2D)  │
│ - Reference img  │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ FastAPI Endpoint                     │
│ POST /api/v1/generate/avatar         │
│ Options:                             │
│ - avatar_type: "realistic" | "anime" │
│ - pose: "T-pose" | "custom"          │
│ - rigging: true/false                │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ LLM Feature Extraction               │
│ Parse prompt for:                    │
│ - Gender, age, ethnicity             │
│ - Hair style, color                  │
│ - Clothing, accessories              │
│ - Facial features                    │
│                                      │
│ Generate structured parameters       │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Avatar Generation Pipeline Worker    │
│                                      │
│ STEP 1: Face Generation              │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ TorchServe - Portrait SD Model       │
│ Generate high-quality face:          │
│ - Resolution: 512x512                │
│ - CFG scale: 7.5                     │
│ - Steps: 40                          │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ STEP 2: 3D Mesh Generation           │
│ (If 3D avatar requested)             │
│                                      │
│ Options:                             │
│ A) PIFu/PIFuHD (single image → 3D)   │
│ B) Text-to-3D (DreamFusion/Magic3D)  │
│                                      │
│ Process:                             │
│ 1. Generate multi-view images (SD)   │
│    - Front, side, back, 3/4 views    │
│ 2. 3D reconstruction                 │
│    - NeRF or mesh generation         │
│ 3. Texture mapping                   │
│ 4. Export as FBX/glTF                │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ STEP 3: Rigging (Optional)           │
│ If rigging enabled:                  │
│ 1. Detect joints (CV model)          │
│ 2. Create skeleton (Mixamo-like)     │
│ 3. Skin weights painting             │
│ 4. Validate deformations             │
│                                      │
│ Output: Rigged FBX with animations   │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ STEP 4: Animation (Optional)         │
│ Generate idle/talking animations:    │
│ - Use motion capture data            │
│ - Procedural animation               │
│ - Lip-sync (if audio provided)       │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Post-Processing                      │
│ 1. Optimize mesh (reduce poly count) │
│ 2. Compress textures (PNG → WebP)    │
│ 3. Generate preview renders          │
│ 4. Package assets (ZIP)              │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Storage                              │
│ - S3: 3D files (FBX, glTF, textures) │
│ - S3: Preview images/videos          │
│ - Database: metadata                 │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────┐
│ User Receives│
│ Avatar Files │
└──────────────┘
```

### Technical Stack
- **2D Generation**: Stable Diffusion (fine-tuned on portraits)
- **3D Reconstruction**: PIFuHD, Instant-NGP, or custom NeRF
- **Rigging**: MediaPipe Pose, OpenPose, or custom CNN
- **Animation**: Motion diffusion model or procedural
- **Processing Time**: 2-5 minutes (2D), 10-20 minutes (3D rigged)

### File Outputs
- **2D Avatar**: PNG (2048x2048), transparent background
- **3D Avatar**:
  - FBX (with rig)
  - glTF/GLB (web-ready)
  - Textures: Diffuse, Normal, Roughness maps
  - Preview: Turntable video (MP4)

---

## Common Infrastructure Components

### Message Queue Flow (Celery + Redis)

```
┌─────────────┐
│ FastAPI     │
│ produces    │───┐
└─────────────┘   │
                  ▼
┌──────────────────────────────────┐
│ Redis (Message Broker)           │
│                                  │
│ Queues:                          │
│ - celery:image (priority 5)      │
│ - celery:video (priority 10)     │
│ - celery:audio (priority 5)      │
│ - celery:caption (priority 3)    │
│ - celery:avatar (priority 8)     │
└──────────────┬───────────────────┘
               │
               │ (Workers subscribe)
               │
    ┌──────────┼──────────┬─────────┐
    ▼          ▼          ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Worker 1│ │Worker 2│ │Worker 3│ │Worker N│
│Image   │ │Video   │ │Audio   │ │Avatar  │
└────────┘ └────────┘ └────────┘ └────────┘
```

### Real-time Updates (WebSocket)

```
┌──────────────┐
│ Celery Worker│
│ (on progress)│
└──────┬───────┘
       │
       ▼
┌─────────────────────────────┐
│ Redis Pub/Sub               │
│ Channel: jobs:{job_id}      │
└──────┬──────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│ WebSocket Server (FastAPI)  │
│ - Subscribes to channels    │
│ - Broadcasts to clients     │
└──────┬──────────────────────┘
       │
       ▼
┌──────────────┐
│ Client (Web) │
│ Receives:    │
│ {            │
│   progress: 45%, │
│   status: "generating frame 30/60" │
│ }            │
└──────────────┘
```

---

## Performance Optimization Strategies

1. **Batching**: Batch multiple inference requests (TorchServe batch size: 4-8)
2. **Caching**: Cache prompt embeddings (CLIP/LLM encodings)
3. **Model Quantization**: Use INT8/FP16 for faster inference
4. **Async Processing**: Non-blocking I/O for all storage operations
5. **CDN**: Serve generated assets via CloudFlare R2/S3 + CDN
6. **GPU Sharing**: Multiple models on same GPU using CUDA MPS

---

This data flow documentation provides the foundation for understanding how each pipeline operates independently while sharing common infrastructure.
