# TXT2CREATE - Comprehensive Q&A

This document covers all the concepts, technologies, and design decisions used in the TXT2CREATE platform.

---

## Table of Contents

1. [General Architecture](#1-general-architecture)
2. [AI Models & Techniques](#2-ai-models--techniques)
3. [Infrastructure & Deployment](#3-infrastructure--deployment)
4. [Pipeline-Specific Questions](#4-pipeline-specific-questions)
5. [Performance & Optimization](#5-performance--optimization)
6. [Scalability & Reliability](#6-scalability--reliability)
7. [Security & Privacy](#7-security--privacy)
8. [Cost & Economics](#8-cost--economics)
9. [Development & Operations](#9-development--operations)

---

## 1. GENERAL ARCHITECTURE

### Q: What is the overall architecture pattern used?

**A:** TXT2CREATE uses a **microservices architecture** with the following layers:

1. **Presentation Layer**: React web app, mobile app
2. **API Gateway**: NGINX/Kong for routing, rate limiting, authentication
3. **Application Layer**: FastAPI services for business logic
4. **Task Orchestration**: Celery + Redis for async job processing
5. **AI Inference Layer**: TorchServe + vLLM for model serving
6. **Data Layer**: PostgreSQL, Redis, MongoDB, S3
7. **Infrastructure**: Kubernetes on cloud (AWS/GCP)

**Why this architecture?**
- ✅ Independent scaling of each component
- ✅ Technology flexibility (different languages/frameworks per service)
- ✅ Fault isolation (one service failure doesn't crash everything)
- ✅ Easy deployment and updates
- ✅ Team autonomy (different teams own different services)

---

### Q: Why FastAPI instead of Flask or Django?

**A:** FastAPI offers several advantages:

| Feature | FastAPI | Flask | Django |
|---------|---------|-------|--------|
| **Performance** | Very fast (Uvicorn/ASGI) | Moderate (WSGI) | Moderate (WSGI) |
| **Async Support** | Native async/await | Requires extensions | Limited |
| **Type Validation** | Built-in (Pydantic) | Manual | Django forms |
| **Auto Documentation** | OpenAPI/Swagger | Manual | Manual |
| **WebSocket** | Built-in | Requires extensions | Channels (complex) |
| **Learning Curve** | Easy | Very easy | Steeper |

**Why it matters for TXT2CREATE:**
- Long-running AI tasks need async processing
- WebSocket for real-time progress updates
- Type validation prevents errors with complex parameters
- Auto-generated API docs for developers

---

### Q: What is the request flow from user to AI model?

**A:** Here's the complete flow for text-to-image:

```
1. User submits prompt "A cat in a garden"
   ↓
2. React app sends POST to /api/v1/generate/image
   ↓
3. NGINX API Gateway
   - Checks SSL/TLS
   - Validates JWT token
   - Applies rate limiting
   ↓
4. FastAPI endpoint
   - Validates input (Pydantic)
   - Checks user quota
   - Generates job_id
   ↓
5. LLM Chain-of-Thought (vLLM)
   - Enhances prompt: "A photorealistic fluffy orange cat..."
   - Generates negative prompt
   ↓
6. Redis cache check
   - Hash: prompt + parameters
   - If HIT → return cached image
   - If MISS → continue
   ↓
7. Celery task queue
   - Publishes task to Redis
   - Returns job_id to user
   ↓
8. Celery worker picks up task
   - Routes to available GPU
   ↓
9. TorchServe inference
   - Loads Stable Diffusion model
   - Runs inference (30-50 steps)
   - Returns image tensor
   ↓
10. Post-processing
    - Convert tensor to PNG
    - Compress (quality 85)
    - Add watermark (optional)
   ↓
11. Upload to S3
    - Store at /images/{user_id}/{job_id}.png
    - Get public URL
   ↓
12. Save to PostgreSQL
    - job_id, user_id, prompt, s3_url, etc.
   ↓
13. WebSocket notification
    - Push to user: {job_id, status: "completed", image_url}
   ↓
14. User sees result in browser (30-60 seconds total)
```

---

### Q: How do you handle real-time updates?

**A:** Using **WebSocket** connections:

```python
# FastAPI WebSocket endpoint
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket.accept()

    # Subscribe to Redis pub/sub for user's jobs
    pubsub = redis.pubsub()
    await pubsub.subscribe(f"user:{user_id}:jobs")

    # Listen for updates
    async for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            # Send to client
            await websocket.send_json(data)
```

**How Celery workers publish updates:**

```python
@celery.task
def generate_image_task(job_id, user_id, prompt):
    # Publish progress updates
    redis.publish(f"user:{user_id}:jobs", json.dumps({
        "job_id": job_id,
        "status": "processing",
        "progress": 25,
        "message": "Enhancing prompt..."
    }))

    enhanced = llm_enhance(prompt)

    redis.publish(f"user:{user_id}:jobs", json.dumps({
        "job_id": job_id,
        "progress": 50,
        "message": "Generating image..."
    }))

    image = stable_diffusion(enhanced)

    redis.publish(f"user:{user_id}:jobs", json.dumps({
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "image_url": s3_url
    }))
```

**Why WebSocket instead of polling?**
- ✅ Real-time (no delay)
- ✅ Efficient (no repeated HTTP requests)
- ✅ Better UX (live progress bars)
- ✅ Lower server load

---

## 2. AI MODELS & TECHNIQUES

### Q: What is Stable Diffusion and how does it work?

**A:** Stable Diffusion is a **latent diffusion model** for text-to-image generation.

**How it works:**

```
1. Text Encoding (CLIP)
   "A cat in a garden" → [text_embedding: 768-dim vector]

2. Random Noise Generation
   Start with pure noise: [1024×1024×3 RGB noise]

3. VAE Encoding (downsampling)
   1024×1024×3 → 128×128×4 latent space (8x compression)

4. Iterative Denoising (U-Net)
   For 30-50 steps:
     - U-Net predicts noise in latent
     - Subtract predicted noise
     - Move closer to target image
     - Guided by text_embedding

5. VAE Decoding (upsampling)
   128×128×4 latent → 1024×1024×3 RGB image

6. Safety Check
   Run NSFW classifier, blur if needed
```

**Why latent diffusion?**
- ✅ 8x smaller than pixel space (faster, less memory)
- ✅ High quality results
- ✅ Controllable with text

**Key parameters:**
- **Steps** (25-150): More = higher quality, slower
- **CFG Scale** (1-20): How much to follow prompt (7.5 is balanced)
- **Seed**: Random seed for reproducibility

---

### Q: What is VAE and why is it used for video compression?

**A:** VAE = **Variational Autoencoder**

**Structure:**
```
Input Image → Encoder → Latent (compressed) → Decoder → Output Image
```

**For video compression:**

```python
# Compress video frames
frames = [frame_0, frame_1, ..., frame_240]  # 240 RGB frames

# Encode to latent
latents = []
for frame in frames:
    latent = vae.encode(frame)  # 1024×1024×3 → 128×128×4
    latents.append(latent)

# Temporal compression (compress across time)
compressed_video = temporal_compressor(latents)

# Later: Decode back
for latent in latents:
    frame = vae.decode(latent)  # 128×128×4 → 1024×1024×3
```

**Benefits:**
- ✅ 8x spatial compression (less storage)
- ✅ Can apply diffusion in latent space (faster)
- ✅ Lossy but high quality
- ✅ Better than traditional codecs for AI-generated content

**Alternative:** H.264/H.265 codecs are used for final delivery, VAE is for intermediate processing.

---

### Q: What is Chain-of-Thought (CoT) and why use it with LLMs?

**A:** Chain-of-Thought prompting makes LLMs **show their reasoning** step-by-step.

**Example:**

**Without CoT:**
```
Prompt: "Generate a detailed prompt for: A cat in a garden"
LLM: "A photorealistic cat in a garden with flowers"
```

**With CoT:**
```
Prompt: "Generate a detailed prompt for: A cat in a garden.
Think step by step:
1. What type of cat?
2. What is the cat doing?
3. What does the garden look like?
4. What is the lighting?
5. What is the art style?

Now generate the enhanced prompt."

LLM:
"Let me think step by step:
1. Type: A fluffy orange tabby cat
2. Action: Sitting peacefully, looking at a butterfly
3. Garden: Vibrant garden with roses, tulips, stone path
4. Lighting: Golden hour sunlight, soft shadows
5. Style: Photorealistic, professional photography

Enhanced prompt: A fluffy orange tabby cat sitting peacefully in a vibrant garden filled with red roses and yellow tulips, watching a blue butterfly, during golden hour with soft sunlight and gentle shadows, professional photography, highly detailed, 8k resolution"
```

**Why it's better:**
- ✅ More detailed and creative prompts
- ✅ Consistent quality
- ✅ LLM understands context better
- ✅ Can handle complex requests

**How we use it in TXT2CREATE:**
1. **Prompt enhancement**: Simple prompt → Detailed prompt
2. **Video scene decomposition**: Video idea → Keyframe prompts
3. **Negative prompt generation**: What to avoid in the image
4. **Style parameter suggestion**: Recommend best settings

---

### Q: What is vLLM and why use it instead of standard transformers?

**A:** vLLM = **Very Fast LLM Inference Engine**

**Key features:**

1. **PagedAttention**: Efficient memory management
   - Stores KV cache in pages (like OS virtual memory)
   - Reduces memory waste by 50%
   - Enables larger batch sizes

2. **Continuous Batching**: Process requests as they arrive
   - Don't wait for full batch
   - Higher throughput

3. **Optimized CUDA Kernels**: Custom GPU code
   - Faster than PyTorch's default
   - Flash Attention 2 integration

**Performance comparison:**

| Method | Throughput (tokens/sec) | Latency (first token) | GPU Memory |
|--------|-------------------------|----------------------|------------|
| Transformers (batch=1) | 30 | 200ms | 20 GB |
| Transformers (batch=8) | 120 | 500ms | 40 GB |
| vLLM (dynamic batch) | 600 | 150ms | 24 GB |

**Why it matters:**
- ✅ 5x faster than standard transformers
- ✅ Handle more concurrent users
- ✅ Lower latency (better UX)
- ✅ Lower cost per token

**When to use:**
- ✅ Production LLM serving (Llama, Mistral, etc.)
- ✅ High traffic scenarios
- ❌ Not needed for small models or low traffic

---

### Q: What is TorchServe and why use it?

**A:** TorchServe = **PyTorch's official model serving framework**

**Features:**
- Multi-model serving (multiple models on same server)
- Batching (combine requests for efficiency)
- Model versioning (A/B testing)
- REST API + gRPC
- Metrics (Prometheus integration)
- GPU support

**Architecture:**
```
HTTP Request → TorchServe Frontend → Queue → Worker (GPU)
                                             ↓
                                        PyTorch Model
                                             ↓
                                        Response
```

**Configuration example:**
```bash
# MAR file (Model Archive)
torch-model-archiver \
  --model-name sd-xl \
  --version 1.0 \
  --model-file model.py \
  --serialized-file model.safetensors \
  --handler custom_handler.py

# Serve
torchserve \
  --start \
  --model-store /models \
  --models sd-xl=sd-xl.mar \
  --ncs  # No config snapshot (dynamic reload)
```

**Alternatives:**
- **Triton Inference Server** (NVIDIA): More features, more complex
- **BentoML**: Simpler, less GPU-optimized
- **Custom Flask API**: Maximum flexibility, more work

**Why TorchServe for TXT2CREATE:**
- ✅ Official PyTorch support
- ✅ Good batching for Stable Diffusion
- ✅ Easy integration with Kubernetes
- ✅ Active development

---

### Q: How does frame interpolation work for video generation?

**A:** Frame interpolation fills in missing frames between keyframes.

**Process:**

```
Keyframes (generated by Stable Diffusion):
Frame 0   [cat sitting]
Frame 30  [cat standing]
Frame 60  [cat walking]

Interpolation (FILM/RIFE):
Frame 0   [cat sitting]
Frame 1   [cat sitting, slight movement] ← interpolated
Frame 2   [cat sitting, more movement]    ← interpolated
...
Frame 30  [cat standing]
Frame 31  [cat standing, lifting paw]     ← interpolated
...
```

**FILM (Frame Interpolation for Large Motion):**
- Uses neural network to predict intermediate frames
- Handles large motion well
- Can generate multiple intermediate frames (not just 1)

**How it works:**
```
Input: frame_A, frame_B, t (time between 0-1)

1. Extract features from both frames (CNN encoder)
2. Estimate optical flow (motion between frames)
3. Warp features based on flow
4. Blend warped features
5. Decode to final frame (CNN decoder)

Output: frame_at_time_t
```

**Why it's important:**
- ✅ Smooth video instead of slideshows
- ✅ 5-10x fewer Stable Diffusion generations needed
- ✅ Faster video creation
- ✅ Lower cost

**Quality factors:**
- Works best with similar frames (small motion)
- Struggles with occlusions, lighting changes
- Need good keyframes (consistent style)

---

### Q: What models are used for video captioning and why?

**A:** Multi-modal vision-language models:

**1. BLIP-2 (Bootstrapping Language-Image Pre-training)**

```
Architecture:
Image → Vision Transformer (ViT) → Q-Former → LLM → Caption
```

- **ViT**: Encodes image to features
- **Q-Former**: Bridges vision and language (cross-attention)
- **LLM**: Generates natural language description

**Strengths:**
- ✅ High quality captions
- ✅ Understands context
- ✅ Can answer questions about image

**2. LLaVA (Large Language and Vision Assistant)**

```
Architecture:
Image → CLIP Vision Encoder → Projection → Llama → Caption
```

- More conversational
- Better reasoning
- Larger model (better quality, slower)

**Why use both?**
- BLIP-2: Per-frame captions (fast, efficient)
- LLaVA: Final narrative generation (high quality)

**Process:**
```
Video (240 frames)
  ↓
Extract keyframes (1 per second = 8 frames)
  ↓
BLIP-2 caption each frame
  Frame 0: "A person walking on a street"
  Frame 1: "The person is approaching a dog"
  Frame 2: "The person is petting the dog"
  ...
  ↓
LLaVA aggregates into narrative
  "A person walks down a sunny street and encounters a friendly dog.
   They stop to pet the dog, which wags its tail happily."
```

---

### Q: How does the avatar generation work?

**A:** Multi-stage pipeline:

**Stage 1: Face Generation (2D)**
```
Text prompt "Young woman with blonde hair"
  ↓
Stable Diffusion (portrait fine-tuned)
  ↓
High-quality face image (2048×2048)
```

**Stage 2: 3D Mesh Generation**

**Option A: PIFuHD (Pixel-aligned Implicit Function)**
```
Single 2D image
  ↓
Estimate depth (monocular depth estimation)
  ↓
Generate 3D mesh (implicit function)
  ↓
Texture mapping (from original image)
  ↓
3D model (OBJ/FBX)
```

**Option B: Multi-view generation + NeRF**
```
Generate 4 views with SD (front, back, left, right)
  ↓
NeRF (Neural Radiance Field) reconstruction
  ↓
Extract mesh (marching cubes)
  ↓
Texture baking
  ↓
3D model
```

**Stage 3: Rigging (Optional)**
```
3D mesh
  ↓
Detect joints (MediaPipe / OpenPose)
  - Shoulders, elbows, hips, knees, etc.
  ↓
Create skeleton
  ↓
Skin weights (automatic or manual)
  - How much each bone affects each vertex
  ↓
Rigged model (ready for animation)
```

**Stage 4: Animation (Optional)**
```
Rigged model
  ↓
Apply motion capture data (idle, walk, talk)
  ↓
Animated avatar (FBX with animations)
```

**Outputs:**
- 2D: PNG (transparent background)
- 3D Static: FBX, OBJ, glTF
- 3D Rigged: FBX (with skeleton)
- 3D Animated: FBX (with animations)

---

## 3. INFRASTRUCTURE & DEPLOYMENT

### Q: Why Kubernetes instead of VMs or serverless?

**A:** Comparison:

| Feature | Kubernetes | VMs (EC2) | Serverless (Lambda) |
|---------|-----------|-----------|---------------------|
| **Auto-scaling** | ✅ HPA, cluster autoscaler | Manual or basic | ✅ Automatic |
| **GPU support** | ✅ Native | ✅ Native | ❌ Very limited |
| **Cold start** | ✅ Minimal (pods running) | ✅ None (always on) | ❌ 1-10 seconds |
| **Cost efficiency** | ✅ Bin packing, scaling | ❌ Over-provisioning | ✅ Pay per use |
| **Complex workflows** | ✅ Services, jobs, crons | Manual | ❌ Function stitching |
| **Stateful apps** | ✅ StatefulSets, PVs | ✅ EBS volumes | ❌ Difficult |
| **Portability** | ✅ Any cloud | ❌ Cloud-specific | ❌ Cloud-specific |
| **Learning curve** | ❌ Steep | ✅ Familiar | ✅ Simple |

**For TXT2CREATE:**
- ✅ Need GPUs (Lambda doesn't support well)
- ✅ Stateful (database, Redis)
- ✅ Complex (microservices, batch jobs)
- ✅ Portable (can move clouds)
- ✅ Efficient scaling (pack workloads on nodes)

**When NOT to use Kubernetes:**
- Simple apps (single server is enough)
- Pure API (no state, no GPUs) → Serverless is simpler
- Small team (K8s adds complexity)

---

### Q: How do you manage GPU resources in Kubernetes?

**A:** Using **NVIDIA GPU Operator** and resource limits:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torchserve-sd
spec:
  replicas: 4
  template:
    spec:
      containers:
      - name: torchserve
        image: torchserve:latest
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1  # Request 1 GPU
          limits:
            memory: "20Gi"
            cpu: "8"
            nvidia.com/gpu: 1  # Limit to 1 GPU

      # Node selector: only schedule on GPU nodes
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-a100

      # Toleration: allow scheduling on tainted nodes
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
```

**GPU Node Pool:**
```yaml
# GKE example
gcloud container node-pools create gpu-pool \
  --accelerator type=nvidia-tesla-a100,count=1 \
  --machine-type n1-standard-16 \
  --num-nodes 4 \
  --min-nodes 2 \
  --max-nodes 8 \
  --enable-autoscaling
```

**GPU Sharing Strategies:**

**1. MIG (Multi-Instance GPU):**
```bash
# Split A100 into 7 instances
nvidia-smi mig -cgi 0,0,0,0,0,0,0

# Each pod gets 1/7 of A100
```

**2. Time-Slicing:**
```yaml
# Multiple pods share same GPU (time-multiplexing)
replicas: 10
nvidia.com/gpu: 1  # All 10 pods share 1 GPU
```

**3. MPS (Multi-Process Service):**
```bash
# Multiple processes share GPU memory
nvidia-cuda-mps-control -d
```

**Best practices:**
- ✅ Use resource limits (prevent OOM)
- ✅ Node affinity (co-locate related services)
- ✅ Taints (keep GPU nodes for GPU workloads only)
- ✅ Monitor GPU utilization (dcgm-exporter → Prometheus)

---

### Q: How do you handle database migrations?

**A:** Using **Alembic** (SQLAlchemy migration tool):

```python
# Generate migration
alembic revision --autogenerate -m "Add avatar_config table"

# This creates: migrations/versions/abc123_add_avatar_config.py

def upgrade():
    op.create_table(
        'avatar_config',
        sa.Column('id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('style', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'])
    )
    op.create_index('idx_avatar_user', 'avatar_config', ['user_id'])

def downgrade():
    op.drop_index('idx_avatar_user', 'avatar_config')
    op.drop_table('avatar_config')
```

**Deployment process:**

```bash
# 1. Run migrations in init container (before app starts)
kubectl apply -f migration-job.yaml

# migration-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration-v2
spec:
  template:
    spec:
      containers:
      - name: migration
        image: txt2create-api:v2
        command: ["alembic", "upgrade", "head"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
      restartPolicy: Never

# 2. Wait for migration to complete
# 3. Deploy new app version
kubectl set image deployment/api api=txt2create-api:v2
```

**Best practices:**
- ✅ Always have `upgrade()` and `downgrade()` functions
- ✅ Test migrations on staging first
- ✅ Backup database before migration
- ✅ Use transactions (migrations are atomic)
- ✅ Avoid data-heavy migrations in sync (use background jobs)

---

### Q: How do you handle secrets (API keys, passwords)?

**A:** Using **HashiCorp Vault** (production) or **Kubernetes Secrets** (simpler):

**Option 1: Kubernetes Secrets (Base64 encoded, encrypted at rest)**

```bash
# Create secret
kubectl create secret generic db-credentials \
  --from-literal=username=postgres \
  --from-literal=password=super-secret-123

# Use in pod
apiVersion: v1
kind: Pod
metadata:
  name: api
spec:
  containers:
  - name: api
    image: txt2create-api
    env:
    - name: DB_USERNAME
      valueFrom:
        secretKeyRef:
          name: db-credentials
          key: username
    - name: DB_PASSWORD
      valueFrom:
        secretKeyRef:
          name: db-credentials
          key: password
```

**Option 2: HashiCorp Vault (Enterprise-grade)**

```python
# Vault client
import hvac

client = hvac.Client(url='http://vault:8200')
client.auth.kubernetes.login(role='txt2create-api', jwt=jwt_token)

# Read secret
secret = client.secrets.kv.v2.read_secret_version(path='database/credentials')
db_password = secret['data']['data']['password']

# Use secret
db = connect(password=db_password)
```

**Vault benefits:**
- ✅ Dynamic secrets (auto-rotated)
- ✅ Audit log (who accessed what)
- ✅ Fine-grained access control (RBAC)
- ✅ Encryption as a service

**Secrets we manage:**
- Database passwords
- S3 access keys
- JWT signing keys
- API keys (OpenAI, etc.)
- TLS certificates

**Best practices:**
- ✅ Never commit secrets to Git
- ✅ Use different secrets per environment (dev/staging/prod)
- ✅ Rotate secrets regularly (90 days)
- ✅ Minimum privilege (each service gets only what it needs)

---

## 4. PIPELINE-SPECIFIC QUESTIONS

### Q: Why enhance prompts with LLM instead of using user prompt directly?

**A:** User prompts are often vague:

```
User: "A cat"
SD Result: Low quality, generic cat photo

User: "A photorealistic fluffy orange tabby cat with green eyes
       sitting on a wooden fence during golden hour, professional
       photography, highly detailed, 8k, sharp focus, bokeh"
SD Result: Beautiful, high-quality image
```

**LLM enhancement:**
```python
user_prompt = "A cat"

llm_enhanced = llm.generate(
    f"""You are an expert at writing Stable Diffusion prompts.
    User wants: {user_prompt}

    Generate a detailed, high-quality prompt that includes:
    - Specific details (breed, color, pose)
    - Setting and environment
    - Lighting and atmosphere
    - Art style and quality modifiers

    Enhanced prompt:"""
)

# Output: "A photorealistic fluffy orange tabby cat..."
```

**Benefits:**
- ✅ Higher quality results (better CLIP alignment)
- ✅ Consistent style (LLM adds quality modifiers)
- ✅ Better user experience (users don't need to be experts)
- ✅ Can incorporate user preferences (remembered style)

**When to skip enhancement:**
- User explicitly says "no enhancement"
- User provides very detailed prompt already
- Emergency/debug mode

---

### Q: How do you ensure temporal consistency in videos?

**A:** Multiple techniques:

**1. ControlNet (pose/depth conditioning)**
```python
# Extract pose from first frame
first_frame_pose = extract_pose(first_frame)

# Generate subsequent frames with same pose
for i in range(num_frames):
    frame = sd_controlnet.generate(
        prompt=prompts[i],
        control_image=first_frame_pose,  # Enforce consistency
        strength=0.8
    )
```

**2. AnimateDiff (motion module)**
```python
# Learns temporal consistency from training
video = animatediff.generate(
    prompt="A cat walking",
    num_frames=60,
    motion_strength=0.7
)
# Frames are temporally coherent by design
```

**3. Latent interpolation**
```python
# Smooth interpolation in latent space
latent_0 = vae.encode(frame_0)
latent_30 = vae.encode(frame_30)

# Interpolate
for t in range(30):
    alpha = t / 30
    latent_t = (1 - alpha) * latent_0 + alpha * latent_30
    frame_t = vae.decode(latent_t)
```

**4. Same seed for keyframes**
```python
# Use seed-based generation
base_seed = 42
for i, prompt in enumerate(prompts):
    frame = sd.generate(
        prompt=prompt,
        seed=base_seed + i  # Sequential seeds are similar
    )
```

**Challenges:**
- Flickering (frame-to-frame inconsistency)
- Style drift (colors change over time)
- Object identity (person's face changes)

**Solutions:**
- Use video-specific models (AnimateDiff, Modelscope)
- Post-processing: deflicker algorithms
- Reference frames: condition on previous frames

---

### Q: How do you prevent NSFW content generation?

**A:** Multi-layer approach:

**Layer 1: Input filtering (prompt)**
```python
from openai import OpenAI

def is_nsfw_prompt(prompt: str) -> bool:
    # Use OpenAI Moderation API
    client = OpenAI()
    response = client.moderations.create(input=prompt)
    result = response.results[0]

    if result.flagged:
        return True

    # Keyword blacklist
    nsfw_keywords = ["nude", "naked", "porn", ...]
    if any(kw in prompt.lower() for kw in nsfw_keywords):
        return True

    return False

# Block request if prompt is NSFW
if is_nsfw_prompt(user_prompt):
    raise HTTPException(400, "NSFW content not allowed")
```

**Layer 2: Output detection (image)**
```python
from nudenet import NudeDetector

detector = NudeDetector()

def check_nsfw_image(image_path: str) -> dict:
    detections = detector.detect(image_path)

    # detections = [{class: 'EXPOSED_BREAST', score: 0.95}, ...]
    nsfw_classes = ['EXPOSED_BREAST', 'EXPOSED_GENITALIA', ...]

    max_score = max([
        d['score'] for d in detections
        if d['class'] in nsfw_classes
    ], default=0)

    return {
        'is_nsfw': max_score > 0.7,
        'score': max_score
    }

# After generation
result = check_nsfw_image(generated_image_path)
if result['is_nsfw']:
    # Blur image
    blur_image(generated_image_path)
    # Flag in database
    db.update(job_id, nsfw=True)
    # Notify user
    notify("Image may contain inappropriate content")
```

**Layer 3: Stable Diffusion Safety Checker (built-in)**
```python
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    safety_checker=True  # Enables safety checker
)

image, has_nsfw = pipe(
    prompt="...",
    return_dict=False
)

if has_nsfw:
    # Return black image or blur
```

**Layer 4: Human moderation**
- Queue for review if automated detection uncertain (score 0.5-0.7)
- User reporting system
- Regular audits

**False positives:**
- Medical images (anatomy education)
- Art (classical paintings with nudity)

**Solution:** Allow appeals, context-aware filtering

---

### Q: How do you handle audio-video sync for avatar animations?

**A:** **Lip-sync technology**:

**Process:**
```
1. Audio Input
   "Hello, how are you?" (WAV file)
   ↓
2. Phoneme Extraction (Whisper or Rhubarb Lip Sync)
   Audio → Phonemes with timestamps
   [
     {phoneme: 'HH', time: 0.0},
     {phoneme: 'EH', time: 0.1},
     {phoneme: 'L', time: 0.2},
     ...
   ]
   ↓
3. Viseme Mapping
   Phoneme → Mouth shape (viseme)
   {
     'A': mouth_open_wide,
     'M': lips_closed,
     'O': mouth_round,
     ...
   }
   ↓
4. Blend Shape Animation
   For each frame at 30 FPS:
     t = frame / 30
     phoneme = get_phoneme_at_time(t)
     viseme = phoneme_to_viseme[phoneme]
     avatar.mouth = viseme
   ↓
5. Smooth Interpolation
   Blend between visemes (ease in/out)
   Add idle animations (blinking, breathing)
   ↓
6. Render Video
   Avatar with lip-synced animation + audio
```

**Tools:**
- **Rhubarb Lip Sync**: Analyzes audio → phonemes
- **FaceMesh (MediaPipe)**: Detects mouth landmarks
- **Audio2Face (NVIDIA)**: AI-driven facial animation

**Challenges:**
- Timing precision (audio-video drift)
- Multiple languages (different phonemes)
- Emotions (smile, sad, angry affects mouth shape)

**Quality factors:**
- Good rigging (blend shapes for all visemes)
- Smooth transitions (avoid snapping)
- Natural idle (don't freeze between words)

---

## 5. PERFORMANCE & OPTIMIZATION

### Q: Why quantization (INT8/FP16) and what's the trade-off?

**A:** **Quantization** = Reduce precision of model weights/activations

**Precision comparison:**

| Precision | Bits | Range | Size (70B params) | Speed | Quality |
|-----------|------|-------|-------------------|-------|---------|
| FP32 | 32 | ±3.4×10³⁸ | 280 GB | 1x (baseline) | 100% |
| FP16 | 16 | ±65,504 | 140 GB | 2x faster | 99.9% |
| INT8 | 8 | -128 to 127 | 70 GB | 3-4x faster | 98-99% |
| INT4 | 4 | -8 to 7 | 35 GB | 5-6x faster | 95-97% |

**How it works (INT8):**

```python
# Original FP32 weight
weight_fp32 = 0.123456789  # 32 bits

# Quantize to INT8 (-128 to 127)
scale = max(abs(weight_fp32)) / 127
weight_int8 = round(weight_fp32 / scale)  # -128 to 127
# Store: weight_int8=16, scale=0.00097

# Dequantize when needed
weight_fp32_approx = weight_int8 * scale  # 0.12345 (close enough)
```

**When to use:**

| Use Case | Precision | Reason |
|----------|-----------|--------|
| Training | FP32 or Mixed (FP16) | Need precision for gradients |
| Inference (quality-critical) | FP16 | 2x speedup, minimal loss |
| Inference (high throughput) | INT8 | 4x speedup, acceptable loss |
| Edge devices (mobile) | INT4 | Extreme compression |

**Trade-offs:**
- ✅ Faster inference (2-4x)
- ✅ Less GPU memory (50-75% reduction)
- ✅ Higher throughput (more requests/sec)
- ❌ Slight quality degradation (1-5% drop in metrics)
- ❌ First-time quantization is slow

**Practical example (Stable Diffusion):**
```python
from diffusers import StableDiffusionPipeline
import torch

# FP16 (most common)
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
)

# INT8 (advanced)
from optimum.quanto import quantize, freeze
quantize(pipe.unet, weights=torch.int8, activations=torch.int8)
freeze(pipe.unet)

# Generation speed: FP32 (60s) → FP16 (30s) → INT8 (20s)
```

---

### Q: What is batch processing and when to use it?

**A:** **Batching** = Process multiple requests together on GPU

**How it helps:**

```
Without batching (sequential):
Request 1 → GPU (10s) → Result 1
Request 2 → GPU (10s) → Result 2
Request 3 → GPU (10s) → Result 3
Total: 30 seconds

With batching (batch=3):
Requests 1, 2, 3 → GPU (12s) → Results 1, 2, 3
Total: 12 seconds (2.5x faster)
```

**Why batching is faster:**
- GPUs have thousands of cores
- Single request doesn't saturate GPU
- Batching increases parallelism

**GPU utilization:**
```
Batch size 1:  GPU usage 30% (underutilized)
Batch size 4:  GPU usage 70%
Batch size 8:  GPU usage 90% (optimal)
Batch size 16: GPU usage 95%, but OOM risk
```

**TorchServe batching config:**
```python
# config.properties
batch_size=8
max_batch_delay=100  # Wait up to 100ms for full batch
```

**Trade-offs:**

| Batch Size | Throughput | Latency (per request) | GPU Memory |
|------------|------------|----------------------|------------|
| 1 | 20 req/min | 3s | 8 GB |
| 4 | 60 req/min | 4s (+33%) | 16 GB |
| 8 | 100 req/min | 5s (+66%) | 28 GB |
| 16 | 120 req/min | 8s (+166%) | 48 GB (OOM!) |

**When to use:**
- ✅ High traffic (many concurrent users)
- ✅ Batch jobs (pre-generate images)
- ✅ Non-real-time (can wait 100ms for batch)
- ❌ Low traffic (not enough requests to batch)
- ❌ Real-time (can't wait for other requests)

**Dynamic batching:**
```python
# Don't wait for full batch if traffic is low
if queue_depth < 4:
    batch_size = queue_depth  # Process what we have
else:
    batch_size = 8  # Full batch
```

---

### Q: How do you handle cache invalidation?

**A:** Cache invalidation strategies:

**1. TTL (Time-to-Live)**
```python
# Cache for 24 hours
redis.setex(
    key=f"result:{hash(prompt)}",
    value=image_url,
    time=86400  # 24 hours
)

# After 24h, key is auto-deleted
```

**Use cases:**
- ✅ Generated images (24h TTL)
- ✅ Prompt embeddings (7d TTL)
- ✅ User sessions (7d TTL)

**2. Explicit invalidation**
```python
# Invalidate when model is updated
async def update_model(model_name, new_version):
    # Deploy new model
    await deploy_model(model_name, new_version)

    # Invalidate all cached results for this model
    keys = await redis.keys(f"result:{model_name}:*")
    if keys:
        await redis.delete(*keys)
```

**Use cases:**
- ✅ Model updates (quality improved)
- ✅ User preferences changed
- ✅ Content moderation (remove NSFW)

**3. Lazy invalidation (read-through)**
```python
async def get_cached_or_generate(prompt, params):
    cache_key = f"result:{hash(prompt + json.dumps(params))}"

    # Try cache
    cached = await redis.get(cache_key)
    if cached:
        cached_data = json.loads(cached)
        # Check if model version matches
        if cached_data['model_version'] == CURRENT_MODEL_VERSION:
            return cached_data['url']
        else:
            # Old model, regenerate
            await redis.delete(cache_key)

    # Generate fresh result
    url = await generate_image(prompt, params)
    await redis.setex(
        cache_key,
        json.dumps({
            'url': url,
            'model_version': CURRENT_MODEL_VERSION
        }),
        86400
    )
    return url
```

**4. Write-through cache**
```python
# Update both cache and database
async def update_user_preferences(user_id, preferences):
    # Update database
    await db.update_user(user_id, preferences)

    # Update cache
    await redis.hset(f"user:{user_id}", "preferences", json.dumps(preferences))
```

**Cache coherence problem:**
```
User 1: Update preferences → Database ✅
User 1: Update preferences → Cache ❌ (network failure)
User 2: Read from cache → Gets stale data!
```

**Solution: Cache-aside pattern with TTL**
```python
# Always check DB for critical data
preferences = await redis.get(f"user:{user_id}:preferences")
if not preferences:
    preferences = await db.get_user_preferences(user_id)
    await redis.setex(f"user:{user_id}:preferences", preferences, 3600)
```

---

## 6. SCALABILITY & RELIABILITY

### Q: How do you handle GPU failures?

**A:** Multi-layer failure handling:

**1. Health checks**
```python
# GPU health check
import pynvml

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def check_gpu_health():
    try:
        # Check temperature
        temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
        if temp > 85:
            return False, "GPU too hot"

        # Check memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if mem_info.free < 1e9:  # < 1 GB free
            return False, "GPU memory low"

        # Check ECC errors
        ecc_errors = pynvml.nvmlDeviceGetTotalEccErrors(handle, ...)
        if ecc_errors > 0:
            return False, "GPU hardware errors"

        return True, "OK"
    except:
        return False, "GPU unavailable"

# Kubernetes liveness probe
@app.get("/healthz")
async def health_check():
    healthy, message = check_gpu_health()
    if not healthy:
        return Response(status_code=503, content=message)
    return {"status": "ok"}
```

**2. Automatic pod restart (Kubernetes)**
```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8080
  initialDelaySeconds: 120
  periodSeconds: 30
  failureThreshold: 3  # Restart after 3 failures
```

**3. Task retry (Celery)**
```python
@celery.task(
    autoretry_for=(GPUError, CUDAOutOfMemoryError),
    retry_kwargs={'max_retries': 3, 'countdown': 60}
)
def generate_image_task(prompt):
    try:
        return generate_image(prompt)
    except CUDAOutOfMemoryError:
        # Clear GPU memory
        torch.cuda.empty_cache()
        raise  # Retry on different worker
```

**4. Circuit breaker**
```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=300)
def call_gpu_worker(gpu_id):
    # If GPU fails 5 times, stop sending requests for 5 minutes
    return worker_pool[gpu_id].generate(...)
```

**5. Failover to CPU (graceful degradation)**
```python
def generate_image(prompt):
    try:
        # Try GPU
        return gpu_inference(prompt)
    except GPUUnavailableError:
        # Fallback to CPU (slower but works)
        logger.warning("GPU unavailable, using CPU")
        return cpu_inference(prompt)
```

**6. Multi-GPU redundancy**
```python
# Load balance across 4 GPUs
gpu_pool = [
    TorchServeClient("http://gpu-0:8080"),
    TorchServeClient("http://gpu-1:8080"),
    TorchServeClient("http://gpu-2:8080"),
    TorchServeClient("http://gpu-3:8080"),
]

def generate_with_failover(prompt):
    for i, client in enumerate(gpu_pool):
        try:
            return client.generate(prompt)
        except Exception as e:
            logger.error(f"GPU {i} failed: {e}")
            continue  # Try next GPU

    raise AllGPUsFailedError()
```

---

### Q: How do you ensure database consistency under high load?

**A:** ACID properties + optimizations:

**1. Transactions**
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

async def create_job_with_payment(user_id, prompt, cost):
    async with db.begin() as transaction:
        try:
            # Deduct credits
            user = await db.query(User).filter_by(id=user_id).with_for_update().first()
            if user.credits < cost:
                raise InsufficientCreditsError()
            user.credits -= cost

            # Create job
            job = Job(user_id=user_id, prompt=prompt, status='pending')
            db.add(job)

            # Commit both or rollback both
            await transaction.commit()
            return job
        except Exception:
            await transaction.rollback()
            raise
```

**2. Connection pooling**
```python
# SQLAlchemy connection pool
engine = create_engine(
    "postgresql://...",
    pool_size=20,  # Normal connections
    max_overflow=10,  # Extra connections under load
    pool_pre_ping=True,  # Check connection before use
    pool_recycle=3600  # Recycle connections every hour
)
```

**3. Read replicas**
```python
# Write to primary
primary_db = create_engine("postgresql://primary:5432/txt2create")

# Read from replicas (round-robin)
replica_dbs = [
    create_engine("postgresql://replica-1:5432/txt2create"),
    create_engine("postgresql://replica-2:5432/txt2create"),
]

def get_read_db():
    return random.choice(replica_dbs)

# Usage
async def get_user(user_id):
    # Read from replica (eventual consistency OK)
    db = get_read_db()
    return await db.query(User).filter_by(id=user_id).first()

async def update_user(user_id, data):
    # Write to primary
    await primary_db.query(User).filter_by(id=user_id).update(data)
```

**4. Optimistic locking (for concurrent updates)**
```python
class Job(Base):
    __tablename__ = 'jobs'
    id = Column(UUID, primary_key=True)
    status = Column(String)
    version = Column(Integer, default=0)  # Version number

async def update_job_status(job_id, new_status):
    while True:
        job = await db.query(Job).filter_by(id=job_id).first()
        old_version = job.version

        # Try to update
        result = await db.query(Job).filter_by(
            id=job_id,
            version=old_version  # Only update if version hasn't changed
        ).update({
            'status': new_status,
            'version': old_version + 1
        })

        if result.rowcount > 0:
            await db.commit()
            break  # Success
        else:
            # Version changed (someone else updated), retry
            await db.rollback()
            await asyncio.sleep(0.1)
```

**5. Database monitoring**
```sql
-- Active connections
SELECT count(*) FROM pg_stat_activity;

-- Slow queries
SELECT query, mean_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Lock waits
SELECT * FROM pg_locks WHERE NOT granted;
```

**6. Auto-scaling (Kubernetes)**
```yaml
# Scale up when connection pool fills
metrics:
- type: Pods
  pods:
    metric:
      name: db_connection_pool_usage
    target:
      type: AverageValue
      averageValue: "80"  # Scale when 80% full
```

---

## 7. SECURITY & PRIVACY

### Q: How do you secure API endpoints?

**A:** Multi-layer security:

**1. Authentication (JWT)**
```python
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=["HS256"]
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(401, "Invalid token")

        user = await db.get_user(user_id)
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.JWTError:
        raise HTTPException(401, "Invalid token")

@app.post("/generate/image")
async def generate_image(
    prompt: str,
    user: User = Depends(get_current_user)  # Requires auth
):
    # user is authenticated
    return await create_job(user.id, prompt)
```

**2. Authorization (RBAC)**
```python
from enum import Enum

class Role(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

PERMISSIONS = {
    Role.FREE: {
        "generate_image": True,
        "generate_video": False,
        "train_lora": False,
    },
    Role.PRO: {
        "generate_image": True,
        "generate_video": True,
        "train_lora": True,
    },
    Role.ENTERPRISE: {
        "generate_image": True,
        "generate_video": True,
        "train_lora": True,
        "api_access": True,
    }
}

def require_permission(permission: str):
    def decorator(func):
        async def wrapper(user: User = Depends(get_current_user), *args, **kwargs):
            if not PERMISSIONS[user.role].get(permission, False):
                raise HTTPException(403, f"Permission denied: {permission}")
            return await func(user=user, *args, **kwargs)
        return wrapper
    return decorator

@app.post("/generate/video")
@require_permission("generate_video")
async def generate_video(prompt: str, user: User = Depends(get_current_user)):
    # Only PRO and ENTERPRISE users can access
    return await create_video_job(user.id, prompt)
```

**3. Rate limiting**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    storage_uri="redis://localhost:6379"
)

@app.post("/generate/image")
@limiter.limit("10/hour")  # 10 requests per hour per IP
async def generate_image(request: Request, prompt: str):
    return await create_job(prompt)

# Tier-based rate limiting
def get_rate_limit(user: User):
    limits = {
        Role.FREE: "10/hour",
        Role.PRO: "100/hour",
        Role.ENTERPRISE: "1000/hour",
    }
    return limits[user.role]

@app.post("/generate/image")
async def generate_image(
    request: Request,
    prompt: str,
    user: User = Depends(get_current_user)
):
    # Check rate limit
    limit = get_rate_limit(user)
    if not limiter.check(limit, user.id):
        raise HTTPException(429, "Rate limit exceeded")

    return await create_job(user.id, prompt)
```

**4. Input validation**
```python
from pydantic import BaseModel, Field, validator

class ImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    steps: int = Field(30, ge=10, le=150)
    width: int = Field(1024, ge=512, le=2048)
    height: int = Field(1024, ge=512, le=2048)

    @validator('width', 'height')
    def check_dimensions(cls, v):
        # Only allow multiples of 64 (SD requirement)
        if v % 64 != 0:
            raise ValueError("Dimensions must be multiples of 64")
        return v

    @validator('prompt')
    def check_prompt(cls, v):
        # Prevent injection attacks
        if '<script>' in v.lower():
            raise ValueError("Invalid prompt")
        return v

@app.post("/generate/image")
async def generate_image(request: ImageRequest):
    # request is validated
    return await create_job(request.prompt, request.steps)
```

**5. HTTPS/TLS**
```yaml
# Kubernetes ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: api-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.txt2create.com
    secretName: api-tls
  rules:
  - host: api.txt2create.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api
            port:
              number: 8000
```

**6. CORS (Cross-Origin Resource Sharing)**
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://txt2create.com",
        "https://app.txt2create.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

### Q: How do you protect user data privacy?

**A:** Privacy measures:

**1. Encryption at rest**
```python
# Database: PostgreSQL with encrypted storage (AWS RDS encryption)
# S3: Server-side encryption (SSE-S3 or SSE-KMS)

# Encrypt sensitive fields
from cryptography.fernet import Fernet

cipher = Fernet(ENCRYPTION_KEY)

class User(Base):
    __tablename__ = 'users'
    id = Column(UUID, primary_key=True)
    email_encrypted = Column(LargeBinary)

    @property
    def email(self):
        return cipher.decrypt(self.email_encrypted).decode()

    @email.setter
    def email(self, value):
        self.email_encrypted = cipher.encrypt(value.encode())
```

**2. Encryption in transit**
- HTTPS (TLS 1.3) for all API calls
- Database connections over SSL
- Internal service mesh (mTLS with Istio)

**3. Data minimization**
```python
# Don't store more than necessary
class JobRecord:
    id: UUID
    user_id: UUID
    prompt: str  # Store prompt (needed for cache)
    # DON'T store: IP address, user agent, full request body
    created_at: datetime
```

**4. Anonymization**
```python
# Analytics: anonymize user IDs
import hashlib

def anonymize_user_id(user_id: str) -> str:
    return hashlib.sha256(
        (user_id + SALT).encode()
    ).hexdigest()

# Store in analytics DB
analytics_db.log({
    "user_id_hash": anonymize_user_id(user.id),
    "action": "image_generated",
    "timestamp": datetime.now()
})
```

**5. Data retention**
```python
# Delete old data
async def cleanup_old_data():
    # Delete jobs older than 90 days
    await db.delete(Job).where(
        Job.created_at < datetime.now() - timedelta(days=90)
    )

    # Delete S3 objects
    old_objects = s3.list_objects(
        Prefix="images/",
        Filter={"LastModified": {"Before": "90 days ago"}}
    )
    for obj in old_objects:
        s3.delete_object(Key=obj['Key'])

# Run daily
celery.beat_schedule = {
    'cleanup-old-data': {
        'task': 'tasks.cleanup_old_data',
        'schedule': crontab(hour=2, minute=0),  # 2 AM daily
    }
}
```

**6. GDPR compliance**
```python
# Right to be forgotten
@app.post("/user/delete-account")
async def delete_account(user: User = Depends(get_current_user)):
    # Delete all user data
    await db.delete(Job).where(Job.user_id == user.id)
    await db.delete(User).where(User.id == user.id)

    # Delete S3 objects
    objects = s3.list_objects(Prefix=f"images/{user.id}/")
    for obj in objects:
        s3.delete_object(Key=obj['Key'])

    # Anonymize analytics
    await analytics_db.update(
        user_id_hash=anonymize_user_id(user.id)
    ).where(user_id == user.id)

    return {"message": "Account deleted"}

# Data export (GDPR)
@app.get("/user/export-data")
async def export_data(user: User = Depends(get_current_user)):
    jobs = await db.query(Job).filter_by(user_id=user.id).all()

    export = {
        "user": {
            "email": user.email,
            "created_at": user.created_at.isoformat()
        },
        "jobs": [
            {
                "id": str(job.id),
                "prompt": job.prompt,
                "created_at": job.created_at.isoformat(),
                "result_url": job.result_url
            }
            for job in jobs
        ]
    }

    return export  # JSON download
```

---

## 8. COST & ECONOMICS

### Q: How much does it cost to generate one image/video/audio?

**A:** Cost breakdown (using AWS pricing):

**Text-to-Image (SD XL, 30 steps, 1024×1024):**
```
GPU: A100 40GB @ $2.50/hour
Inference time: 30 seconds = 0.0083 hours
GPU cost: $2.50 × 0.0083 = $0.021

Additional costs:
- vLLM (prompt enhancement): $0.005
- Storage (S3): $0.001
- Bandwidth: $0.002
- Compute (worker): $0.001

Total: $0.03 per image
```

**Text-to-Video (8 seconds, 720p, 30 FPS = 240 frames):**
```
- SD keyframes (8 frames): 8 × $0.021 = $0.168
- FILM interpolation (232 frames): $0.080
- VAE compression: $0.050
- Audio (optional): $0.015
- Storage: $0.010
- Bandwidth: $0.020

Total: $0.34 per video
```

**Text-to-Audio (30 seconds, MusicGen):**
```
GPU: A100 @ $2.50/hour
Inference time: 20 seconds = 0.0056 hours
GPU cost: $2.50 × 0.0056 = $0.014

Additional:
- LLM processing: $0.003
- Storage: $0.001
- Bandwidth: $0.002

Total: $0.02 per audio clip
```

**Video Captioning (60 seconds video):**
```
- Keyframe extraction: $0.001
- BLIP-2 (8 frames): 8 × $0.002 = $0.016
- LLM aggregation (vLLM): $0.010
- Whisper (audio transcription): $0.015

Total: $0.04 per video
```

**Virtual Avatar (3D rigged):**
```
- SD face generation: $0.021
- Multi-view generation (4 views): 4 × $0.021 = $0.084
- 3D reconstruction (PIFuHD): $0.150
- Rigging: $0.080
- Storage: $0.005

Total: $0.34 per avatar
```

**Monthly cost (10,000 images/day):**
```
10,000 images × $0.03 = $300/day
$300 × 30 days = $9,000/month (just generation)

Infrastructure:
- GPU cluster (8× A100): $14,600/month
- Database, Redis, workers: $4,000/month
- Storage, bandwidth: $2,000/month
- Monitoring, misc: $500/month

Total: ~$30,000/month for 300k images
```

**Cost per active user (assuming 30 images/month):**
```
$0.03 × 30 = $0.90/user/month
```

**Pricing tiers:**
```
Free: 10 images/month → $0.30 cost (loss leader)
Pro: $20/month → 200 images ($6 cost) → $14 profit
Enterprise: $200/month → 2000 images ($60 cost) → $140 profit
```

---

### Q: How can costs be reduced?

**A:** Optimization strategies:

**1. Spot instances (40-70% discount)**
```python
# Use spot for batch workloads
batch_worker_pool = EC2SpotInstances(
    instance_type="p4d.24xlarge",
    max_price=1.50,  # 60% of on-demand ($2.50)
    interruption_handler=graceful_shutdown
)

# Expected savings: $14,600 → $6,000/month
```

**2. Model quantization (2x throughput)**
```python
# INT8 models → half the GPU time
# $0.021 → $0.011 per image
# Savings: $4,500/month (for 300k images)
```

**3. Caching (avoid duplicate generations)**
```python
# 40% cache hit rate
# 300k generations → 120k cached, 180k generated
# Cost: 180k × $0.03 = $5,400 (instead of $9,000)
# Savings: $3,600/month
```

**4. Tiered storage (archive old content)**
```python
# S3 Glacier for >90 day content
# $0.023/GB → $0.004/GB
# For 50 TB: $1,150 → $200
# Savings: $950/month
```

**5. GPU sharing (MIG)**
```python
# Run 4 models on 1 A100 instead of 4 A100s
# 12 GPUs → 4 GPUs
# $14,600 → $4,900
# Savings: $9,700/month
```

**6. Reserved instances (1-year commit, 40% discount)**
```python
# For baseline GPU load (4 GPUs always needed)
# $2.50/hour → $1.50/hour
# 4 GPUs × $1.50 × 730 hours = $4,380 (instead of $7,300)
# Savings: $2,920/month
```

**Total potential savings:**
```
Spot instances:       $8,600
Quantization:         $4,500
Caching:              $3,600
GPU sharing:          $9,700
Reserved instances:   $2,920
Tiered storage:       $950

Total: $30,270/month → $15,000/month (50% reduction)
```

---

## 9. DEVELOPMENT & OPERATIONS

### Q: How do you deploy new models without downtime?

**A:** Blue-green deployment:

```
Step 1: Current state (Blue)
┌─────────────────────┐
│ TorchServe v1       │
│ Model: sd-xl-1.0    │  ← 100% traffic
│ Pods: 4             │
└─────────────────────┘

Step 2: Deploy new version (Green)
┌─────────────────────┐
│ TorchServe v1       │
│ Model: sd-xl-1.0    │  ← 100% traffic
│ Pods: 4             │
└─────────────────────┘

┌─────────────────────┐
│ TorchServe v2       │
│ Model: sd-xl-1.1    │  ← 0% traffic (testing)
│ Pods: 2             │
└─────────────────────┘

Step 3: Canary (route 10% traffic to green)
TorchServe v1: 90% traffic
TorchServe v2: 10% traffic (monitor for errors)

Step 4: Gradual rollout
v1: 70%, v2: 30%
v1: 50%, v2: 50%
v1: 30%, v2: 70%
v1: 0%, v2: 100%

Step 5: Remove old version
┌─────────────────────┐
│ TorchServe v2       │
│ Model: sd-xl-1.1    │  ← 100% traffic
│ Pods: 4             │
└─────────────────────┘
```

**Kubernetes implementation:**

```yaml
# Deployment v1 (blue)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torchserve-v1
spec:
  replicas: 4
  selector:
    matchLabels:
      app: torchserve
      version: v1
  template:
    metadata:
      labels:
        app: torchserve
        version: v1
    spec:
      containers:
      - name: torchserve
        image: torchserve:v1-sd-xl-1.0

---
# Deployment v2 (green)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: torchserve-v2
spec:
  replicas: 2
  selector:
    matchLabels:
      app: torchserve
      version: v2
  template:
    metadata:
      labels:
        app: torchserve
        version: v2
    spec:
      containers:
      - name: torchserve
        image: torchserve:v2-sd-xl-1.1

---
# Service (routes to both)
apiVersion: v1
kind: Service
metadata:
  name: torchserve
spec:
  selector:
    app: torchserve  # Matches both v1 and v2
  ports:
  - port: 8080
```

**Traffic splitting (Istio):**

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: torchserve
spec:
  hosts:
  - torchserve
  http:
  - match:
    - headers:
        user-agent:
          regex: ".*test.*"  # Test traffic → v2
    route:
    - destination:
        host: torchserve
        subset: v2
  - route:
    - destination:
        host: torchserve
        subset: v1
      weight: 90  # 90% to v1
    - destination:
        host: torchserve
        subset: v2
      weight: 10  # 10% to v2
```

**Rollback:**
```bash
# If v2 has issues, instant rollback
kubectl patch virtualservice torchserve --type=merge -p '
{
  "spec": {
    "http": [{
      "route": [{"destination": {"host": "torchserve", "subset": "v1"}, "weight": 100}]
    }]
  }
}'

# Or scale down v2
kubectl scale deployment torchserve-v2 --replicas=0
```

---

### Q: How do you handle model versioning?

**A:** Model registry + versioning:

**1. Model storage structure:**
```
s3://txt2create-models/
├── stable-diffusion/
│   ├── v1.0/
│   │   ├── model.safetensors
│   │   ├── config.json
│   │   └── metadata.json
│   ├── v1.1/
│   │   └── ...
│   └── latest → v1.1/  # Symlink
├── llm/
│   ├── llama-3-8b-v1/
│   └── llama-3-8b-v2/
└── vae/
    └── ...
```

**2. Metadata tracking:**
```json
// metadata.json
{
  "model_name": "stable-diffusion-xl",
  "version": "1.1",
  "created_at": "2025-12-01T00:00:00Z",
  "trained_by": "user@example.com",
  "base_model": "stabilityai/sdxl-1.0",
  "fine_tuning": "LoRA rank 64",
  "training_steps": 10000,
  "metrics": {
    "fid_score": 15.2,
    "clip_score": 0.87
  },
  "deployment_status": "production",
  "checksum": "sha256:abc123..."
}
```

**3. Version resolution:**
```python
from enum import Enum

class ModelVersion(Enum):
    LATEST = "latest"
    STABLE = "stable"
    CANARY = "canary"

# Config
MODEL_VERSIONS = {
    "stable-diffusion": {
        ModelVersion.LATEST: "v1.1",
        ModelVersion.STABLE: "v1.0",
        ModelVersion.CANARY: "v1.2-beta"
    }
}

def get_model_path(model_name: str, version: ModelVersion = ModelVersion.STABLE):
    resolved_version = MODEL_VERSIONS[model_name][version]
    return f"s3://txt2create-models/{model_name}/{resolved_version}/"

# Usage
model_path = get_model_path("stable-diffusion", ModelVersion.STABLE)
# Returns: s3://txt2create-models/stable-diffusion/v1.0/
```

**4. A/B testing:**
```python
import random

def select_model_version(user_id: str, experiment: str):
    # Consistent hashing (same user always gets same version)
    hash_val = int(hashlib.md5(f"{user_id}:{experiment}".encode()).hexdigest(), 16)

    if hash_val % 100 < 10:
        # 10% of users get canary
        return ModelVersion.CANARY
    else:
        # 90% get stable
        return ModelVersion.STABLE

# Use in generation
model_version = select_model_version(user.id, "sdxl-1.2-test")
model = load_model("stable-diffusion", model_version)
```

**5. Quality regression testing:**
```python
# Before deploying new version, run test suite
async def test_model_quality(model_path: str):
    test_prompts = [
        "a cat in a garden",
        "cyberpunk city at night",
        # ... 100 test prompts
    ]

    results = []
    for prompt in test_prompts:
        image = await generate(model_path, prompt)

        # Measure quality
        clip_score = await compute_clip_score(image, prompt)
        fid_score = await compute_fid_score(image)

        results.append({
            "prompt": prompt,
            "clip_score": clip_score,
            "fid_score": fid_score
        })

    avg_clip = sum(r["clip_score"] for r in results) / len(results)
    avg_fid = sum(r["fid_score"] for r in results) / len(results)

    # Compare to baseline
    if avg_clip < BASELINE_CLIP * 0.95:
        raise ModelQualityError("CLIP score regression")

    return {"clip": avg_clip, "fid": avg_fid}

# CI/CD pipeline
# 1. Train model → v1.2
# 2. Upload to S3
# 3. Run quality tests
# 4. If pass, mark as canary
# 5. Monitor in production
# 6. If good, promote to stable
```

---

This comprehensive Q&A covers the core concepts, technologies, and design decisions behind the TXT2CREATE platform. Each answer provides both the "what" and "why" to help understand the architecture deeply.

