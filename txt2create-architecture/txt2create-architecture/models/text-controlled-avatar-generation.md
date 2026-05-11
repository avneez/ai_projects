# Text-Controlled Avatar Generation with StyleGAN

## The Challenge: GANs Don't Naturally Accept Text

**Problem**: StyleGAN3 (and GANs in general) are trained to generate from **random noise vectors**, NOT text prompts.

```
Traditional GAN:
Random noise z ∈ R^512 → Generator → Face image

What we want:
Text "blonde hair, blue eyes" → Generator → Face matching description
```

**Solution**: We need to **bridge the gap** between text and GAN's latent space.

---

## COMPLETE TEXT-TO-AVATAR PIPELINE

### Overview of Methods

We use **3 complementary approaches** to achieve text-controlled generation:

```
┌─────────────────────────────────────────────────────────────┐
│              METHOD 1: CLIP-Guided Optimization             │
│  Use CLIP to optimize latent code to match text            │
│  ✓ High quality, exact text matching                        │
│  ✗ Slow (requires optimization loop)                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              METHOD 2: Text-to-Latent Mapping               │
│  Train a neural network: Text → GAN latent code            │
│  ✓ Fast (single forward pass)                              │
│  ✗ Requires training dataset                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              METHOD 3: Attribute-Based Control              │
│  Discover semantic directions in latent space              │
│  ✓ Precise control over specific attributes                │
│  ✗ Limited to predefined attributes                         │
└─────────────────────────────────────────────────────────────┘
```

---

## METHOD 1: CLIP-GUIDED OPTIMIZATION (Primary Method)

### How It Works

**CLIP** (Contrastive Language-Image Pre-training) creates a shared embedding space for images and text:

```
┌──────────────────────────────────────────────────────┐
│              CLIP Shared Embedding Space             │
│                                                      │
│   Text: "blonde hair"  ──→  [embedding_text]        │
│                                  ↓                   │
│                            Cosine similarity         │
│                                  ↓                   │
│   Image: [blonde woman] ──→ [embedding_image]       │
│                                                      │
│   If image matches text → High similarity (0.8-0.9) │
│   If mismatch → Low similarity (0.1-0.3)            │
└──────────────────────────────────────────────────────┘
```

### Complete Architecture

```
Step 1: Setup
┌────────────────────────────────────────┐
│ Load Pre-trained Models                │
│ - StyleGAN3 (trained on FFHQ faces)    │
│ - CLIP (ViT-B/32 or ViT-L/14)         │
└────────────────────────────────────────┘

Step 2: Initialize Random Latent
┌────────────────────────────────────────┐
│ z ~ N(0, I) ∈ R^512                    │
│ Random starting point in latent space  │
└────────────────────────────────────────┘

Step 3: Optimization Loop (300 steps)
┌────────────────────────────────────────┐
│ For step in 1..300:                    │
│                                        │
│   1. Generate image from z             │
│      z → StyleGAN3 → image             │
│                                        │
│   2. Encode with CLIP                  │
│      text → CLIP → text_features       │
│      image → CLIP → image_features     │
│                                        │
│   3. Compute similarity loss           │
│      loss = -cosine_similarity(        │
│         text_features, image_features  │
│      )                                 │
│                                        │
│   4. Add regularization                │
│      loss += λ * ||z||²  (stay realistic)│
│                                        │
│   5. Backpropagate & update z          │
│      z ← z - lr * ∇z(loss)             │
│                                        │
│   6. Project z to valid range          │
│      z ← clip(z, -2, 2)                │
└────────────────────────────────────────┘

Step 4: Final Generation
┌────────────────────────────────────────┐
│ optimized_z → StyleGAN3 → final_image  │
│ Image now matches text description!    │
└────────────────────────────────────────┘
```

### Implementation with Detailed Comments

```python
import torch
import torch.nn.functional as F
from torchvision import transforms
import clip
from stylegan3 import Generator
from PIL import Image

class TextToAvatarCLIP:
    """
    Text-controlled avatar generation using CLIP-guided StyleGAN3 optimization
    """

    def __init__(self, stylegan_checkpoint='stylegan3-ffhq-1024x1024.pkl'):
        # Load StyleGAN3 generator
        print("Loading StyleGAN3...")
        self.G = Generator(
            z_dim=512,        # Latent dimension
            c_dim=0,          # No class conditioning
            w_dim=512,        # W latent dimension
            img_resolution=1024,
            img_channels=3
        ).cuda()

        # Load pretrained weights
        checkpoint = torch.load(stylegan_checkpoint)
        self.G.load_state_dict(checkpoint['G_ema'])
        self.G.eval()

        # Load CLIP
        print("Loading CLIP...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device="cuda")

        # CLIP preprocessing for generated images
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )

    def generate_from_text(
        self,
        text_prompt,
        num_steps=300,
        learning_rate=0.1,
        lambda_reg=0.01
    ):
        """
        Generate avatar from text description

        Args:
            text_prompt: Text description (e.g., "A young woman with blonde hair and blue eyes")
            num_steps: Number of optimization steps
            learning_rate: Learning rate for optimization
            lambda_reg: Regularization strength

        Returns:
            PIL Image of generated avatar
        """

        # Step 1: Encode text with CLIP
        print(f"Encoding text: '{text_prompt}'")
        text_tokens = clip.tokenize([text_prompt]).cuda()

        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Step 2: Initialize random latent code
        z = torch.randn(1, 512, device='cuda', requires_grad=True)

        # Optimizer
        optimizer = torch.optim.Adam([z], lr=learning_rate)

        # Step 3: Optimization loop
        print(f"Optimizing for {num_steps} steps...")
        best_similarity = -float('inf')
        best_z = None

        for step in range(num_steps):
            optimizer.zero_grad()

            # Generate image from current latent
            # z → w (mapping network)
            with torch.no_grad():
                w = self.G.mapping(z, None)

            # w → image (synthesis network)
            # Note: We don't use no_grad here because we need gradients w.r.t. z
            image = self.G.synthesis(w, noise_mode='const')

            # image is in [-1, 1], shape [1, 3, 1024, 1024]

            # Step 4: Prepare image for CLIP
            # Resize to CLIP input size (224x224)
            image_224 = F.interpolate(image, size=224, mode='bicubic', align_corners=False)

            # Normalize from [-1, 1] to [0, 1]
            image_01 = (image_224 + 1) / 2

            # Apply CLIP normalization
            image_normalized = self.clip_normalize(image_01)

            # Step 5: Encode image with CLIP
            image_features = self.clip_model.encode_image(image_normalized)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Step 6: Compute CLIP similarity loss
            # We want to MAXIMIZE similarity, so minimize negative similarity
            similarity = (image_features * text_features).sum()
            clip_loss = -similarity

            # Step 7: Regularization
            # Prevent z from going too far from typical latent distribution
            # This keeps faces realistic
            latent_reg = lambda_reg * (z ** 2).sum()

            # Total loss
            total_loss = clip_loss + latent_reg

            # Step 8: Backpropagation
            total_loss.backward()

            # Step 9: Update z
            optimizer.step()

            # Step 10: Optional - project z to reasonable range
            # Most good latents are within [-2, 2]
            with torch.no_grad():
                z.data = torch.clamp(z.data, -2, 2)

            # Track best result
            if similarity.item() > best_similarity:
                best_similarity = similarity.item()
                best_z = z.data.clone()

            # Print progress
            if step % 50 == 0:
                print(f"Step {step}/{num_steps} | "
                      f"Similarity: {similarity.item():.4f} | "
                      f"Loss: {total_loss.item():.4f}")

        # Step 11: Generate final image with best latent
        print(f"Best similarity: {best_similarity:.4f}")
        print("Generating final image...")

        with torch.no_grad():
            w_best = self.G.mapping(best_z, None)
            final_image = self.G.synthesis(w_best, noise_mode='const')

            # Convert to PIL Image
            final_image = (final_image + 1) / 2  # [-1, 1] → [0, 1]
            final_image = final_image.squeeze(0).permute(1, 2, 0).cpu()
            final_image = (final_image * 255).clamp(0, 255).numpy().astype('uint8')

        return Image.fromarray(final_image)

# Usage
generator = TextToAvatarCLIP(stylegan_checkpoint='stylegan3-ffhq-1024x1024.pkl')

# Generate avatar from text
avatar = generator.generate_from_text(
    "A young woman with long blonde hair, blue eyes, and a warm smile",
    num_steps=300,
    learning_rate=0.1
)

avatar.save("avatar_blonde_woman.png")
```

---

## METHOD 1 ADVANCED: Multi-Attribute Control

### Handle Complex Prompts with Multiple Attributes

```python
class MultiAttributeAvatarGenerator:
    """
    Advanced text-to-avatar with separate control over multiple attributes
    """

    def __init__(self):
        self.generator = TextToAvatarCLIP()

    def parse_attributes(self, text_prompt):
        """
        Parse text prompt into separate attributes using LLM

        Example:
        Input: "A young Asian woman with long black hair, brown eyes, wearing glasses"
        Output: {
            'age': 'young',
            'ethnicity': 'Asian',
            'gender': 'woman',
            'hair_style': 'long',
            'hair_color': 'black',
            'eye_color': 'brown',
            'accessories': 'glasses'
        }
        """
        # Use Llama 3 to extract structured attributes
        from vllm import LLM, SamplingParams

        llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct")

        extraction_prompt = f"""Extract facial attributes from this description into JSON format:
"{text_prompt}"

Output as JSON with keys: age, gender, ethnicity, hair_style, hair_color, eye_color, facial_hair, accessories, expression

JSON:"""

        output = llm.generate([extraction_prompt], SamplingParams(temperature=0))[0]
        attributes = json.loads(output.outputs[0].text)

        return attributes

    def generate_with_attributes(self, text_prompt, num_steps=500):
        """
        Generate avatar with multi-stage optimization for different attributes
        """
        # Parse attributes
        attributes = self.parse_attributes(text_prompt)

        # Stage 1: Generate base face (age, gender, ethnicity)
        base_prompt = f"{attributes['age']} {attributes['ethnicity']} {attributes['gender']}"
        print(f"Stage 1: Generating base face - {base_prompt}")

        # Initialize with base attributes
        z = self._optimize_for_text(base_prompt, num_steps=200)

        # Stage 2: Add hair attributes
        if 'hair_style' in attributes or 'hair_color' in attributes:
            hair_prompt = f"{attributes.get('hair_style', '')} {attributes.get('hair_color', '')} hair"
            print(f"Stage 2: Adding hair - {hair_prompt}")
            z = self._optimize_for_text(hair_prompt, num_steps=150, init_z=z)

        # Stage 3: Add eye color
        if 'eye_color' in attributes:
            eye_prompt = f"{attributes['eye_color']} eyes"
            print(f"Stage 3: Adding eyes - {eye_prompt}")
            z = self._optimize_for_text(eye_prompt, num_steps=100, init_z=z)

        # Stage 4: Add accessories
        if 'accessories' in attributes:
            acc_prompt = f"wearing {attributes['accessories']}"
            print(f"Stage 4: Adding accessories - {acc_prompt}")
            z = self._optimize_for_text(acc_prompt, num_steps=100, init_z=z)

        # Stage 5: Add expression
        if 'expression' in attributes:
            exp_prompt = f"{attributes['expression']} expression"
            print(f"Stage 5: Adding expression - {exp_prompt}")
            z = self._optimize_for_text(exp_prompt, num_steps=50, init_z=z)

        # Generate final image
        with torch.no_grad():
            w = self.generator.G.mapping(z, None)
            final_image = self.generator.G.synthesis(w, noise_mode='const')

        return self._tensor_to_pil(final_image)

    def _optimize_for_text(self, text, num_steps, init_z=None):
        """Helper function to optimize latent for specific text"""
        if init_z is None:
            z = torch.randn(1, 512, device='cuda', requires_grad=True)
        else:
            z = init_z.clone().detach().requires_grad_(True)

        # ... optimization loop similar to main method ...

        return z

# Usage
advanced_gen = MultiAttributeAvatarGenerator()

avatar = advanced_gen.generate_with_attributes(
    "A young Asian woman with long black hair, brown eyes, wearing glasses, gentle smile"
)
avatar.save("avatar_detailed.png")
```

---

## METHOD 2: TEXT-TO-LATENT MAPPING NETWORK

### Train a Direct Mapping: Text → GAN Latent

**Architecture:**
```
┌────────────────────────────────────────────────────┐
│              Text-to-Latent Network                │
│                                                    │
│  Text Prompt                                       │
│       ↓                                            │
│  Text Encoder (CLIP or BERT)                       │
│       ↓                                            │
│  Text Embedding [768-dim]                          │
│       ↓                                            │
│  MLP Layers:                                       │
│    Linear(768 → 1024) + ReLU                       │
│    Linear(1024 → 1024) + ReLU                      │
│    Linear(1024 → 512)  # Output: w latent          │
│       ↓                                            │
│  StyleGAN3 Latent Code w ∈ R^512                   │
│       ↓                                            │
│  StyleGAN3 Generator                               │
│       ↓                                            │
│  Generated Face Image                              │
└────────────────────────────────────────────────────┘
```

### Training Process

```python
class TextToLatentMapper(nn.Module):
    """
    Neural network that maps text to StyleGAN latent space
    """

    def __init__(self, text_dim=768, latent_dim=512):
        super().__init__()

        # Text encoder (frozen CLIP)
        self.clip_model, _ = clip.load("ViT-L/14", device="cuda")
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Mapping network (trainable)
        self.mapper = nn.Sequential(
            nn.Linear(text_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(1024, 512),
        )

        # StyleGAN3 generator (frozen)
        self.G = load_stylegan3()
        for param in self.G.parameters():
            param.requires_grad = False

    def forward(self, text_prompt):
        """
        text_prompt → latent code → image
        """
        # Encode text
        text_tokens = clip.tokenize([text_prompt]).cuda()
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)

        # Map to latent code
        w = self.mapper(text_features.float())

        # Generate image
        image = self.G.synthesis(w.unsqueeze(1).repeat(1, 18, 1), noise_mode='const')

        return image, w

# Training loop
def train_text_to_latent(dataset, num_epochs=100):
    """
    Train text-to-latent mapper

    Dataset format:
    [
        ("young woman with blonde hair", latent_code_1),
        ("old man with gray beard", latent_code_2),
        ...
    ]
    """
    model = TextToLatentMapper().cuda()
    optimizer = torch.optim.Adam(model.mapper.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        for text, target_latent in dataset:
            # Forward pass
            generated_image, predicted_latent = model(text)

            # Loss 1: Latent space distance
            latent_loss = F.mse_loss(predicted_latent, target_latent)

            # Loss 2: Image space distance (perceptual loss)
            target_image = model.G.synthesis(target_latent, noise_mode='const')
            perceptual_loss = compute_perceptual_loss(generated_image, target_image)

            # Loss 3: CLIP consistency
            clip_loss = compute_clip_loss(generated_image, text)

            # Total loss
            loss = latent_loss + 0.1 * perceptual_loss + 0.5 * clip_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Advantage: Fast inference (single forward pass, ~0.2s)
# Disadvantage: Requires training dataset of (text, latent) pairs
```

---

## METHOD 3: ATTRIBUTE-BASED CONTROL (InterfaceGAN / GANSpace)

### Discover Semantic Directions in Latent Space

**Concept**: Find directions in StyleGAN's latent space that correspond to specific attributes.

```
Example: "Blonde Hair" direction

w_blonde = w_base + α * direction_blonde
                     ↑
                     α controls strength:
                     α = 0  → original hair color
                     α = +3 → very blonde
                     α = -3 → very dark hair
```

### Pre-computed Semantic Directions

```python
class AttributeController:
    """
    Control specific facial attributes using pre-discovered directions
    """

    def __init__(self):
        self.G = load_stylegan3()

        # Pre-computed directions (from InterFaceGAN or GANSpace)
        # These are discovered by analyzing many generated faces
        self.directions = {
            'age': torch.load('directions/age.pt'),          # Young ← → Old
            'gender': torch.load('directions/gender.pt'),    # Feminine ← → Masculine
            'hair_color': torch.load('directions/blonde.pt'), # Dark ← → Blonde
            'smile': torch.load('directions/smile.pt'),      # Neutral ← → Smiling
            'glasses': torch.load('directions/glasses.pt'),  # No glasses ← → Glasses
            'ethnicity': torch.load('directions/asian.pt'),  # Caucasian ← → Asian
            'eye_size': torch.load('directions/eyes.pt'),    # Small ← → Large eyes
        }

    def text_to_attributes(self, text_prompt):
        """
        Parse text to attribute values using LLM

        Example:
        Input: "young blonde woman with glasses"
        Output: {
            'age': -2.0,      # Young (negative = younger)
            'gender': -1.5,   # Female (negative = feminine)
            'hair_color': 3.0, # Blonde (positive = lighter)
            'glasses': 2.0    # Wearing glasses
        }
        """
        # Use LLM to extract attribute values
        prompt = f"""Given this face description: "{text_prompt}"

Extract attribute values on scale -5 to +5:
- age: -5 (very young) to +5 (very old)
- gender: -5 (very feminine) to +5 (very masculine)
- hair_color: -5 (very dark) to +5 (very light/blonde)
- smile: -5 (frowning) to +5 (big smile)
- glasses: 0 (no glasses) to 5 (strong glasses)

Output as JSON:"""

        llm = LLM("meta-llama/Meta-Llama-3-8B-Instruct")
        output = llm.generate([prompt], SamplingParams(temperature=0))[0]
        attributes = json.loads(output.outputs[0].text)

        return attributes

    def generate_from_text(self, text_prompt):
        """
        Generate avatar by applying attribute directions
        """
        # Parse text to attributes
        attributes = self.text_to_attributes(text_prompt)

        # Start with random base latent
        z = torch.randn(1, 512, device='cuda')
        w = self.G.mapping(z, None)

        # Apply each attribute direction
        for attr_name, attr_value in attributes.items():
            if attr_name in self.directions and attr_value != 0:
                direction = self.directions[attr_name].cuda()
                w = w + attr_value * direction

        # Generate image
        image = self.G.synthesis(w, noise_mode='const')

        return image

# Usage
controller = AttributeController()

# Method 1: Direct attribute control
avatar = controller.generate_from_text(
    "young blonde woman with glasses and a big smile"
)

# Method 2: Fine-grained control
z = torch.randn(1, 512, device='cuda')
w = controller.G.mapping(z, None)

# Apply specific modifications
w = w + 3.0 * controller.directions['hair_color']   # Make blonde
w = w - 2.0 * controller.directions['age']          # Make younger
w = w + 2.5 * controller.directions['smile']        # Add smile
w = w + 1.5 * controller.directions['glasses']      # Add glasses

image = controller.G.synthesis(w, noise_mode='const')
```

---

## PRODUCTION PIPELINE: HYBRID APPROACH

### Combine All Methods for Best Results

```python
class ProductionAvatarGenerator:
    """
    Production-ready avatar generation combining all methods
    """

    def __init__(self):
        # Method 1: CLIP-guided optimization
        self.clip_generator = TextToAvatarCLIP()

        # Method 3: Attribute control
        self.attribute_controller = AttributeController()

    def generate_avatar(self, text_prompt, method='hybrid'):
        """
        Generate avatar using specified method

        Methods:
        - 'clip': CLIP-guided optimization (slow, best quality)
        - 'attributes': Attribute-based (fast, limited control)
        - 'hybrid': Combine both (recommended)
        """

        if method == 'clip':
            # Pure CLIP optimization (300 steps, ~60s)
            return self.clip_generator.generate_from_text(text_prompt, num_steps=300)

        elif method == 'attributes':
            # Pure attribute control (instant, ~0.2s)
            return self.attribute_controller.generate_from_text(text_prompt)

        elif method == 'hybrid':
            # Hybrid: Attributes for initialization + CLIP refinement

            # Step 1: Quick initialization with attributes (0.2s)
            print("Step 1: Initializing with attributes...")
            z_init = self._generate_with_attributes(text_prompt)

            # Step 2: Refine with CLIP (100 steps, ~20s)
            print("Step 2: Refining with CLIP...")
            final_image = self.clip_generator.generate_from_text(
                text_prompt,
                num_steps=100,
                init_z=z_init  # Start from attribute-based initialization
            )

            return final_image

    def _generate_with_attributes(self, text_prompt):
        """Initialize latent using attributes"""
        # ... attribute-based generation ...
        return z_latent

# Usage
gen = ProductionAvatarGenerator()

# For production (balanced speed/quality)
avatar = gen.generate_avatar(
    "A young Asian woman with long black hair, brown eyes, wearing glasses, gentle smile",
    method='hybrid'  # 20-25 seconds total
)

# For preview (fast)
preview = gen.generate_avatar(
    "young blonde woman",
    method='attributes'  # 0.2 seconds
)

# For highest quality (slow)
final = gen.generate_avatar(
    "A 30-year-old Caucasian man with short brown hair, beard, blue eyes",
    method='clip'  # 60 seconds
)
```

---

## COMPARISON OF METHODS

| Method | Speed | Quality | Flexibility | Training Required |
|--------|-------|---------|-------------|-------------------|
| **CLIP-Guided** | Slow (60s) | Excellent | Very High | No |
| **Text-to-Latent** | Fast (0.2s) | Good | High | Yes (large dataset) |
| **Attribute-Based** | Instant (0.1s) | Good | Medium | No (pre-computed) |
| **Hybrid** | Medium (20s) | Excellent | Very High | No |

---

## SUMMARY

✅ **GANs don't natively accept text** - They're trained on random noise
✅ **Solution 1 (CLIP)**: Optimize latent code to match CLIP text-image similarity
✅ **Solution 2 (Mapper)**: Train neural network to map text → latent directly
✅ **Solution 3 (Attributes)**: Use pre-discovered semantic directions
✅ **Best Practice**: Hybrid approach (attributes + CLIP refinement)

For txt2create.com, we use **Method 1 (CLIP-guided)** as primary with **Method 3 (attributes)** for quick previews!
