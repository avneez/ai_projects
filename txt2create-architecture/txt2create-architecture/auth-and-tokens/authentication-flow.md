# Authentication & Token-Based Usage System

## Overview

TXT2CREATE uses a dual-token system:
1. **JWT Tokens**: For user authentication and authorization
2. **Usage Tokens**: For tracking and limiting API usage (credits/quotas)

---

## PART 1: USER AUTHENTICATION FLOW

### Complete Authentication Journey

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER REGISTRATION FLOW                       │
└─────────────────────────────────────────────────────────────────┘

Step 1: User Registration
┌──────────────┐
│   User       │ POST /api/v1/auth/register
│   Browser    │ {
└──────┬───────┘   "email": "user@example.com",
       │           "password": "SecurePass123!",
       │           "name": "John Doe"
       │         }
       ▼
┌─────────────────────────────────────┐
│ FastAPI Auth Service                │
│                                     │
│ 1. Validate email format            │
│ 2. Check if email exists            │
│ 3. Hash password (bcrypt)           │
│ 4. Create user record               │
│ 5. Assign default tier (FREE)       │
│ 6. Initialize token balance         │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ PostgreSQL Database                 │
│                                     │
│ INSERT INTO users (                 │
│   id: "550e8400-...",               │
│   email: "user@example.com",        │
│   password_hash: "$2b$12$...",      │
│   name: "John Doe",                 │
│   tier: "FREE",                     │
│   created_at: "2025-12-21...",      │
│   email_verified: false             │
│ );                                  │
│                                     │
│ INSERT INTO user_tokens (           │
│   user_id: "550e8400-...",          │
│   balance: 100,  -- Free tier gets  │
│   monthly_quota: 100,                │
│   reset_date: "2026-01-21"          │
│ );                                  │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Email Verification Service          │
│                                     │
│ 1. Generate verification token      │
│ 2. Store in Redis (24h TTL)         │
│ 3. Send verification email          │
└──────┬──────────────────────────────┘
       │
       ▼
┌──────────────┐
│   User       │ Response: {
│   Browser    │   "message": "Registration successful",
└──────────────┘   "user_id": "550e8400-...",
                   "email_verification_sent": true
                 }


Step 2: Email Verification
┌──────────────┐
│   User       │ Clicks link in email
│   Email      │ GET /api/v1/auth/verify-email?token=abc123...
└──────┬───────┘
       ▼
┌─────────────────────────────────────┐
│ FastAPI Auth Service                │
│ 1. Validate token from Redis        │
│ 2. Mark user as verified             │
│ 3. Delete token from Redis           │
└──────┬──────────────────────────────┘
       ▼
┌──────────────┐
│   Database   │ UPDATE users
└──────────────┘ SET email_verified = true
                 WHERE id = "550e8400-..."


Step 3: User Login
┌──────────────┐
│   User       │ POST /api/v1/auth/login
│   Browser    │ {
└──────┬───────┘   "email": "user@example.com",
       │           "password": "SecurePass123!"
       │         }
       ▼
┌──────────────────────────────────────────────────────────┐
│ FastAPI Auth Service                                     │
│                                                          │
│ 1. Query user by email                                   │
│ 2. Verify password hash                                  │
│    bcrypt.checkpw(password, user.password_hash)          │
│ 3. Check if email verified                               │
│ 4. Generate JWT tokens                                   │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ JWT Token Generation                                     │
│                                                          │
│ Access Token (15 minutes expiry):                        │
│ {                                                        │
│   "sub": "550e8400-...",  # User ID                      │
│   "email": "user@example.com",                           │
│   "name": "John Doe",                                    │
│   "tier": "FREE",                                        │
│   "permissions": ["generate_image", "generate_audio"],   │
│   "exp": 1703174400,  # Expires in 15 min                │
│   "iat": 1703173500,  # Issued at                        │
│   "type": "access"                                       │
│ }                                                        │
│ Signed with: HS256 + SECRET_KEY                          │
│                                                          │
│ Refresh Token (7 days expiry):                           │
│ {                                                        │
│   "sub": "550e8400-...",                                 │
│   "exp": 1703779200,  # Expires in 7 days                │
│   "iat": 1703173500,                                     │
│   "type": "refresh",                                     │
│   "jti": "unique-token-id"  # For revocation             │
│ }                                                        │
│ Signed with: HS256 + REFRESH_SECRET_KEY                  │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Redis - Session Storage                                  │
│                                                          │
│ Key: "session:{user_id}:{jti}"                           │
│ Value: {                                                 │
│   "user_id": "550e8400-...",                             │
│   "refresh_token_jti": "unique-token-id",                │
│   "ip_address": "192.168.1.1",                           │
│   "user_agent": "Mozilla/5.0...",                        │
│   "created_at": "2025-12-21T10:00:00Z",                  │
│   "last_active": "2025-12-21T10:00:00Z"                  │
│ }                                                        │
│ TTL: 7 days                                              │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────┐
│   User       │ Response: {
│   Browser    │   "access_token": "eyJhbGciOiJIUzI1NiIs...",
└──────────────┘   "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
                   "token_type": "Bearer",
                   "expires_in": 900  # seconds
                 }

                 Browser stores tokens:
                 - access_token → localStorage
                 - refresh_token → httpOnly cookie (more secure)
```

---

## Authentication Code Implementation

### 1. User Registration

```python
# auth/routes.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from passlib.hash import bcrypt
from pydantic import BaseModel, EmailStr, validator
import uuid
from datetime import datetime

router = APIRouter(prefix="/api/v1/auth")

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    name: str

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v

@router.post("/register")
async def register(
    request: RegisterRequest,
    background_tasks: BackgroundTasks
):
    # Check if user exists
    existing_user = await db.query(User).filter_by(email=request.email).first()
    if existing_user:
        raise HTTPException(400, "Email already registered")

    # Hash password
    password_hash = bcrypt.hash(request.password)

    # Create user
    user_id = str(uuid.uuid4())
    user = User(
        id=user_id,
        email=request.email,
        password_hash=password_hash,
        name=request.name,
        tier=UserTier.FREE,
        email_verified=False,
        created_at=datetime.utcnow()
    )
    await db.add(user)

    # Initialize token balance
    user_tokens = UserTokens(
        user_id=user_id,
        balance=100,  # Free tier gets 100 tokens
        monthly_quota=100,
        quota_used=0,
        reset_date=datetime.utcnow() + timedelta(days=30)
    )
    await db.add(user_tokens)
    await db.commit()

    # Send verification email (async)
    verification_token = str(uuid.uuid4())
    await redis.setex(
        f"verify_email:{verification_token}",
        86400,  # 24 hours
        user_id
    )
    background_tasks.add_task(
        send_verification_email,
        request.email,
        verification_token
    )

    return {
        "message": "Registration successful",
        "user_id": user_id,
        "email_verification_sent": True
    }
```

### 2. Login & JWT Generation

```python
# auth/jwt.py
import jwt
from datetime import datetime, timedelta
from typing import Dict, Optional

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
REFRESH_SECRET_KEY = os.getenv("JWT_REFRESH_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

def create_access_token(user: User) -> str:
    """Create JWT access token"""
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "name": user.name,
        "tier": user.tier.value,
        "permissions": get_user_permissions(user.tier),
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat": datetime.utcnow(),
        "type": "access"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(user: User) -> tuple[str, str]:
    """Create JWT refresh token with unique ID"""
    jti = str(uuid.uuid4())  # Unique token ID for revocation
    payload = {
        "sub": str(user.id),
        "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        "iat": datetime.utcnow(),
        "type": "refresh",
        "jti": jti
    }
    token = jwt.encode(payload, REFRESH_SECRET_KEY, algorithm=ALGORITHM)
    return token, jti

def get_user_permissions(tier: UserTier) -> list[str]:
    """Get permissions based on user tier"""
    permissions_map = {
        UserTier.FREE: [
            "generate_image",
            "generate_audio"
        ],
        UserTier.PRO: [
            "generate_image",
            "generate_video",
            "generate_audio",
            "caption_video",
            "train_lora"
        ],
        UserTier.ENTERPRISE: [
            "generate_image",
            "generate_video",
            "generate_audio",
            "caption_video",
            "generate_avatar",
            "train_lora",
            "api_access",
            "batch_processing"
        ]
    }
    return permissions_map.get(tier, [])

# auth/routes.py
@router.post("/login")
async def login(email: str, password: str, request: Request):
    # Find user
    user = await db.query(User).filter_by(email=email).first()
    if not user:
        raise HTTPException(401, "Invalid credentials")

    # Verify password
    if not bcrypt.verify(password, user.password_hash):
        raise HTTPException(401, "Invalid credentials")

    # Check email verified
    if not user.email_verified:
        raise HTTPException(403, "Email not verified")

    # Generate tokens
    access_token = create_access_token(user)
    refresh_token, jti = create_refresh_token(user)

    # Store session in Redis
    session_data = {
        "user_id": str(user.id),
        "refresh_token_jti": jti,
        "ip_address": request.client.host,
        "user_agent": request.headers.get("user-agent"),
        "created_at": datetime.utcnow().isoformat(),
        "last_active": datetime.utcnow().isoformat()
    }
    await redis.setex(
        f"session:{user.id}:{jti}",
        REFRESH_TOKEN_EXPIRE_DAYS * 86400,
        json.dumps(session_data)
    )

    # Update last login
    user.last_login = datetime.utcnow()
    await db.commit()

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }
```

### 3. Token Verification & Protected Endpoints

```python
# auth/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Verify JWT and return current user"""
    token = credentials.credentials

    try:
        # Decode JWT
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        # Check token type
        if payload.get("type") != "access":
            raise HTTPException(401, "Invalid token type")

        # Get user ID
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(401, "Invalid token")

        # Load user from database (with caching)
        user = await get_user_cached(user_id)
        if not user:
            raise HTTPException(401, "User not found")

        return user

    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token expired")
    except jwt.JWTError:
        raise HTTPException(401, "Invalid token")

async def get_user_cached(user_id: str) -> Optional[User]:
    """Get user from cache or database"""
    # Try cache first
    cached = await redis.get(f"user:{user_id}")
    if cached:
        return User(**json.loads(cached))

    # Load from database
    user = await db.query(User).filter_by(id=user_id).first()
    if user:
        # Cache for 5 minutes
        await redis.setex(
            f"user:{user_id}",
            300,
            user.to_json()
        )
    return user

def require_permission(permission: str):
    """Decorator to require specific permission"""
    def decorator(func):
        async def wrapper(
            user: User = Depends(get_current_user),
            *args,
            **kwargs
        ):
            permissions = get_user_permissions(user.tier)
            if permission not in permissions:
                raise HTTPException(
                    403,
                    f"Permission denied: {permission} required"
                )
            return await func(user=user, *args, **kwargs)
        return wrapper
    return decorator

# Usage in endpoints
@router.post("/generate/video")
@require_permission("generate_video")
async def generate_video(
    prompt: str,
    user: User = Depends(get_current_user)
):
    # User is authenticated and has permission
    return await create_video_job(user.id, prompt)
```

### 4. Token Refresh Flow

```python
@router.post("/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token"""
    try:
        # Decode refresh token
        payload = jwt.decode(
            refresh_token,
            REFRESH_SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        # Check token type
        if payload.get("type") != "refresh":
            raise HTTPException(401, "Invalid token type")

        user_id = payload.get("sub")
        jti = payload.get("jti")

        # Check if session exists (not revoked)
        session = await redis.get(f"session:{user_id}:{jti}")
        if not session:
            raise HTTPException(401, "Session expired or revoked")

        # Load user
        user = await db.query(User).filter_by(id=user_id).first()
        if not user:
            raise HTTPException(401, "User not found")

        # Generate new access token
        new_access_token = create_access_token(user)

        # Update session last active
        session_data = json.loads(session)
        session_data["last_active"] = datetime.utcnow().isoformat()
        await redis.setex(
            f"session:{user_id}:{jti}",
            REFRESH_TOKEN_EXPIRE_DAYS * 86400,
            json.dumps(session_data)
        )

        return {
            "access_token": new_access_token,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }

    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Refresh token expired")
    except jwt.JWTError:
        raise HTTPException(401, "Invalid refresh token")

@router.post("/logout")
async def logout(
    user: User = Depends(get_current_user),
    refresh_token: str = None
):
    """Logout user and revoke session"""
    if refresh_token:
        try:
            payload = jwt.decode(
                refresh_token,
                REFRESH_SECRET_KEY,
                algorithms=[ALGORITHM]
            )
            jti = payload.get("jti")

            # Delete session
            await redis.delete(f"session:{user.id}:{jti}")
        except:
            pass  # Invalid token, but logout anyway

    # Clear user cache
    await redis.delete(f"user:{user.id}")

    return {"message": "Logged out successfully"}
```

---

## PART 2: USAGE TOKEN SYSTEM

### Token Calculation & Usage Tracking

```
┌─────────────────────────────────────────────────────────────────┐
│               USAGE TOKEN SYSTEM FLOW                            │
└─────────────────────────────────────────────────────────────────┘

Step 1: User initiates generation
┌──────────────┐
│   User       │ POST /api/v1/generate/image
│   Browser    │ Headers: {
└──────┬───────┘   Authorization: "Bearer eyJhbGci..."
       │         }
       │         Body: {
       │           "prompt": "A cat in a garden",
       │           "steps": 30,
       │           "resolution": "1024x1024"
       │         }
       ▼
┌──────────────────────────────────────────────────────────┐
│ FastAPI - Token Calculation Service                      │
│                                                          │
│ 1. Authenticate user (JWT)                               │
│ 2. Calculate required tokens                             │
│ 3. Check user token balance                              │
│ 4. Reserve tokens (atomic operation)                     │
│ 5. Create job if tokens available                        │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Token Calculation Logic                                  │
│                                                          │
│ Base Cost Factors:                                       │
│ - Pipeline type (image/video/audio/avatar)               │
│ - Complexity (resolution, duration, steps)               │
│ - Model used (SD XL vs SD 2.1)                           │
│ - Priority (normal/high)                                 │
│                                                          │
│ Formula for Text-to-Image:                               │
│ tokens = base_cost × resolution_multiplier ×             │
│          steps_multiplier × model_multiplier             │
│                                                          │
│ Example:                                                 │
│ - Base: 10 tokens                                        │
│ - Resolution 1024x1024: 2.0x                             │
│ - Steps 30: 1.0x (default)                               │
│ - SD XL: 1.5x                                            │
│ Total: 10 × 2.0 × 1.0 × 1.5 = 30 tokens                  │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ PostgreSQL - Check & Reserve Tokens (Atomic)             │
│                                                          │
│ BEGIN TRANSACTION;                                       │
│                                                          │
│ -- Lock user's token row                                 │
│ SELECT balance, monthly_quota, quota_used                │
│ FROM user_tokens                                         │
│ WHERE user_id = '550e8400-...'                           │
│ FOR UPDATE;  -- Row-level lock                           │
│                                                          │
│ -- Check if enough tokens                                │
│ IF balance >= 30 THEN                                    │
│   -- Deduct tokens                                       │
│   UPDATE user_tokens                                     │
│   SET balance = balance - 30,                            │
│       quota_used = quota_used + 30,                      │
│       last_used = NOW()                                  │
│   WHERE user_id = '550e8400-...';                        │
│                                                          │
│   -- Log usage                                           │
│   INSERT INTO token_usage_log (                          │
│     user_id, job_id, pipeline, tokens_used,              │
│     timestamp                                            │
│   ) VALUES (...);                                        │
│                                                          │
│   COMMIT;                                                │
│   RETURN success;                                        │
│ ELSE                                                     │
│   ROLLBACK;                                              │
│   RETURN insufficient_tokens;                            │
│ END IF;                                                  │
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ If tokens sufficient:                                    │
│ - Create job in database                                 │
│ - Queue task in Celery                                   │
│ - Return job_id to user                                  │
│                                                          │
│ If tokens insufficient:                                  │
│ - Return error with current balance                      │
│ - Suggest upgrade or purchase tokens                     │
└──────────────────────────────────────────────────────────┘
```

---

## Token Calculation Implementation

### 1. Token Cost Configuration

```python
# config/token_costs.py
from enum import Enum
from dataclasses import dataclass

class PipelineType(Enum):
    TEXT_TO_IMAGE = "text_to_image"
    TEXT_TO_VIDEO = "text_to_video"
    TEXT_TO_AUDIO = "text_to_audio"
    VIDEO_CAPTION = "video_caption"
    AVATAR_GENERATION = "avatar_generation"

@dataclass
class TokenCostConfig:
    """Token cost configuration"""

    # Base costs per pipeline (in tokens)
    BASE_COSTS = {
        PipelineType.TEXT_TO_IMAGE: 10,
        PipelineType.TEXT_TO_VIDEO: 100,
        PipelineType.TEXT_TO_AUDIO: 15,
        PipelineType.VIDEO_CAPTION: 20,
        PipelineType.AVATAR_GENERATION: 150
    }

    # Resolution multipliers (for image/video)
    RESOLUTION_MULTIPLIERS = {
        "512x512": 1.0,
        "768x768": 1.5,
        "1024x1024": 2.0,
        "1536x1536": 3.0,
        "2048x2048": 4.0,
        # Video resolutions
        "720p": 2.0,
        "1080p": 3.5,
        "4k": 8.0
    }

    # Steps multipliers (for diffusion models)
    STEPS_MULTIPLIERS = {
        "range": {
            (1, 20): 0.5,
            (21, 30): 1.0,
            (31, 50): 1.5,
            (51, 100): 2.0,
            (101, 150): 3.0
        }
    }

    # Duration multipliers (for video/audio, per second)
    DURATION_MULTIPLIER = 1.0  # Base
    DURATION_PER_SECOND = 2.0  # Additional tokens per second

    # Model multipliers
    MODEL_MULTIPLIERS = {
        "sd-2.1": 1.0,
        "sd-xl-1.0": 1.5,
        "sd-xl-refiner": 2.0,
        "llama-3-8b": 1.0,
        "llama-3-70b": 3.0,
        "musicgen-small": 1.0,
        "musicgen-large": 2.0
    }

    # Priority multipliers
    PRIORITY_MULTIPLIERS = {
        "low": 0.8,
        "normal": 1.0,
        "high": 1.5,
        "urgent": 2.0
    }

class TokenCalculator:
    """Calculate token cost for different operations"""

    @staticmethod
    def calculate_image_cost(
        resolution: str = "1024x1024",
        steps: int = 30,
        model: str = "sd-xl-1.0",
        priority: str = "normal"
    ) -> int:
        """Calculate tokens for image generation"""
        base = TokenCostConfig.BASE_COSTS[PipelineType.TEXT_TO_IMAGE]

        # Resolution multiplier
        res_mult = TokenCostConfig.RESOLUTION_MULTIPLIERS.get(resolution, 2.0)

        # Steps multiplier
        steps_mult = 1.0
        for (min_steps, max_steps), mult in TokenCostConfig.STEPS_MULTIPLIERS["range"].items():
            if min_steps <= steps <= max_steps:
                steps_mult = mult
                break

        # Model multiplier
        model_mult = TokenCostConfig.MODEL_MULTIPLIERS.get(model, 1.0)

        # Priority multiplier
        priority_mult = TokenCostConfig.PRIORITY_MULTIPLIERS.get(priority, 1.0)

        # Calculate total
        total = int(base * res_mult * steps_mult * model_mult * priority_mult)

        return max(total, 1)  # Minimum 1 token

    @staticmethod
    def calculate_video_cost(
        resolution: str = "720p",
        duration: int = 5,  # seconds
        fps: int = 30,
        model: str = "sd-xl-1.0",
        priority: str = "normal"
    ) -> int:
        """Calculate tokens for video generation"""
        base = TokenCostConfig.BASE_COSTS[PipelineType.TEXT_TO_VIDEO]

        # Resolution multiplier
        res_mult = TokenCostConfig.RESOLUTION_MULTIPLIERS.get(resolution, 2.0)

        # Duration cost
        duration_cost = TokenCostConfig.DURATION_MULTIPLIER + \
                       (duration * TokenCostConfig.DURATION_PER_SECOND)

        # Model multiplier
        model_mult = TokenCostConfig.MODEL_MULTIPLIERS.get(model, 1.0)

        # Priority multiplier
        priority_mult = TokenCostConfig.PRIORITY_MULTIPLIERS.get(priority, 1.0)

        # Calculate total
        total = int(base * res_mult * duration_cost * model_mult * priority_mult)

        return max(total, 10)  # Minimum 10 tokens

    @staticmethod
    def calculate_audio_cost(
        duration: int = 30,  # seconds
        quality: str = "standard",  # standard or high
        model: str = "musicgen-large",
        priority: str = "normal"
    ) -> int:
        """Calculate tokens for audio generation"""
        base = TokenCostConfig.BASE_COSTS[PipelineType.TEXT_TO_AUDIO]

        # Duration cost
        duration_cost = TokenCostConfig.DURATION_MULTIPLIER + \
                       (duration * TokenCostConfig.DURATION_PER_SECOND)

        # Quality multiplier
        quality_mult = 1.5 if quality == "high" else 1.0

        # Model multiplier
        model_mult = TokenCostConfig.MODEL_MULTIPLIERS.get(model, 1.0)

        # Priority multiplier
        priority_mult = TokenCostConfig.PRIORITY_MULTIPLIERS.get(priority, 1.0)

        # Calculate total
        total = int(base * duration_cost * quality_mult * model_mult * priority_mult)

        return max(total, 5)  # Minimum 5 tokens

    @staticmethod
    def calculate_avatar_cost(
        dimension: str = "2d",  # 2d or 3d
        rigged: bool = False,
        animated: bool = False,
        priority: str = "normal"
    ) -> int:
        """Calculate tokens for avatar generation"""
        base = TokenCostConfig.BASE_COSTS[PipelineType.AVATAR_GENERATION]

        # Dimension multiplier
        dim_mult = 2.0 if dimension == "3d" else 1.0

        # Rigging adds cost
        rig_mult = 1.5 if rigged else 1.0

        # Animation adds cost
        anim_mult = 2.0 if animated else 1.0

        # Priority multiplier
        priority_mult = TokenCostConfig.PRIORITY_MULTIPLIERS.get(priority, 1.0)

        # Calculate total
        total = int(base * dim_mult * rig_mult * anim_mult * priority_mult)

        return max(total, 50)  # Minimum 50 tokens
```

### 2. Token Management Service

```python
# services/token_service.py
from sqlalchemy import select
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import uuid

class TokenService:
    """Manage user tokens and usage"""

    @staticmethod
    async def get_user_tokens(user_id: str) -> dict:
        """Get user's current token balance"""
        tokens = await db.query(UserTokens).filter_by(user_id=user_id).first()

        if not tokens:
            # Initialize tokens for new user
            tokens = await TokenService.initialize_tokens(user_id)

        return {
            "balance": tokens.balance,
            "monthly_quota": tokens.monthly_quota,
            "quota_used": tokens.quota_used,
            "quota_remaining": tokens.monthly_quota - tokens.quota_used,
            "reset_date": tokens.reset_date.isoformat()
        }

    @staticmethod
    async def initialize_tokens(user_id: str, tier: UserTier = UserTier.FREE) -> UserTokens:
        """Initialize token balance for new user"""
        quotas = {
            UserTier.FREE: 100,
            UserTier.PRO: 1000,
            UserTier.ENTERPRISE: 10000
        }

        tokens = UserTokens(
            user_id=user_id,
            balance=quotas[tier],
            monthly_quota=quotas[tier],
            quota_used=0,
            reset_date=datetime.utcnow() + timedelta(days=30),
            created_at=datetime.utcnow()
        )
        await db.add(tokens)
        await db.commit()
        return tokens

    @staticmethod
    async def reserve_tokens(
        user_id: str,
        job_id: str,
        pipeline: PipelineType,
        tokens_required: int
    ) -> tuple[bool, str]:
        """
        Reserve tokens for a job (atomic operation)
        Returns: (success: bool, message: str)
        """
        async with db.begin():  # Start transaction
            # Lock user's token row
            result = await db.execute(
                select(UserTokens)
                .where(UserTokens.user_id == user_id)
                .with_for_update()  # Row-level lock
            )
            tokens = result.scalar_one_or_none()

            if not tokens:
                return False, "Token account not found"

            # Check if enough tokens
            if tokens.balance < tokens_required:
                return False, f"Insufficient tokens. Required: {tokens_required}, Available: {tokens.balance}"

            # Deduct tokens
            tokens.balance -= tokens_required
            tokens.quota_used += tokens_required
            tokens.last_used = datetime.utcnow()

            # Log usage
            usage_log = TokenUsageLog(
                id=str(uuid.uuid4()),
                user_id=user_id,
                job_id=job_id,
                pipeline=pipeline.value,
                tokens_used=tokens_required,
                timestamp=datetime.utcnow()
            )
            await db.add(usage_log)

            # Commit transaction
            await db.commit()

            return True, f"Reserved {tokens_required} tokens. New balance: {tokens.balance}"

    @staticmethod
    async def refund_tokens(
        user_id: str,
        job_id: str,
        tokens_to_refund: int,
        reason: str = "Job failed"
    ) -> bool:
        """Refund tokens if job fails"""
        async with db.begin():
            tokens = await db.query(UserTokens).filter_by(user_id=user_id).first()
            if not tokens:
                return False

            # Refund tokens
            tokens.balance += tokens_to_refund
            tokens.quota_used = max(0, tokens.quota_used - tokens_to_refund)

            # Log refund
            refund_log = TokenRefundLog(
                id=str(uuid.uuid4()),
                user_id=user_id,
                job_id=job_id,
                tokens_refunded=tokens_to_refund,
                reason=reason,
                timestamp=datetime.utcnow()
            )
            await db.add(refund_log)
            await db.commit()

            return True

    @staticmethod
    async def add_tokens(
        user_id: str,
        tokens_to_add: int,
        source: str = "purchase"
    ) -> int:
        """Add tokens to user's balance (purchase or gift)"""
        async with db.begin():
            tokens = await db.query(UserTokens).filter_by(user_id=user_id).first()
            if not tokens:
                tokens = await TokenService.initialize_tokens(user_id)

            tokens.balance += tokens_to_add

            # Log addition
            addition_log = TokenAdditionLog(
                id=str(uuid.uuid4()),
                user_id=user_id,
                tokens_added=tokens_to_add,
                source=source,
                timestamp=datetime.utcnow()
            )
            await db.add(addition_log)
            await db.commit()

            return tokens.balance

    @staticmethod
    async def reset_monthly_quota(user_id: str):
        """Reset monthly quota (called by cron job)"""
        async with db.begin():
            tokens = await db.query(UserTokens).filter_by(user_id=user_id).first()
            if not tokens:
                return

            # Get user's tier to determine quota
            user = await db.query(User).filter_by(id=user_id).first()
            quotas = {
                UserTier.FREE: 100,
                UserTier.PRO: 1000,
                UserTier.ENTERPRISE: 10000
            }

            # Reset quota
            tokens.quota_used = 0
            tokens.monthly_quota = quotas.get(user.tier, 100)
            tokens.balance = max(tokens.balance, tokens.monthly_quota)  # Ensure at least quota
            tokens.reset_date = datetime.utcnow() + timedelta(days=30)

            await db.commit()
```

### 3. Integration with Generation Endpoints

```python
# routes/generation.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/generate")

class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: str = Field(default="", max_length=500)
    resolution: str = Field(default="1024x1024")
    steps: int = Field(default=30, ge=10, le=150)
    model: str = Field(default="sd-xl-1.0")
    priority: str = Field(default="normal")

@router.post("/image")
async def generate_image(
    request: ImageGenerationRequest,
    user: User = Depends(get_current_user)
):
    """Generate image with token-based usage tracking"""

    # 1. Calculate token cost
    tokens_required = TokenCalculator.calculate_image_cost(
        resolution=request.resolution,
        steps=request.steps,
        model=request.model,
        priority=request.priority
    )

    # 2. Create job ID
    job_id = str(uuid.uuid4())

    # 3. Reserve tokens (atomic operation)
    success, message = await TokenService.reserve_tokens(
        user_id=user.id,
        job_id=job_id,
        pipeline=PipelineType.TEXT_TO_IMAGE,
        tokens_required=tokens_required
    )

    if not success:
        # Not enough tokens
        current_balance = await TokenService.get_user_tokens(user.id)
        raise HTTPException(
            402,  # Payment Required
            detail={
                "error": "Insufficient tokens",
                "tokens_required": tokens_required,
                "tokens_available": current_balance["balance"],
                "message": message
            }
        )

    # 4. Create job in database
    job = Job(
        id=job_id,
        user_id=user.id,
        pipeline=PipelineType.TEXT_TO_IMAGE.value,
        prompt=request.prompt,
        parameters=request.dict(),
        tokens_used=tokens_required,
        status="pending",
        created_at=datetime.utcnow()
    )
    await db.add(job)
    await db.commit()

    # 5. Queue Celery task
    task = generate_image_task.apply_async(
        args=[job_id, user.id, request.dict()],
        priority=get_priority_value(request.priority)
    )

    # 6. Return job info
    return {
        "job_id": job_id,
        "status": "queued",
        "tokens_used": tokens_required,
        "estimated_time": "30-60 seconds",
        "task_id": task.id
    }

# Worker task with token refund on failure
@celery.task(bind=True, max_retries=3)
def generate_image_task(self, job_id: str, user_id: str, params: dict):
    """Generate image (with automatic token refund on failure)"""
    try:
        # Update job status
        update_job_status(job_id, "processing")

        # Generate image
        result = generate_image_with_sd(params)

        # Upload to S3
        image_url = upload_to_s3(result, job_id)

        # Update job with result
        update_job_status(job_id, "completed", result_url=image_url)

        return {
            "job_id": job_id,
            "status": "completed",
            "image_url": image_url
        }

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")

        # Refund tokens on failure
        job = get_job(job_id)
        if job:
            await TokenService.refund_tokens(
                user_id=user_id,
                job_id=job_id,
                tokens_to_refund=job.tokens_used,
                reason=f"Generation failed: {str(e)}"
            )

        # Update job status
        update_job_status(job_id, "failed", error=str(e))

        # Retry if not max retries
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)

        raise
```

---

Continued in next file...
