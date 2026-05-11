# DocAI — System Architecture & Design

> Interview-ready deep dive: problem, solution, technology stack, and end-to-end data flow.

---

## Page 1 — Problem Statement & High-Level Architecture

### The Problem

Enterprise document pipelines face three hard constraints simultaneously:

- **Low labeled data** — Getting 200+ labeled samples per class across 12 document types is already expensive; thousands is impossible.
- **Scale demand** — Business units submit 10K+ documents per hour at peak; synchronous processing creates unacceptable latency.
- **Reliability requirements** — Any model degradation must be detected and rolled back automatically, not caught in production.

### Solution Overview

A three-layer intelligent document processing system:

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                          │
│   S3 / File Upload ──► FastAPI Async Queue ──► Spark Preprocessing│
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                       INFERENCE LAYER                           │
│   LayoutLMv3 (LoRA fine-tuned) ──► CUDA PyTorch ──► FastAPI     │
│   Classification (12 types)  ──► Field Extraction ──► Results   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    ORCHESTRATION & QUALITY LAYER                │
│   Airflow DAGs ──► MLflow Registry ──► Data Quality Checks      │
│   Accuracy Monitor ──► Auto Rollback on Regression              │
└─────────────────────────────────────────────────────────────────┘
```

### Why LayoutLMv3?

Standard NLP models read text only. Documents have **spatial structure** — a number in the top-right corner of an invoice means something different than the same number at the bottom. LayoutLMv3 jointly encodes:
- **Text tokens** (what it says)
- **Bounding box coordinates** (where it is on the page)
- **Visual patch embeddings** (what the page looks like)

This tri-modal approach is why it outperforms pure OCR + NLP pipelines on structured documents.

---

## Page 2 — Technology Stack & Component Design

### Training Pipeline (AWS SageMaker)

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Base model | LayoutLMv3 (microsoft/layoutlmv3-base) | Pre-trained document understanding |
| Fine-tuning method | LoRA (Low-Rank Adaptation) | Parameter-efficient training on low data |
| Training infra | AWS SageMaker Training Jobs | Managed GPU training, spot instances |
| Experiment tracking | MLflow | Hyperparameter logs, metric curves, artifact storage |
| Model registry | MLflow Model Registry | Versioned model store with staging/production stages |

**Why LoRA over full fine-tuning?**

LoRA freezes the original model weights and injects small trainable rank-decomposition matrices (rank r << d) into the attention layers. With 200 labeled samples per class:
- Full fine-tuning → overfits, catastrophic forgetting
- LoRA → only ~0.1% of parameters are trained → regularizes naturally
- Result: **96% accuracy across 12 document types**

### Inference Pipeline

```
Document Input (PDF / Image)
        │
        ▼
   OCR Engine (Tesseract / AWS Textract)
   → extracts: text, bounding boxes, confidence scores
        │
        ▼
   Tokenizer + Layout Encoding
   → maps words to token IDs + normalized (x0,y0,x1,y1) bbox coords
        │
        ▼
   LayoutLMv3 + LoRA Adapter (GPU / CUDA)
   → forward pass: classification head + token-level extraction head
        │
        ▼
   Post-processing
   → decode predicted class, extract field values, confidence thresholds
        │
        ▼
   FastAPI Response (JSON)
```

**Throughput engineering (4x improvement):**

| Approach | Throughput |
|----------|-----------|
| Synchronous baseline (CPU, sequential) | ~2,500 docs/hour |
| CUDA-accelerated PyTorch + async FastAPI task queues | **10,000+ docs/hour** |

Key levers:
- **CUDA batching** — group incoming documents into batches of 32–64, process in parallel on GPU
- **Async task queues** — FastAPI endpoints enqueue jobs immediately (non-blocking); workers pull from queue
- **Mixed precision (FP16)** — halves memory bandwidth, doubles effective throughput on modern GPUs

### Preprocessing at Scale (Spark)

Apache Spark handles the raw document preprocessing before inference:
- PDF → image rendering at consistent DPI
- Image normalization (resize, denoise)
- OCR parallelization across Spark executors
- Output written to S3 as structured Parquet for audit trail

---

## Page 3 — End-to-End Data Flow & Orchestration

### Airflow DAG — Complete Workflow

```
[Trigger: S3 event / schedule]
         │
         ▼
  ┌─────────────────┐
  │  Spark Preprocess│  ← PDF rendering, OCR, normalization
  │  (EMR cluster)  │    parallelized across N executors
  └────────┬────────┘
           │  Parquet files → S3
           ▼
  ┌─────────────────┐
  │ Model Inference │  ← LayoutLMv3 + LoRA on SageMaker endpoint
  │  (batch scored) │    or self-hosted GPU instance via FastAPI
  └────────┬────────┘
           │  Predictions + confidence scores
           ▼
  ┌─────────────────┐
  │  Data Quality   │  ← checks: confidence threshold, field completeness,
  │     Checks      │    distribution drift vs. training baseline
  └────────┬────────┘
           │
     ┌─────┴──────┐
     │            │
  PASS          FAIL
     │            │
     ▼            ▼
  Write to    Trigger MLflow
  Data Store  rollback to last
  (S3/DB)     stable version
                   │
                   ▼
              Alert + re-run
              with previous model
```

### MLflow Model Registry — Rollback Strategy

```
Model Versions in Registry:
  v1 (archived) ──► v2 (production) ──► v3 (staging)

On accuracy regression detection:
  1. Airflow quality check task fails
  2. DAG triggers rollback hook
  3. MLflow API call: transition v3 → archived, v1 → production
  4. SageMaker endpoint updated to serve v1 weights
  5. Alert sent (PagerDuty / Slack)
  6. One-click — no manual intervention needed
```

### Interview Talking Points — Key Decisions

**"Why not just use GPT-4 Vision for this?"**
→ We needed deterministic, auditable extraction with per-field confidence scores and sub-100ms latency at 10K docs/hour. GPT-4 Vision is non-deterministic, expensive at scale, and has no structured output guarantee for financial documents.

**"How did you handle 12 document types with only 200 samples each?"**
→ LoRA fine-tuning with LayoutLMv3's strong pre-training on IIT-CDIP (11M documents). The model already understands document layout; LoRA just teaches it our specific schema. 200 samples is enough to steer, not teach from scratch.

**"What does the data quality check actually verify?"**
→ Three checks: (1) model confidence score above threshold (e.g., 0.85), (2) mandatory fields present and non-null, (3) statistical distribution of predicted classes matches historical baseline — catches silent drift before it impacts downstream systems.

**"What's your latency SLA?"**
→ Async queue design decouples ingestion latency from processing latency. Documents are acknowledged immediately; results are available within SLA window. For real-time use cases, a synchronous path with pre-warmed GPU instances and connection pooling is available.

---

*Architecture covers: LayoutLMv3 + LoRA fine-tuning · AWS SageMaker · FastAPI async queues · CUDA PyTorch · Apache Spark · Airflow DAGs · MLflow registry · automated rollback*
