# 🛡️ Fraud Detection & Anomaly Detection — A Complete Learning Guide

> A deep-dive explanation of every concept, bullet point, and project from the resume.
> Written for someone new to the field — every term is explained from first principles.

---

## Table of Contents

1. [Why Fraud Detection is Hard](#1-why-fraud-detection-is-hard)
2. [Core Concepts You Must Know First](#2-core-concepts-you-must-know-first)
3. [Real-Time Anomaly Detection & Streaming](#3-real-time-anomaly-detection--streaming)
4. [Unsupervised Anomaly Detection Methods](#4-unsupervised-anomaly-detection-methods)
5. [Model Calibration](#5-model-calibration)
6. [Adversarial Robustness Testing](#6-adversarial-robustness-testing)
7. [Feature Store (Feast + BigQuery)](#7-feature-store-feast--bigquery)
8. [LLM Fine-Tuning in Fraud Context](#8-llm-fine-tuning-in-fraud-context)
9. [Real-Time Model Monitoring & Drift Detection](#9-real-time-model-monitoring--drift-detection)
10. [MLOps Stack](#10-mlops-stack)
11. [Ensemble Methods for Fraud](#11-ensemble-methods-for-fraud)
12. [AML — Anti-Money Laundering Pipeline](#12-aml--anti-money-laundering-pipeline)
13. [Graph-Based Fraud Detection](#13-graph-based-fraud-detection)
14. [Multi-Agent AI Systems](#14-multi-agent-ai-systems)
15. [LLM Deployment Optimization](#15-llm-deployment-optimization)
16. [Projects Deep Dive](#16-projects-deep-dive)
17. [Key Metrics & Numbers Explained](#17-key-metrics--numbers-explained)
18. [Full Glossary](#18-full-glossary)

---

## 1. Why Fraud Detection is Hard

Before any technical detail, understand *why* this problem is uniquely difficult.

### The Class Imbalance Problem
In a dataset of 1 million transactions, maybe 1,000 are fraud. That's **0.1%**. If your model just says "not fraud" every time, it's 99.9% accurate — and completely useless. This is called **class imbalance** and it's the central challenge.

### Fraud Evolves
Fraudsters observe what gets blocked and adapt. A rule that catches fraud today may not catch it next week. This is why **unsupervised learning** (detecting *anything unusual*, not just known fraud) matters alongside supervised models.

### Speed Matters
A credit card transaction must be approved or declined in **under 100ms**. You can't run a slow model. Every design decision involves the trade-off: **accuracy vs. latency**.

### Two Types of Errors Matter
- **False Positive**: Flagging a legitimate transaction as fraud → customer frustration, blocked cards
- **False Negative**: Letting fraud through → financial loss

The job is to minimize both, but they pull in opposite directions.

---

## 2. Core Concepts You Must Know First

### What is a Feature?
A feature is any piece of information you feed into a model. For fraud:
- Raw features: `amount`, `merchant_category`, `country`
- Engineered features: `avg_spend_last_7_days`, `transactions_in_last_hour`, `distance_from_home`

**Feature engineering** — creating useful signals from raw data — is often more impactful than which model you pick.

### Supervised vs Unsupervised Learning

| | Supervised | Unsupervised |
|---|---|---|
| Needs labels? | Yes (fraud / not fraud) | No |
| What it learns | "These patterns = fraud" | "This looks unlike everything else" |
| Weakness | Can't catch new fraud types | Higher false positives |
| Example | XGBoost classifier | Isolation Forest |

The best systems use **both together**.

### What is a Score vs a Label?
- A **label** is binary: fraud (1) or not fraud (0)
- A **score** is a probability: 0.87 means "87% likely to be fraud"

Scores are more useful than labels because downstream systems (human analysts, rule engines) can set their own **thresholds** based on risk appetite.

### Precision, Recall, and the F1 Score

```
Precision = Of all flagged fraud, how many were actually fraud?
           = True Positives / (True Positives + False Positives)

Recall    = Of all actual fraud, how many did we catch?
           = True Positives / (True Positives + False Negatives)

F1 Score  = Harmonic mean of Precision and Recall
           = 2 * (Precision * Recall) / (Precision + Recall)
```

High precision → fewer false alarms. High recall → fewer missed frauds. You can't usually have both.

---

## 3. Real-Time Anomaly Detection & Streaming

### Resume Bullet:
> *"Designed real-time anomaly detection system with sliding window aggregations and velocity features over streaming event data, cutting false positive rates by 41% and flagging novel fraud patterns within 200ms of occurrence."*

### What is Streaming Event Data?
Every user action — a click, a transaction, a login — is an **event**. Streaming means these events flow in continuously, like a river. Unlike batch processing (processing yesterday's data overnight), streaming processes events the moment they arrive.

**Tools used:** Apache Kafka (event bus), Apache Spark Structured Streaming (processing engine)

### What is a Sliding Window Aggregation?

Imagine you want to know: "How much has this user spent in the last 1 hour?"

A **sliding window** looks back a fixed time period from *right now*, and that window moves forward as time passes.

```
Timeline: ----[--1 hour window--]---> now

As each new transaction arrives:
- Add it to the window
- Drop any transactions older than 1 hour
- Recompute: sum, count, average, max
```

**Why this reduces false positives:** Instead of comparing a $500 purchase to all historical behavior (too broad), you compare it to behavior in the last hour, day, or week — much more relevant context.

Common windows used:
- 1 minute (velocity detection)
- 1 hour (session-level behavior)
- 7 days (weekly spending patterns)
- 30 days (monthly baseline)

### What are Velocity Features?
Velocity = rate of change. In fraud, velocity features answer questions like:
- How many transactions in the last 10 minutes?
- How many different merchants in the last hour?
- How many failed login attempts in the last 5 minutes?

**Why they matter:** A fraudster who gets card details will try to use them quickly and repeatedly. Velocity spikes are a strong fraud signal.

```python
# Example: velocity feature computation
velocity_features = {
    "txn_count_1min":   count of transactions in last 1 minute,
    "txn_count_10min":  count of transactions in last 10 minutes,
    "txn_count_1hr":    count of transactions in last 1 hour,
    "unique_merchants_1hr": count of distinct merchants in last 1 hour,
    "total_amount_1hr": sum of amounts in last 1 hour,
    "amt_velocity":     current amount / avg_amount_last_7_days,
}
```

### 200ms Latency — Why That Matters
The entire pipeline — ingest event → compute features → run model → return decision — must complete in under 200ms (in this case). This forces engineering choices:
- Features must be **pre-computed** and cached, not computed on-demand
- Models must be **lightweight** at inference time
- No database joins at inference time; only key lookups

---

## 4. Unsupervised Anomaly Detection Methods

### Resume Bullet:
> *"Built production unsupervised anomaly detection pipeline combining Isolation Forest and autoencoder-based reconstruction error scoring on high-cardinality transaction data — detecting out-of-distribution behavior missed by supervised baselines."*

### What is "High-Cardinality" Data?
Cardinality = number of unique values. A feature like `merchant_id` might have 500,000 unique values. That's high cardinality. You can't one-hot-encode it (500,000 columns!), so you need special techniques like embeddings or frequency encoding.

### Isolation Forest

**Core Idea:** Normal points are hard to isolate; anomalies are easy to isolate.

Imagine a forest of random decision trees. At each node, the algorithm randomly picks a feature and a random split value. An anomaly — a point that is very different from others — gets isolated (separated into its own leaf) very quickly, with few splits. A normal point, surrounded by many similar points, takes many splits to isolate.

```
Anomaly score = average depth needed to isolate across all trees
              → Short depth = anomaly
              → Long depth = normal
```

**Why it works for fraud:**
- No labels needed
- Fast at inference time (just traverse trees)
- Works well with high-dimensional data
- Naturally handles mixed feature types

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,    # number of trees
    contamination=0.01,  # expected fraction of anomalies (~1%)
    random_state=42
)

# scores: -1 = anomaly, 1 = normal
# decision_function: negative = more anomalous
scores = model.fit_predict(X_transactions)
anomaly_scores = model.decision_function(X_transactions)
```

### Autoencoders and Reconstruction Error

**What is an Autoencoder?**
An autoencoder is a neural network trained to **compress then reconstruct** its own input.

```
Input (100 features)
       ↓
  Encoder (compress to 10 features — the "bottleneck")
       ↓
  Latent Space (compressed representation)
       ↓
  Decoder (expand back to 100 features)
       ↓
Output (reconstructed input)

Loss = difference between Input and Output (reconstruction error)
```

**Training:** Train only on normal transactions. The autoencoder learns to reconstruct normal behavior efficiently.

**At inference:** When a fraudulent/anomalous transaction comes in, the autoencoder struggles to reconstruct it accurately (it's never seen patterns like this), so **reconstruction error is high**.

```
Reconstruction Error = Mean Squared Error(original, reconstructed)

Low error  → similar to training data → likely normal
High error → unlike training data    → likely anomaly
```

**Why this is powerful:** The autoencoder learns a dense, compressed representation of "what normal looks like." Anything that doesn't fit that representation stands out automatically — even new fraud types it's never seen.

### What is "Out-of-Distribution" (OOD) Behavior?
A supervised model trained on known fraud patterns will only catch those patterns. But if fraudsters develop a new attack vector:
- The supervised model says "this doesn't look like any known fraud" → passes through
- The unsupervised model says "this doesn't look like normal behavior either" → flags it

OOD detection is catching the **unknown unknowns**.

---

## 5. Model Calibration

### Resume Bullet:
> *"Implemented model calibration (Platt scaling and isotonic regression) for probability-calibrated risk scores consumed by downstream decisioning systems, improving Brier score by 0.08 across deployed fraud models."*

### The Problem: Uncalibrated Scores
A model might output 0.9 for a transaction. Does that mean there's a 90% chance of fraud? **Not necessarily.** Raw model outputs are scores, not probabilities. An XGBoost model saying 0.9 might correspond to only 60% true fraud rate.

**Calibration** is the process of making model outputs match real probabilities.

```
Ideal calibration: when model says 0.8 → 80% of those cases are actually fraud
Uncalibrated:      when model says 0.8 → only 55% are fraud (overconfident)
```

### Platt Scaling
Fits a **logistic regression** on top of the raw model scores using a held-out validation set.

```python
from sklearn.calibration import CalibratedClassifierCV

# Platt scaling = sigmoid calibration
calibrated_model = CalibratedClassifierCV(
    base_estimator=xgboost_model,
    method='sigmoid',  # Platt scaling
    cv='prefit'        # model already trained
)
calibrated_model.fit(X_val, y_val)

# Now outputs are true probabilities
prob = calibrated_model.predict_proba(X_test)[:, 1]
```

### Isotonic Regression
A more flexible, non-parametric calibration method. Fits a **monotonically increasing step function** to the calibration data — no assumption of a specific shape (unlike Platt's sigmoid).

Better when you have lots of calibration data. More prone to overfitting with small samples.

### Brier Score
The Brier Score measures how well-calibrated a model is:

```
Brier Score = (1/N) * Σ (predicted_prob - actual_outcome)²

Range: 0 (perfect) to 1 (worst)
Lower is better.
```

Improving Brier score by 0.08 is significant — it means the risk scores are now meaningfully more trustworthy for downstream systems that make decisions based on those probabilities.

---

## 6. Adversarial Robustness Testing

### Resume Bullet:
> *"Conducted adversarial robustness testing using FGSM and PGD perturbations to stress-test production fraud models against evasion attacks — reducing adversarial vulnerability by 34% without accuracy regression."*

### What is an Evasion Attack?
A sophisticated fraudster reverse-engineers what your model looks for and slightly modifies their transactions to avoid detection. For example:
- Breaking one large transaction into many small ones (structuring)
- Adding small random noise to transaction amounts
- Changing transaction timing patterns slightly

This is called an **evasion attack** — the fraud still happens, but the model is fooled into thinking it's normal.

### FGSM — Fast Gradient Sign Method
FGSM is the simplest way to generate adversarial examples. It takes the gradient of the loss with respect to the input, and perturbs the input in the direction that maximizes the loss.

```
Adversarial example = original_input + ε * sign(∇_input Loss)

Where:
  ε = perturbation size (small → subtle attack)
  ∇_input Loss = gradient of model loss w.r.t. input features
  sign() = direction only (not magnitude)
```

**In fraud terms:** FGSM simulates a fraudster who knows exactly which features to tweak (by a tiny amount) to flip the model's prediction from "fraud" to "not fraud."

### PGD — Projected Gradient Descent
PGD is a stronger, iterative version of FGSM. It applies FGSM multiple times with a small step size, projecting back onto an allowed perturbation ball each time.

```
x_0 = original_input
For t in range(num_steps):
    x_{t+1} = Clip(x_t + α * sign(∇ Loss(x_t)), x_0 ± ε)

Where α = step size, ε = total budget
```

PGD finds much stronger adversarial examples because it searches iteratively.

### Why This Matters in Production
Fraud systems are high-stakes. Knowing that your model can be fooled by subtle feature perturbations is critical. Adversarial testing reveals these weaknesses before real attackers do.

**Reducing adversarial vulnerability by 34%** was likely achieved through:
- **Adversarial training**: Including adversarial examples in the training data
- **Feature smoothing**: Making the model less sensitive to tiny feature changes
- **Ensemble methods**: Harder to fool multiple models simultaneously

---

## 7. Feature Store (Feast + BigQuery)

### Resume Bullet:
> *"Implemented centralized Feature Store with Feast and BigQuery, eliminating training–serving skew and lifting production model accuracy by 18%."*

### What is Training-Serving Skew?
This is one of the most common — and most damaging — problems in production ML.

**Training:** You compute features from historical data in a batch job (e.g., "average spend last 7 days" computed from a database query).

**Serving:** At inference time, you compute the same feature differently (e.g., from a cache that's slightly stale, or uses different code).

The model was trained on one version of the feature, but at inference it sees a different version. This is **skew** — and it silently degrades model performance.

```
Training data:    avg_spend_7d = query from data warehouse (batch, accurate)
Production data:  avg_spend_7d = query from Redis cache (may be stale or computed differently)

Model sees different numbers → predictions drift from training expectations
```

### What is a Feature Store?
A Feature Store is a centralized system that:
1. **Defines** features once (single source of truth)
2. **Computes** features consistently (same code path for training and serving)
3. **Stores** both historical values (for training) and latest values (for serving)
4. **Serves** features at low latency for real-time inference

```
Feature Store Architecture:

Offline Store (BigQuery)          Online Store (Redis)
┌─────────────────────────┐       ┌──────────────────────┐
│ Historical feature       │       │ Latest feature values│
│ values for training      │ ←sync→│ for real-time serving│
│ (batch, large scale)     │       │ (low latency, <5ms)  │
└─────────────────────────┘       └──────────────────────┘
        ↑                                   ↑
   Training job                    Production model
```

### Feast
Feast is an open-source Feature Store. You define features as code:

```python
from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Float64, Int64

# Define what a "transaction entity" is
transaction = Entity(name="user_id", join_keys=["user_id"])

# Define a feature view (group of related features)
transaction_features = FeatureView(
    name="transaction_stats",
    entities=[transaction],
    schema=[
        Field(name="avg_spend_7d", dtype=Float64),
        Field(name="txn_count_1hr", dtype=Int64),
        Field(name="unique_merchants_7d", dtype=Int64),
    ],
    ttl=timedelta(days=1),
    source=bigquery_source,
)

# Training: pull historical features (point-in-time correct)
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["transaction_stats:avg_spend_7d"]
).to_df()

# Serving: pull latest features for real-time inference
features = store.get_online_features(
    features=["transaction_stats:avg_spend_7d"],
    entity_rows=[{"user_id": "user_123"}],
).to_dict()
```

**Point-in-time correctness** (for training) means: when you train on a transaction from 3 months ago, you use the feature value *as it was 3 months ago*, not the current value. This prevents **data leakage**.

---

## 8. LLM Fine-Tuning in Fraud Context

### Resume Bullet:
> *"Fine-tuned 7B/13B Qwen3-VL LLMs using SFT and QLoRA on AWS SageMaker GPU clusters, improving domain-specific accuracy by 21% while cutting inference cost by 60%."*

### Why Fine-Tune LLMs for Fraud?
LLMs are used in fraud systems for:
- Parsing and understanding unstructured documents (KYC forms, ID documents)
- Generating natural language explanations of fraud decisions
- Extracting entities from AML-related text (names, amounts, account numbers)
- Analyzing customer support conversations for fraud signals

### SFT — Supervised Fine-Tuning
Take a pre-trained LLM and continue training it on a domain-specific dataset of (instruction, response) pairs.

```
Base model: Qwen3-7B (pre-trained on internet text)
Your dataset: 10,000 examples of fraud-related documents + expected outputs
Fine-tuned model: Qwen3-7B that understands your specific domain
```

### QLoRA — Quantized Low-Rank Adaptation
Training a 7 billion parameter model is expensive. QLoRA makes it affordable:

**LoRA (Low-Rank Adaptation):** Instead of updating all 7B parameters, only update a small set of "adapter" matrices injected into the model. The original model weights stay frozen.

```
Original weight matrix W (frozen, 4096 × 4096 = 16M parameters)
LoRA adapters: A (4096 × 8) + B (8 × 4096) = 65,536 parameters

New output = W·x + BA·x
           = original output + small update

Training only updates A and B — ~0.4% of total parameters!
```

**Quantization (the Q in QLoRA):** Load the frozen base model in 4-bit precision (normally 32-bit), saving 8x memory. Adapters stay in 16-bit for training stability.

```
7B params × 32 bits = 28 GB GPU memory (expensive!)
7B params × 4 bits  =  3.5 GB GPU memory (affordable on a single GPU)
```

This is how a 7B model can be fine-tuned on a single A100 GPU instead of a cluster.

---

## 9. Real-Time Model Monitoring & Drift Detection

### Resume Bullet:
> *"Designed real-time model monitoring with drift detection and automated retraining triggers, cutting production incidents by 73% and MTTR from 4 hours to 23 minutes."*

### Why Models Degrade in Production
A fraud model trained in January might start performing poorly by June. Why?
1. **Fraud patterns change** — attackers adapt
2. **Data distribution shifts** — new products, new user demographics
3. **Feature pipeline changes** — upstream data changes subtly
4. **Concept drift** — the relationship between features and labels changes

Without monitoring, you won't know until your loss rate spikes.

### Types of Drift

**Data Drift (Feature Drift):** The input distribution changes.
```
Training:   avg_transaction = $85
Production: avg_transaction = $240 (inflation, new user segment, etc.)
```

**Concept Drift:** The relationship between inputs and outputs changes.
```
Training time:   high velocity + foreign country → fraud
Production time: high velocity + foreign country → normal (post-COVID travel resumed)
```

**Label Drift:** The fraud rate itself changes.
```
Training: 0.5% fraud rate
Production: 2.0% fraud rate (new attack campaign)
```

### Detection Methods

**Population Stability Index (PSI):** Measures how much a feature distribution has shifted.
```
PSI < 0.1:  No significant change
PSI 0.1-0.25: Some change, investigate
PSI > 0.25: Significant shift, retrain
```

**KL Divergence / Jensen-Shannon Divergence:** Statistical measures of how different two distributions are.

**Model Performance Metrics:** Track precision, recall, AUC-ROC on a rolling window of recent labeled data (requires a feedback loop — delayed labels from confirmed fraud reports).

### Automated Retraining Triggers
```python
# Pseudocode for monitoring system
if psi_score > 0.25:
    trigger_alert("Feature drift detected")
    
if model_auc_rolling_7d < 0.85:  # below threshold
    trigger_alert("Model performance degraded")
    schedule_retraining_job()
    
if fraud_rate_7d > fraud_rate_baseline * 1.5:
    trigger_alert("Fraud rate spike")
    escalate_to_team()
```

**MTTR (Mean Time to Recovery):** The average time between "something goes wrong" and "it's fixed." Dropping from 4 hours to 23 minutes means the system detects and responds to problems automatically, rather than waiting for a human to notice.

---

## 10. MLOps Stack

### Resume Bullet:
> *"Orchestrated cloud-native ML systems on Kubernetes with MLflow, Terraform, and Vertex AI Pipelines; led platform modernization for 15+ ML teams, compressing model deployment cycle from 3 weeks to 2 days."*

### MLflow — Experiment Tracking
MLflow tracks ML experiments: what hyperparameters were used, what metrics were achieved, which dataset version, which model artifacts.

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_metric("auc_roc", 0.94)
    mlflow.log_metric("false_positive_rate", 0.03)
    mlflow.sklearn.log_model(model, "fraud_model")
```

This creates a registry of every experiment ever run — you can compare, reproduce, and deploy any past model.

### Kubernetes — Container Orchestration
Kubernetes runs containerized services at scale. For ML:
- Each model is packaged as a Docker container
- Kubernetes manages scaling (more traffic → more replicas)
- Health checks restart failed containers automatically
- Rolling deployments mean no downtime during model updates

### Terraform — Infrastructure as Code
Instead of clicking through AWS/GCP consoles, you define all infrastructure in code:

```hcl
resource "google_container_cluster" "ml_cluster" {
  name     = "fraud-ml-cluster"
  location = "us-central1"
  
  node_pool {
    name       = "gpu-pool"
    node_count = 3
    node_config {
      machine_type = "n1-standard-8"
      guest_accelerator {
        type  = "nvidia-tesla-t4"
        count = 1
      }
    }
  }
}
```

Now your entire infrastructure can be version-controlled, reviewed, and reproduced exactly.

### Vertex AI Pipelines
Orchestrates multi-step ML workflows as directed acyclic graphs (DAGs):

```
Data Validation → Feature Engineering → Model Training → Evaluation → 
→ [If metrics pass] → Deployment → Monitoring
```

Each step is isolated, retryable, and tracked. If step 3 fails, you don't re-run steps 1 and 2.

---

## 11. Ensemble Methods for Fraud

### Resume Bullet:
> *"Built production fraud detection and credit risk scoring system using ensemble methods (XGBoost, LightGBM, Isolation Forest)..."*

### Why Ensembles?
No single model is best at everything. Ensembles combine multiple models so their weaknesses cancel out.

### XGBoost — eXtreme Gradient Boosting
XGBoost builds trees **sequentially**. Each new tree corrects the errors of all previous trees.

```
Tree 1: Learns to predict fraud/not-fraud from scratch
Tree 2: Focuses on transactions Tree 1 got wrong
Tree 3: Focuses on transactions Trees 1-2 got wrong
...
Final prediction: weighted sum of all trees
```

**Why it dominates tabular fraud data:**
- Handles missing values natively
- Very fast training with histogram-based splits
- Built-in regularization (prevents overfitting)
- Captures complex non-linear interactions between features

### LightGBM
Similar to XGBoost but builds trees **leaf-wise** (not level-wise). Grows the most impactful leaf first. Often faster and uses less memory — critical for large transaction datasets.

```
XGBoost grows:         LightGBM grows:
Level by level         Best leaf first

    root                   root
   /    \                 /    \
  L1     R1             L1     R1
 / \    / \            / \
L2  R2 L3  R3         L2  R2  ← grew this because it reduced loss most
```

### Stacking Multiple Models
```python
# Example: combine supervised and unsupervised scores
xgb_score       = xgb_model.predict_proba(X)[:, 1]      # 0 to 1
isolation_score = -isolation_forest.decision_function(X)  # higher = more anomalous
autoencoder_score = reconstruction_error(autoencoder, X)  # higher = more anomalous

# Normalize and combine
final_score = (
    0.5 * xgb_score +
    0.3 * normalize(isolation_score) +
    0.2 * normalize(autoencoder_score)
)
```

---

## 12. AML — Anti-Money Laundering Pipeline

### Resume Bullet:
> *"Developed AML transaction monitoring pipeline with unsupervised anomaly detection (autoencoders, DBSCAN clustering) to surface suspicious money flows and structuring patterns — flagging high-risk accounts with 91% precision for SAR filing workflows."*

### What is Money Laundering?
Money laundering is the process of making illegally-obtained money appear legitimate. The three stages:
1. **Placement:** Putting dirty money into the financial system (cash deposits, etc.)
2. **Layering:** Moving money through multiple transactions to obscure the trail
3. **Integration:** Merging the clean-looking money back into the economy

### Structuring (Smurfing)
A common technique: breaking large transactions into many small ones to avoid reporting thresholds. In the US, cash transactions over $10,000 are automatically reported. So a money launderer deposits $9,800, $9,500, $9,700 across multiple days.

**Detection:** Look for clusters of transactions just below reporting thresholds.

```python
# Structuring feature
def structuring_score(transactions, threshold=10000):
    suspicious = transactions[
        (transactions['amount'] > threshold * 0.8) &  # 80-100% of threshold
        (transactions['amount'] < threshold)
    ]
    return len(suspicious) / len(transactions)
```

### DBSCAN for Clustering Suspicious Accounts
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) groups points that are close together in feature space. Points with no nearby neighbors are labeled as noise (outliers).

```
Core concept:
- Dense region = cluster of similar behavior
- Sparse region = outlier

For AML:
- Cluster accounts with similar transaction patterns
- Outlier accounts = unusual behavior → investigate
```

**Why DBSCAN over K-Means for AML:**
- Doesn't need to specify number of clusters upfront
- Can find clusters of arbitrary shape
- Explicitly labels outliers as noise (not forced into a cluster)
- Robust to noise in the data

### SAR — Suspicious Activity Report
When a bank identifies a suspicious transaction pattern, it files a SAR with financial regulators. This is a legal requirement. The model's job is to **prioritize** which accounts get human analyst attention for SAR filing.

**91% precision** means: of every 100 accounts the model flags as SAR-worthy, 91 are genuinely suspicious. This is high — it keeps analyst workload manageable.

---

## 13. Graph-Based Fraud Detection

### Resume Bullet:
> *"Built graph-based fraud detection module using community detection algorithms (Louvain, label propagation) to map money flow networks, identifying fraud rings and mule account clusters invisible to row-level models."*

### Why Graphs?
Row-level models look at each transaction in isolation. But fraud rings involve **coordinated behavior across multiple accounts**. To see the network, you need a graph.

```
Transaction Graph:
Account A → Account B ($500)
Account B → Account C ($480)
Account C → Account D ($450)
Account D → Account A ($420)  ← circular flow! classic money laundering
```

A row-level model looking at Account B's single transaction ($480) has no idea it's part of a cycle.

### Graph Structure for Fraud
```
Nodes: accounts, devices, IP addresses, phone numbers, email addresses
Edges: transactions, shared attributes (same device, same address)

Edge weight: transaction amount, frequency
Edge direction: money flow direction
```

**Shared attribute links are powerful:**
```
Account X → registered on → Device_1
Account Y → registered on → Device_1
→ X and Y are likely the same person (or in the same ring)
```

### Louvain Community Detection
Louvain finds **communities** — groups of nodes that are more connected to each other than to the rest of the network.

```
Optimization target: Modularity
  = (actual edges within community) - (expected random edges within community)

Higher modularity = more distinct communities

Algorithm:
1. Each node starts as its own community
2. Greedily merge nodes that most increase modularity
3. Collapse communities into super-nodes, repeat
4. Until no improvement possible
```

**In fraud detection:** A community where money flows in circles, shared devices connect multiple accounts, and all accounts are newly opened = fraud ring.

### Label Propagation
A simpler algorithm that spreads labels through the network:
```
1. Known fraud accounts start with label "fraud"
2. Their neighbors get partially labeled "fraud" (weighted by connection strength)
3. Propagate outward until convergence
```

Even if you only know a few confirmed fraud accounts, label propagation can identify the broader network they belong to.

### Mule Accounts
Money mules are people who receive and transfer stolen funds, often unknowingly. They appear as intermediate nodes in the money flow graph — not the originators of fraud, but critical to the network. Graph analysis reveals them by their position (high betweenness centrality) rather than their individual behavior.

---

## 14. Multi-Agent AI Systems

### Resume Bullet:
> *"Architected multi-agent AI systems using LangGraph and CrewAI with stateful agent graphs — automating financial research pipelines across 8 data sources and cutting manual analyst overhead by 60%."*

### What is a Multi-Agent System?
Instead of one LLM doing everything, you have multiple specialized agents:

```
Orchestrator Agent
├── Research Agent    → searches news, filings, databases
├── Analysis Agent    → interprets the findings
├── Risk Agent        → evaluates AML/fraud risk
└── Report Agent      → writes the final narrative
```

Each agent has a specific role, tools, and context. They communicate through a shared state.

### LangGraph
LangGraph builds agent systems as **stateful graphs**. Nodes are agents or tools; edges are transitions between states.

```python
from langgraph.graph import StateGraph

workflow = StateGraph(AgentState)

# Add nodes (agents)
workflow.add_node("research", research_agent)
workflow.add_node("analyze", analysis_agent)
workflow.add_node("risk_score", risk_agent)

# Define transitions
workflow.add_edge("research", "analyze")
workflow.add_conditional_edges(
    "analyze",
    # If high risk → deep risk assessment; else → report
    lambda state: "risk_score" if state["risk"] > 0.7 else "report"
)

app = workflow.compile()
result = app.invoke({"entity": "Company XYZ", "task": "AML check"})
```

**Stateful** means the agent can remember what it's done across multiple steps — critical for multi-step research workflows.

### CrewAI
CrewAI provides a higher-level abstraction — you define agents with roles, goals, and backstories, and they collaborate:

```python
from crewai import Agent, Task, Crew

researcher = Agent(
    role="Financial Research Analyst",
    goal="Find all public information about the entity",
    tools=[web_search_tool, sec_filing_tool]
)

aml_analyst = Agent(
    role="AML Compliance Officer",
    goal="Identify money laundering red flags",
    tools=[sanctions_check_tool, pep_database_tool]
)

crew = Crew(agents=[researcher, aml_analyst], tasks=[...])
result = crew.kickoff()
```

---

## 15. LLM Deployment Optimization

### Resume Bullet:
> *"Deployed fine-tuned Qwen3 8B via ONNX and TensorRT optimization with INT8 quantization, achieving 3.2x throughput improvement and sub-90ms p99 latency."*

### The Problem: LLMs Are Slow and Expensive
A 7B parameter model in float32 requires 28GB of memory and takes hundreds of milliseconds per inference. For a production system serving millions of requests, this is prohibitively expensive.

### ONNX — Open Neural Network Exchange
ONNX is a standardized format for representing ML models. Convert from PyTorch → ONNX → run on any optimized runtime.

```python
# Export to ONNX
import torch.onnx
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=17,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq_len"}}
)
```

### TensorRT
NVIDIA's inference optimizer. Takes an ONNX model and:
1. **Fuses layers:** Combines multiple operations into one kernel
2. **Optimizes data types:** Uses FP16 or INT8 where precision loss is acceptable
3. **Calibrates for your GPU:** Optimizes specifically for the target hardware

Result: 3-5x faster inference on NVIDIA GPUs.

### INT8 Quantization
```
FP32: each weight is 32 bits (4 bytes)  → full precision
FP16: each weight is 16 bits (2 bytes)  → half precision, some loss
INT8: each weight is 8 bits (1 byte)    → 4x smaller, some loss

7B params × 32 bits = 28 GB
7B params × 8 bits  = 7 GB   → fits on a single GPU!
```

Post-training quantization with calibration:
```python
# Calibration: run representative examples through the model
# Measure the range of activations
# Scale them to fit in INT8 range [-128, 127]
# Accuracy loss: typically <1% with proper calibration
```

### p99 Latency
"p99 latency" = the 99th percentile latency. If your p99 is 90ms, it means 99% of requests complete in under 90ms. The worst 1% might take longer (due to garbage collection, cold starts, etc.).

p99 is more important than average latency for user-facing systems — it describes the worst-case experience for most users.

---

## 16. Projects Deep Dive

### TradeBrains — Real-Time Financial Intelligence & Fraud Analytics Platform

**Tech Stack:** Python, LangGraph, Kafka, Spark, NetworkX

#### 1. Graph-Based Fraud Detection with Louvain (89% recall across 500K records)
```
Steps:
1. Build transaction graph from 500K+ records
   - Nodes: accounts, devices, IPs
   - Edges: transactions with amount/timestamp

2. Run Louvain community detection
   - Finds tightly-connected account clusters

3. Feature engineering per community:
   - Cycle detection (money flowing in circles)
   - Avg account age within cluster
   - Cross-border transaction ratio
   - Velocity of fund movement through cluster

4. Flag communities where multiple features are anomalous
   - 89% recall means catching 89 out of 100 actual fraud rings
```

#### 2. Real-Time Feature Engineering (Kafka + Spark Structured Streaming)
```
Architecture:
Transaction Event → Kafka Topic → Spark Streaming Consumer
                                          ↓
                               Compute sliding windows:
                               - 1min, 10min, 1hr, 24hr, 7d
                                          ↓
                               Write to Feature Store (online)
                                          ↓
                               Risk scoring model reads features
                               (sub-second freshness guaranteed)
```

**Inter-account distance metrics:** How far (geographically or in network terms) are two accounts that transact together? Large distances with high frequency = suspicious.

#### 3. LangGraph Multi-Agent for AML Documents (F1: 0.71 → 0.89)
```
Agent Graph:
[Input Doc] → Extractor Agent → [Extraction Result]
                                      ↓
                              Validator Agent checks:
                              "Did we get all entities?"
                                      ↓
                         ┌── Yes: done ──┐
                         └── No: Retrieval Failure Detected
                                ↓
                        Reroute to different extraction strategy
                        (different prompt / tool / document chunk)
```

**F1 improvement from 0.71 to 0.89** means the agent system extracting names, amounts, and account numbers from AML documents improved significantly by having a self-correcting loop.

---

### txt2create.com — Multimodal AI Generation Platform

**Tech Stack:** PyTorch, CUDA, GRPO, QLoRA, vLLM, Kubernetes

#### 1. vLLM on Kubernetes (500+ concurrent requests, p99 < 4s)
vLLM is a serving framework specifically optimized for LLM inference. Its key innovation: **PagedAttention** — manages the KV cache like OS virtual memory, eliminating memory waste from fixed-length allocations.

```
Traditional serving: Reserve max_seq_len memory per request → wasteful
vLLM PagedAttention: Allocate in pages, share across requests → 3-4x more throughput
```

**GPU-aware scheduling in Kubernetes:**
```yaml
resources:
  limits:
    nvidia.com/gpu: "1"   # Request GPU resources
nodeSelector:
  accelerator: nvidia-tesla-a100   # Target specific GPU type
```

#### 2. GRPO — Group Relative Policy Optimization
A reinforcement learning algorithm for aligning LLM outputs to human preferences (an alternative to RLHF).

```
Standard SFT:    Train on (prompt, good_response) pairs
RLHF:            Train on human preference rankings, uses separate reward model
GRPO:            Generate multiple responses per prompt,
                 rank them with a reward function,
                 train model to prefer higher-ranked responses
                 (no separate reward model needed!)
```

**Custom reward functions** in this project rewarded:
- Prompt adherence (did the generation follow instructions?)
- Quality metrics specific to the generation type
- Diversity and creativity scores

**38% improvement over SFT-only** means GRPO-trained models follow user instructions much more reliably.

---

### DocAI — Intelligent Document Processing System

**Tech Stack:** Python, PyTorch, LayoutLMv3, LoRA, PEFT, Spark, FastAPI

#### 1. LayoutLMv3 — Understanding Document Layout
Standard NLP models understand text but ignore where text appears on the page. LayoutLMv3 is trained on three modalities simultaneously:

```
Input modalities:
1. Text tokens (words in the document)
2. 2D position embeddings (x, y coordinates of each word's bounding box)
3. Image patches (visual appearance of the document)

This lets the model understand:
"This number is in the top-right = invoice number"
"This number is in the bottom-right = total amount"
(Same number, different location → different meaning)
```

#### 2. Few-Shot LoRA Fine-Tuning (96% accuracy with 200 samples per class)
The challenge: You want to classify 12 document types (invoices, contracts, receipts, etc.) but only have 200 labeled examples per type — far too few to train from scratch.

**Solution:** LoRA fine-tuning on LayoutLMv3 (already pre-trained on millions of documents). The pre-trained model already understands document structure; you just need to adapt the final classification layers.

```
Few-shot LoRA pipeline:
1. Load LayoutLMv3 (pre-trained, frozen)
2. Add LoRA adapters to attention layers
3. Train only adapters on your 200 samples per class
4. Very few parameters to update → no overfitting on small dataset
5. Result: 96% accuracy across 12 document types
```

#### 3. CUDA-Accelerated Inference (4x throughput improvement)
```
Synchronous baseline (slow):
Request 1 → Process → Response
Request 2 → Process → Response  (waits for #1)

Async FastAPI with batch processing (fast):
Request 1 ─┐
Request 2 ─┼→ GPU Batch → All responses
Request 3 ─┘

GPU processes the entire batch in parallel (CUDA cores)
→ 10,000+ documents/hour
```

---

## 17. Key Metrics & Numbers Explained

| Metric | Value | What It Means |
|--------|-------|---------------|
| False positive rate reduction | 41% | 41% fewer legitimate transactions wrongly blocked |
| Flagging latency | 200ms | Novel fraud detected within 200 milliseconds |
| Brier score improvement | 0.08 | Risk probabilities are significantly more accurate |
| Adversarial vulnerability reduction | 34% | Much harder for attackers to game the model |
| Production incident reduction | 73% | Far fewer model failures in production |
| MTTR reduction | 4hr → 23min | Problems fixed 10x faster |
| Deployment cycle | 3 weeks → 2 days | New models ship 10x faster |
| False positives (fraud) | 22% reduction | 22% fewer wrongly-blocked transactions |
| AML precision | 91% | 91% of flagged accounts are genuinely suspicious |
| Graph recall | 89% | Catching 89% of all fraud rings |
| F1 improvement | 0.71 → 0.89 | Entity extraction dramatically more accurate |
| Throughput (LLM) | 3.2x | Serving 3.2x more requests per second |
| Accuracy (LayoutLMv3) | 96% | Near-perfect document classification |
| Inference requests/day | 2M+ | Large-scale production system |
| Cloud cost reduction | 55% | Half the cloud bill with optimization |

---

## 18. Full Glossary

| Term | Definition |
|------|-----------|
| **AML** | Anti-Money Laundering — regulations and systems to detect money laundering |
| **Anomaly** | A data point significantly different from the norm |
| **AUC-ROC** | Area Under the ROC Curve — model discrimination ability (1.0 = perfect) |
| **Autoencoder** | Neural network trained to compress and reconstruct data |
| **Behavioral Embedding** | Dense vector representation of a user's behavioral patterns |
| **Brier Score** | Calibration metric; MSE between predicted probabilities and outcomes |
| **Calibration** | Aligning model output scores to true probabilities |
| **Class Imbalance** | When one class (fraud) is far rarer than the other (normal) |
| **Community Detection** | Finding groups of densely connected nodes in a graph |
| **Concept Drift** | When the relationship between features and labels changes over time |
| **Data Drift** | When the distribution of input data changes over time |
| **DBSCAN** | Density-Based Spatial Clustering; finds clusters + explicitly labels outliers |
| **Ensemble** | Combining multiple models to get better predictions than any single model |
| **Evasion Attack** | Manipulating inputs to fool a fraud model into missing fraud |
| **False Negative** | Fraud that was not detected (missed fraud) |
| **False Positive** | Legitimate transaction incorrectly flagged as fraud |
| **Feature Engineering** | Creating informative signals from raw data |
| **Feature Store** | Centralized system for defining, storing, and serving ML features |
| **FGSM** | Fast Gradient Sign Method; simple adversarial attack |
| **Fine-tuning** | Continuing to train a pre-trained model on domain-specific data |
| **F1 Score** | Harmonic mean of precision and recall |
| **Graph** | Mathematical structure of nodes (entities) connected by edges (relationships) |
| **GRPO** | Group Relative Policy Optimization; RL-based LLM alignment |
| **High Cardinality** | Feature with very many unique values (e.g., merchant_id) |
| **INT8 Quantization** | Representing weights in 8-bit integers instead of 32-bit floats |
| **Isolation Forest** | Anomaly detection by isolating points using random trees |
| **KV Cache** | Key-Value cache in transformer inference; memory-intensive |
| **Kafka** | Distributed event streaming platform |
| **Kubernetes** | Container orchestration system |
| **LayoutLMv3** | Transformer model understanding text, layout, and visual content in documents |
| **LangGraph** | Framework for building stateful multi-agent LLM systems |
| **Latency** | Time taken to process a request |
| **LightGBM** | Fast gradient boosting; leaf-wise tree growth |
| **LoRA** | Low-Rank Adaptation; efficient fine-tuning with adapter matrices |
| **Louvain** | Community detection algorithm optimizing modularity |
| **MLflow** | Experiment tracking and model registry tool |
| **Money Mule** | Account used to transfer stolen funds, often unknowingly |
| **MTTR** | Mean Time to Recovery — how quickly incidents are resolved |
| **ONNX** | Open Neural Network Exchange; portable model format |
| **OOD** | Out-of-Distribution; inputs unlike the training data |
| **p99 Latency** | 99th percentile latency; the slowest 1% of requests |
| **PagedAttention** | vLLM's KV cache management (inspired by OS virtual memory) |
| **PGD** | Projected Gradient Descent; iterative adversarial attack |
| **Platt Scaling** | Logistic regression applied to model scores for calibration |
| **Point-in-Time** | Using feature values as they existed at a historical point (no leakage) |
| **Precision** | Of all flagged, how many are correct? |
| **PSI** | Population Stability Index; measures feature distribution shift |
| **QLoRA** | LoRA + 4-bit quantized base model for memory-efficient fine-tuning |
| **Recall** | Of all actual positives, how many were caught? |
| **Reconstruction Error** | Difference between autoencoder input and output |
| **SAR** | Suspicious Activity Report; legal filing for suspicious financial activity |
| **SFT** | Supervised Fine-Tuning; training on labeled (instruction, response) pairs |
| **Sliding Window** | Aggregation over a moving time window (e.g., "last 1 hour") |
| **Spark** | Distributed computing framework for large-scale data processing |
| **Structuring** | Breaking transactions into smaller pieces to avoid reporting thresholds |
| **TensorRT** | NVIDIA's inference optimizer for GPU acceleration |
| **Training-Serving Skew** | When features computed differently at training vs inference time |
| **Velocity Feature** | Rate of an event occurring (e.g., transactions per minute) |
| **vLLM** | High-throughput LLM serving library |
| **XGBoost** | Extreme Gradient Boosting; sequential tree-based ensemble method |

---

*This guide was written to teach every concept from the resume in depth. For each topic, start by understanding the "why" before the "how."*



# 🏗️ Project Architecture Deep-Dive
## TradeBrains · txt2create.com · DocAI

> Full system design, data flows, component-level explanation, and the "why" behind every decision.

---

# PROJECT 1: TradeBrains
## Real-Time Financial Intelligence & Fraud Analytics Platform

---

## 🎯 What Problem Does This Solve?

TransUnion (and similar credit bureaus) provides **credit card transaction data** — millions of rows of raw financial events:

```
{
  "transaction_id": "TXN_98234",
  "account_id": "ACC_00123",
  "amount": 452.00,
  "merchant": "ONLINE_STORE_XYZ",
  "merchant_category": "ecommerce",
  "timestamp": "2024-03-15T14:23:05Z",
  "country": "US",
  "ip_address": "192.168.x.x",
  "device_id": "DEV_77331",
  "card_present": false
}
```

The challenge: **500K+ records** flowing continuously. Within each record, fraud is invisible. But across records — across accounts, devices, timeframes — fraud rings emerge. TradeBrains is the platform that sees those patterns in real time.

---

## 🗺️ Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION LAYER                                 │
│                                                                               │
│  TransUnion Credit Card Data Feed                                             │
│  (batch + streaming, 500K+ records)                                           │
│         │                                                                     │
│         ▼                                                                     │
│  ┌─────────────┐    ┌──────────────────────────────────────────────────┐     │
│  │  Kafka       │    │  Topics:                                          │     │
│  │  Event Bus   │───▶│  - raw_transactions     (all incoming txns)       │     │
│  │             │    │  - high_risk_alerts      (flagged in real time)    │     │
│  │             │    │  - graph_events          (account link updates)    │     │
│  └─────────────┘    └──────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      REAL-TIME FEATURE ENGINEERING LAYER                     │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  Spark Structured Streaming (consumes from Kafka)                     │    │
│  │                                                                        │    │
│  │  Per account, compute in real time:                                    │    │
│  │  ┌──────────────┐ ┌──────────────┐ ┌───────────────────────────────┐  │    │
│  │  │ Sliding       │ │ Velocity     │ │ Inter-account Distance Metrics │  │    │
│  │  │ Window Aggs   │ │ Features     │ │                                │  │    │
│  │  │               │ │              │ │ - Graph distance between       │  │    │
│  │  │ - amt_1min    │ │ - txn/1min   │ │   transacting accounts         │  │    │
│  │  │ - amt_10min   │ │ - txn/10min  │ │ - Shared device count          │  │    │
│  │  │ - amt_1hr     │ │ - txn/1hr    │ │ - Geographic distance          │  │    │
│  │  │ - amt_24hr    │ │ - merchants/ │ │ - Time since last shared txn   │  │    │
│  │  │ - amt_7d      │ │   hr         │ │                                │  │    │
│  │  └──────────────┘ └──────────────┘ └───────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│         │                                                                     │
│         ▼                                                                     │
│  ┌─────────────────────────────────────────────────────┐                     │
│  │  Feature Store (Online: Redis, Offline: BigQuery)    │                     │
│  │  Sub-second freshness guaranteed                      │                     │
│  └─────────────────────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────────────────┐
         ▼                                             ▼
┌─────────────────────────┐             ┌─────────────────────────────────────┐
│   RISK SCORING LAYER    │             │      GRAPH ANALYSIS LAYER            │
│                         │             │                                       │
│  XGBoost / LightGBM     │             │  NetworkX Transaction Graph           │
│  Isolation Forest       │             │                                       │
│  Autoencoder            │             │  Nodes: accounts, devices, IPs        │
│                         │             │  Edges: transactions, shared attrs    │
│  → Real-time score      │             │                                       │
│    per transaction      │             │  Louvain Community Detection          │
│    <200ms               │             │  → Finds fraud rings                  │
│                         │             │                                       │
│                         │             │  Label Propagation                    │
│                         │             │  → Spreads known fraud labels         │
└──────────┬──────────────┘             └───────────────┬─────────────────────┘
           │                                            │
           └──────────────────┬─────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INTELLIGENCE LAYER                                   │
│                                                                               │
│  LangGraph Multi-Agent System                                                 │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │  [Document Input: AML Filing, KYC Doc, Wire Transfer Record]          │    │
│  │         ↓                                                              │    │
│  │  [Extractor Agent] ──→ [Validator Agent]                               │    │
│  │         ↑                     │                                        │    │
│  │         │         Failure?    ↓                                        │    │
│  │         └──────── [Router Agent: reroutes to alternate strategy]       │    │
│  │                               │                                        │    │
│  │                         Success? ↓                                     │    │
│  │                   [Risk Scoring Agent]                                  │    │
│  │                               ↓                                        │    │
│  │                   [Report Generation Agent]                             │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT / ACTION LAYER                                │
│                                                                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐    │
│  │ Real-time Alert  │  │ SAR Filing Queue  │  │ Analyst Dashboard        │    │
│  │ (block/review)   │  │ (high-risk accts) │  │ (graph visualization,    │    │
│  │                  │  │                   │  │  trend monitoring)       │    │
│  └─────────────────┘  └──────────────────┘  └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Component 1: TransUnion Data Ingestion

### What is TransUnion Data?
TransUnion is one of the three major credit bureaus. They aggregate credit card transaction data across banks and lenders. The raw data pipeline receives:
- **Credit card transactions** (purchases, refunds, balance transfers)
- **Account metadata** (credit limit, account age, delinquency history)
- **Device and behavioral signals** (IP, device fingerprints, geolocation)

### How the Pipeline Ingests It

```
TransUnion API / S3 Batch Feed
         │
         ├── Batch (historical): Parquet files → Spark → BigQuery (offline store)
         │
         └── Streaming (real-time): Webhook / Kinesis → Kafka → Spark Streaming

Kafka Topic: raw_transactions
  Partition key: account_id (ensures all events for same account go to same partition)
  Retention: 7 days
  Throughput: designed for 50K+ messages/second at peak
```

**Why partition by `account_id`?**
All Spark stateful operations (sliding windows) need all events for the same account to arrive at the same worker. Partitioning by `account_id` guarantees this.

---

## 📦 Component 2: Real-Time Feature Engineering (Kafka + Spark Streaming)

This is the heart of TradeBrains. Features must be **fresh** (sub-second) and **accurate** (same computation as training time).

### Sliding Window Aggregations in Spark

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import window, sum, count, avg, countDistinct, col

spark = SparkSession.builder.appName("TradeBrains-Features").getOrCreate()

# Read from Kafka
raw_stream = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "raw_transactions") \
    .load()

# Parse JSON
transactions = raw_stream.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# --- Sliding Window Feature: amounts per account ---
amount_windows = transactions \
    .withWatermark("timestamp", "5 minutes") \  # handle late data
    .groupBy(
        col("account_id"),
        window(col("timestamp"), "1 hour", "1 minute")   # 1hr window, slide every 1min
    ) \
    .agg(
        sum("amount").alias("total_amount_1hr"),
        count("*").alias("txn_count_1hr"),
        avg("amount").alias("avg_amount_1hr"),
        countDistinct("merchant_id").alias("unique_merchants_1hr"),
        countDistinct("country").alias("unique_countries_1hr"),
    )

# Write to Feature Store (Redis online store)
amount_windows.writeStream \
    .foreachBatch(write_to_redis) \
    .outputMode("update") \
    .start()
```

### Velocity Features
```python
# Velocity: rate of transactions per time bucket
velocity = transactions \
    .withWatermark("timestamp", "2 minutes") \
    .groupBy("account_id", window("timestamp", "10 minutes", "1 minute")) \
    .agg(
        count("*").alias("txn_velocity_10min"),
        sum(col("amount") > 1000).alias("large_txn_count_10min"),  # count txns > $1000
    )
```

### Inter-Account Distance Metrics
This is the unique feature of TradeBrains — measuring how "distant" two accounts are before they transact together.

```python
# For each transaction between Account A and Account B, compute:

# 1. Graph distance (shortest path in transaction graph)
#    - If A → B → C → D have transacted, A and D are distance 3
#    - A and D suddenly transacting directly is unusual

# 2. Geographic distance
#    - Distance between last known locations of both accounts
#    - Two accounts 2000 miles apart suddenly transacting = suspicious

# 3. Behavioral distance
#    - Cosine similarity between account spending embeddings
#    - High behavioral dissimilarity + new connection = suspicious

def compute_distance_features(account_a, account_b, graph, embeddings):
    return {
        "graph_distance": nx.shortest_path_length(graph, account_a, account_b),
        "geo_distance_km": haversine(
            get_last_location(account_a),
            get_last_location(account_b)
        ),
        "behavioral_similarity": cosine_similarity(
            embeddings[account_a],
            embeddings[account_b]
        ),
        "days_since_connection": days_since_first_transaction(account_a, account_b),
        "is_new_connection": not graph.has_edge(account_a, account_b)
    }
```

---

## 📦 Component 3: Graph-Based Fraud Detection with NetworkX + Louvain

### Building the Transaction Graph

```python
import networkx as nx
from community import community_louvain  # python-louvain library

# Build directed weighted graph
G = nx.DiGraph()

for txn in transactions:
    # Money flow edge
    G.add_edge(
        txn["sender_account"],
        txn["receiver_account"],
        weight=txn["amount"],
        timestamp=txn["timestamp"],
        txn_id=txn["transaction_id"]
    )

# Also add "shared attribute" edges (undirected)
for pair in shared_device_pairs:
    G.add_edge(pair[0], pair[1], edge_type="shared_device", weight=10.0)

for pair in shared_ip_pairs:
    G.add_edge(pair[0], pair[1], edge_type="shared_ip", weight=5.0)
```

### What the Graph Looks Like (conceptually)

```
Normal behavior:
  Customer → Walmart
  Customer → Amazon
  Customer → Local Restaurant
  → Star pattern, all edges pointing OUT to merchants

Fraud ring behavior:
  ACC_001 → ACC_002 → ACC_003 → ACC_004 → ACC_001  ← circular!
      ↕           ↕
  (shared device)(shared IP)
  → Dense cluster with circular flows
```

### Louvain Community Detection

```python
# Convert to undirected for community detection
G_undirected = G.to_undirected()

# Run Louvain — finds communities automatically
partition = community_louvain.best_partition(G_undirected)
# partition = {account_id: community_id, ...}
# e.g. {"ACC_001": 0, "ACC_002": 0, "ACC_003": 0, "ACC_004": 1, ...}

# Analyze each community
for community_id in set(partition.values()):
    members = [acc for acc, c in partition.items() if c == community_id]
    subgraph = G.subgraph(members)
    
    community_features = {
        "size": len(members),
        "has_cycle": len(list(nx.simple_cycles(subgraph))) > 0,
        "avg_account_age_days": np.mean([get_account_age(m) for m in members]),
        "total_flow": sum(d["weight"] for _, _, d in subgraph.edges(data=True)),
        "cross_border_ratio": cross_border_transactions(subgraph) / total_transactions(subgraph),
        "shared_device_count": count_shared_devices(members),
        "density": nx.density(subgraph),  # how interconnected they are
    }
    
    # Rule: Dense + new accounts + circular flow + shared devices = fraud ring
    risk_score = fraud_ring_classifier.predict([community_features])
```

### Why 89% Recall?
Recall = catching 89 out of 100 actual fraud rings. The graph approach catches rings that row-level models miss entirely — because individual accounts in the ring might look completely normal in isolation.

---

## 📦 Component 4: LangGraph Multi-Agent System (F1: 0.71 → 0.89)

### The Problem It Solves
AML-relevant documents (wire transfer confirmations, KYC forms, suspicious activity narratives) need structured data extracted:
- Entity names (individuals, companies)
- Account numbers
- Transaction amounts and dates
- Jurisdictions / countries

Simple LLM extraction fails ~30% of the time on complex documents.

### Agent Graph Architecture

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class ExtractionState(TypedDict):
    document: str
    extracted_entities: dict
    extraction_attempts: int
    retrieval_failures: List[str]
    final_output: dict

workflow = StateGraph(ExtractionState)

# --- Node 1: Extractor Agent ---
def extractor_agent(state: ExtractionState):
    """Tries to extract entities from the document"""
    prompt = f"""
    Extract the following from this AML document:
    - All person names and company names
    - All account numbers (format: check for routing + account)
    - All transaction amounts with currencies
    - All countries/jurisdictions mentioned
    
    Document: {state['document']}
    
    Return as JSON.
    """
    result = llm.invoke(prompt)
    return {"extracted_entities": parse_json(result)}

# --- Node 2: Validator Agent ---
def validator_agent(state: ExtractionState):
    """Checks if extraction is complete and coherent"""
    entities = state["extracted_entities"]
    failures = []
    
    # Check for missing required fields
    if not entities.get("account_numbers"):
        failures.append("No account numbers found — document may need OCR pre-processing")
    if not entities.get("amounts"):
        failures.append("No amounts found — table extraction may have failed")
    if len(entities.get("names", [])) == 0:
        failures.append("No entity names found — document language may be non-English")
    
    return {"retrieval_failures": failures}

# --- Node 3: Router Agent ---
def router_agent(state: ExtractionState):
    """Decides what to do based on failures"""
    if not state["retrieval_failures"] or state["extraction_attempts"] > 3:
        return "success"  # done or give up
    
    failure_types = state["retrieval_failures"]
    
    if "OCR" in str(failure_types):
        return "ocr_retry"       # route to OCR-enhanced extraction
    elif "table" in str(failure_types):
        return "table_retry"     # route to table-specific parser
    elif "non-English" in str(failure_types):
        return "translate_retry" # route to translation first
    else:
        return "chunk_retry"     # try chunking the document differently

# Add nodes
workflow.add_node("extractor", extractor_agent)
workflow.add_node("validator", validator_agent)
workflow.add_node("router", router_agent)
workflow.add_node("risk_scorer", risk_scoring_agent)

# Add edges
workflow.set_entry_point("extractor")
workflow.add_edge("extractor", "validator")
workflow.add_conditional_edges("validator", router_agent, {
    "success":        "risk_scorer",
    "ocr_retry":      "extractor",   # loops back with different strategy
    "table_retry":    "extractor",
    "translate_retry":"extractor",
    "chunk_retry":    "extractor",
})
workflow.add_edge("risk_scorer", END)

app = workflow.compile()
```

### Why F1 Improved from 0.71 to 0.89
```
Without self-correction (F1 = 0.71):
- LLM tries once, returns partial result
- Missing fields treated as "not present" → false negatives
- Malformed documents silently fail

With LangGraph retry loop (F1 = 0.89):
- Validator catches specific failure modes
- Router selects the right remediation strategy
- System tries up to 3 different extraction approaches
- Falls back gracefully with partial results vs. empty result
```

---

---

# PROJECT 2: txt2create.com
## Multimodal AI Generation Platform

---

## 🎯 What Problem Does This Solve?

txt2create.com is a consumer-facing platform where users generate AI content (images, video, text, multimodal outputs) using open-source models — **self-hosted**, not using expensive third-party APIs like OpenAI or Midjourney.

The core engineering challenge: **Make self-hosted LLM/diffusion model inference scale to hundreds of simultaneous users, at low cost, with low latency.**

---

## 🗺️ Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                                       │
│                                                                               │
│   Browser / Mobile App                                                        │
│   User submits: "Generate a product photo of a sneaker on a marble floor"    │
└─────────────────────────────────────────────────────────────────────────────┘
         │  HTTPS request
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TYPESCRIPT / NODE.JS ORCHESTRATION LAYER                  │
│                                                                               │
│  ┌───────────────────────┐   ┌──────────────────────────────────────────┐   │
│  │  FastAPI REST Gateway │   │  Async Job Queue (BullMQ / Redis)         │   │
│  │  (request validation, │──▶│                                           │   │
│  │   auth, rate limiting)│   │  Job: {                                   │   │
│  └───────────────────────┘   │    id: "job_abc123",                      │   │
│                               │    prompt: "sneaker on marble floor",     │   │
│                               │    model: "SDXL",                         │   │
│                               │    priority: "standard",                  │   │
│                               │    user_id: "user_xyz"                    │   │
│                               │  }                                        │   │
│                               └──────────────────────────────────────────┘   │
│                                          │                                    │
│  ┌────────────────────────────────────── │─────────────────────────────────┐ │
│  │  Model Load Balancer                  │                                  │ │
│  │                                       │                                  │ │
│  │  Checks which backend has capacity:   ▼                                  │ │
│  │  ┌────────────────┐ ┌───────────────────────┐ ┌─────────────────────┐  │ │
│  │  │ vLLM Backend 1 │ │ vLLM Backend 2        │ │ vLLM Backend 3      │  │ │
│  │  │ (GPU: A100)    │ │ (GPU: A100)           │ │ (GPU: A100)         │  │ │
│  │  │ Load: 45%      │ │ Load: 78% → skip      │ │ Load: 30% → route   │  │ │
│  │  └────────────────┘ └───────────────────────┘ └─────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      KUBERNETES + vLLM INFERENCE LAYER                       │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Kubernetes Cluster (GKE / EKS)                                       │   │
│  │                                                                        │   │
│  │  ┌────────────────────────────────────────┐                           │   │
│  │  │  GPU Node Pool                          │                           │   │
│  │  │  nodeSelector: accelerator=a100         │                           │   │
│  │  │                                          │                           │   │
│  │  │  Pod: vllm-server-1                     │                           │   │
│  │  │  ┌──────────────────────────────────┐   │                           │   │
│  │  │  │  vLLM (PagedAttention)            │   │                           │   │
│  │  │  │  Model: Qwen3-8B (fine-tuned)    │   │                           │   │
│  │  │  │  Max concurrent: 200 requests    │   │                           │   │
│  │  │  │  Batch size: dynamic             │   │                           │   │
│  │  │  └──────────────────────────────────┘   │                           │   │
│  │  │                                          │                           │   │
│  │  │  Pod: vllm-server-2 (same config)       │                           │   │
│  │  │  Pod: vllm-server-3 (same config)       │                           │   │
│  │  └────────────────────────────────────────┘                           │   │
│  │                                                                        │   │
│  │  HPA (Horizontal Pod Autoscaler):                                      │   │
│  │  if GPU utilization > 80% → scale up pods                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RESULT DELIVERY                                      │
│                                                                               │
│  Generated output stored → S3/GCS                                            │
│  Signed URL returned to client                                                │
│  WebSocket pushes completion notification                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Component 1: vLLM Inference Stack

### Why vLLM?

Standard LLM serving wastes GPU memory. Here's why:

```
Problem with naive serving:
  Request arrives: "Write a poem" → model reserves memory for MAX_TOKENS=2048
  Model only generates 80 tokens → 1968 tokens of memory wasted
  This memory block is unavailable to other requests

vLLM's PagedAttention solution:
  Memory managed in "pages" (like OS virtual memory)
  Only allocate pages as tokens are generated
  Pages can be shared across requests (for shared prefixes)
  Result: 3-4x more requests fit in the same GPU memory
```

```python
# Launching vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model /models/qwen3-8b-finetuned \
    --tensor-parallel-size 1 \         # use 1 GPU per replica
    --max-num-seqs 200 \               # 200 concurrent requests
    --max-model-len 4096 \             # max context length
    --quantization awq \               # AWQ quantization for memory saving
    --gpu-memory-utilization 0.90 \    # use 90% of GPU memory
    --enable-prefix-caching            # cache common prefixes (system prompts)
```

### GPU-Aware Kubernetes Scheduling

```yaml
# Kubernetes deployment spec
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
spec:
  replicas: 3
  template:
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-a100
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          resources:
            requests:
              nvidia.com/gpu: "1"
              memory: "40Gi"
            limits:
              nvidia.com/gpu: "1"
              memory: "40Gi"
```

**500+ concurrent requests at p99 < 4s** is achieved by:
1. 3 pods × 200 concurrent each = 600 concurrent slots
2. PagedAttention keeps GPU utilization high
3. Async job queue absorbs burst traffic, prevents overload

---

## 📦 Component 2: GRPO Fine-Tuning Pipeline

### What is GRPO?

GRPO (Group Relative Policy Optimization) is a recent alternative to PPO/RLHF for aligning LLMs. The key difference:

```
RLHF (traditional):
  1. Train separate reward model on human preferences
  2. Use reward model to score LLM outputs
  3. Optimize LLM with PPO
  Complexity: two models to train, unstable training

GRPO:
  1. Generate G responses per prompt (e.g., G=8)
  2. Score each with your reward function(s)
  3. Compute advantage = (score - mean(scores)) / std(scores)
  4. Optimize LLM to prefer higher-advantage responses
  Complexity: one model, no separate reward model needed
```

### Custom Reward Functions for txt2create

```python
def compute_rewards(prompts: List[str], responses: List[str]) -> List[float]:
    rewards = []
    for prompt, response in zip(prompts, responses):
        score = 0.0
        
        # Reward 1: Prompt adherence (did it follow instructions?)
        adherence = clip_similarity(prompt, response)    # CLIP score
        score += 0.4 * adherence
        
        # Reward 2: Quality (aesthetic score for images, coherence for text)
        quality = aesthetic_scorer(response)
        score += 0.3 * quality
        
        # Reward 3: Safety (penalize harmful content)
        safety = safety_classifier(response)
        score -= 0.5 * (1 - safety)
        
        # Reward 4: Format compliance (did it return valid JSON/structure?)
        format_ok = validate_output_format(response)
        score += 0.2 * float(format_ok)
        
        # Reward 5: Diversity (penalize repeating the same output)
        diversity = novelty_score(response, recent_outputs)
        score += 0.1 * diversity
        
        rewards.append(score)
    return rewards
```

### GRPO Training Loop

```python
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    num_generations=8,          # Generate 8 responses per prompt
    max_prompt_length=512,
    max_completion_length=512,
    learning_rate=1e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[compute_rewards],   # your custom reward
    args=config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
# After training: model generates outputs that score higher on your reward functions
# Result: 38% improvement in prompt adherence vs SFT-only baseline
```

---

## 📦 Component 3: TypeScript Orchestration Layer

### Async Job Queue Architecture

```
Why async queues?
  User requests take 2-30 seconds (LLM generation)
  You can't keep HTTP connections open that long (timeouts, mobile disconnects)
  Solution: Submit job → get job_id → poll or websocket for result

Architecture:
  POST /generate → creates job, returns {job_id: "abc123", status: "queued"}
  GET  /jobs/abc123 → returns {status: "processing" | "complete" | "failed"}
  WS   /jobs/abc123/stream → streams tokens as they're generated
```

```typescript
// BullMQ job queue
import { Queue, Worker } from 'bullmq';
import Redis from 'ioredis';

const connection = new Redis({ host: 'redis', port: 6379 });
const generationQueue = new Queue('generation', { connection });

// Add job to queue
async function submitGenerationJob(request: GenerationRequest) {
  const job = await generationQueue.add(
    'generate',
    { prompt: request.prompt, model: request.model, userId: request.userId },
    {
      priority: request.isPremium ? 1 : 10,  // premium users get higher priority
      attempts: 3,                             // retry up to 3 times on failure
      backoff: { type: 'exponential', delay: 2000 }
    }
  );
  return { jobId: job.id, status: 'queued' };
}

// Worker processes jobs
const worker = new Worker('generation', async (job) => {
  const backend = loadBalancer.selectBackend();  // pick least-loaded vLLM instance
  const result = await backend.generate(job.data);
  await storage.upload(result, `outputs/${job.id}`);
  return { outputUrl: storage.getSignedUrl(`outputs/${job.id}`) };
}, { connection, concurrency: 20 });
```

### MLflow Experiment Tracking (20+ fine-tuning runs)

```python
import mlflow

# Each fine-tuning run is tracked
with mlflow.start_run(run_name=f"grpo_run_{timestamp}"):
    mlflow.log_params({
        "model": "qwen3-8b",
        "num_generations": 8,
        "learning_rate": 1e-6,
        "reward_functions": "adherence+quality+safety+format+diversity",
        "training_steps": 5000,
    })
    
    for step, batch_rewards in enumerate(training_loop()):
        mlflow.log_metrics({
            "mean_reward": np.mean(batch_rewards),
            "reward_std": np.std(batch_rewards),
            "prompt_adherence_score": eval_adherence(),
            "gpu_memory_gb": get_gpu_memory(),
        }, step=step)
    
    mlflow.pytorch.log_model(model, "grpo_fine_tuned_model")
```

The MLflow UI shows all 20+ runs side by side — you can compare which reward function weights, learning rates, and generation counts produced the best prompt-adherence improvement.

---

---

# PROJECT 3: DocAI
## Intelligent Document Processing System

---

## 🎯 What Problem Does This Solve?

Enterprises process millions of documents — invoices, contracts, insurance forms, tax documents, medical records. Manually extracting structured data from them is expensive and error-prone.

DocAI automates this: **Feed any document → Get structured data out**, across 12 document types, at 10,000+ documents/hour.

The hard part: **Only 200 labeled samples per document type** (getting labeled data is expensive). The solution: leverage a pre-trained model that already understands documents deeply.

---

## 🗺️ Full System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                        │
│                                                                               │
│  Documents arrive via:                                                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ S3 bucket   │  │ REST API     │  │ Email        │  │ Scanner/OCR     │  │
│  │ drop        │  │ upload       │  │ attachment   │  │ integration     │  │
│  └──────┬──────┘  └──────┬───────┘  └──────┬───────┘  └────────┬────────┘  │
│         └────────────────┴──────────────────┴───────────────────┘           │
│                                    │                                          │
│                                    ▼                                          │
│                    ┌───────────────────────────────┐                         │
│                    │  Apache Airflow DAG            │                         │
│                    │  (orchestrates processing)     │                         │
│                    └───────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PRE-PROCESSING LAYER                                  │
│                                                                               │
│  ┌─────────────────────────┐    ┌──────────────────────────────────────┐    │
│  │  Document Intake        │    │  For each document:                   │    │
│  │                         │    │                                       │    │
│  │  1. PDF → image pages   │    │  1. Detect language                  │    │
│  │  2. Image denoising     │    │  2. Detect orientation (rotate if    │    │
│  │  3. OCR (Tesseract /    │    │     needed)                          │    │
│  │     AWS Textract)       │    │  3. Extract bounding boxes for       │    │
│  │  4. Layout detection    │    │     each word                        │    │
│  │     (SAM integration)   │    │  4. Classify document type           │    │
│  └─────────────────────────┘    └──────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LayoutLMv3 INFERENCE LAYER                              │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                        │   │
│  │  Input to model:                                                       │   │
│  │  ┌──────────────────┐  ┌────────────────────┐  ┌──────────────────┐  │   │
│  │  │ Text tokens      │  │ 2D positions       │  │ Image patches    │  │   │
│  │  │ ["Invoice", "#", │  │ [(x1,y1,x2,y2) of │  │ (visual patches  │  │   │
│  │  │  "2024-03", ...] │  │  each token]       │  │  of document)    │  │   │
│  │  └──────────────────┘  └────────────────────┘  └──────────────────┘  │   │
│  │                              │                                         │   │
│  │                    LayoutLMv3 Transformer                              │   │
│  │                    (pre-trained on IIT-CDIP, DocVQA)                  │   │
│  │                              │                                         │   │
│  │                    LoRA Adapters (fine-tuned on your 200 samples)     │   │
│  │                              │                                         │   │
│  │                    ┌─────────▼──────────┐                             │   │
│  │                    │ Token Classification│                             │   │
│  │                    │ B-INVOICE_NO        │  ← "Invoice" tagged        │   │
│  │                    │ I-INVOICE_NO        │  ← "#2024-03" tagged       │   │
│  │                    │ B-DATE              │  ← date field tagged       │   │
│  │                    │ B-TOTAL_AMOUNT      │  ← amount field tagged     │   │
│  │                    └────────────────────┘                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     FastAPI ASYNC SERVING LAYER                              │
│                                                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                        │   │
│  │  POST /process-document                                                │   │
│  │    └─→ Adds to async task queue (Celery + Redis)                      │   │
│  │                                                                        │   │
│  │  GPU Batch Processing:                                                 │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │   │
│  │  │ docs_in_queue:  [doc1, doc2, doc3, ... doc32]  ← batch of 32   │  │   │
│  │  │                                                                   │  │   │
│  │  │ GPU processes ALL 32 in parallel (CUDA parallelism)              │  │   │
│  │  │ Time for 1 doc: ~400ms                                            │  │   │
│  │  │ Time for 32 docs (batched): ~450ms → 71x effective throughput    │  │   │
│  │  └─────────────────────────────────────────────────────────────────┘  │   │
│  │                                                                        │   │
│  │  Throughput: 10,000+ documents/hour (4x over synchronous baseline)    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                       │
│                                                                               │
│  Structured JSON output per document:                                        │
│  {                                                                            │
│    "document_type": "invoice",                                               │
│    "confidence": 0.97,                                                       │
│    "fields": {                                                               │
│      "invoice_number": "INV-2024-0312",                                     │
│      "date": "2024-03-12",                                                   │
│      "vendor": "Acme Corp",                                                  │
│      "total_amount": 4521.00,                                                │
│      "line_items": [...]                                                     │
│    }                                                                          │
│  }                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Component 1: LayoutLMv3 — Why It's the Right Model

### What Makes Document Understanding Hard

A number is just a number to a plain NLP model. But its *position* on the page determines its *meaning*:

```
┌─────────────────────────────────────┐
│  INVOICE                            │
│                         INV-2024-03 │  ← top right = invoice number
│  Date: 2024-03-15                   │
│                                     │
│  SHIP TO:            BILL TO:       │
│  123 Main St         456 Oak Ave    │
│                                     │
│  Item        Qty    Unit    Total   │
│  Widget A     5    $10.00   $50.00  │
│  Widget B    10     $8.00   $80.00  │
│                                     │
│                     TOTAL: $130.00  │  ← bottom right = total (not a quantity)
└─────────────────────────────────────┘
```

LayoutLMv3 encodes the (x, y) coordinates of every token as part of its input, so "130.00" at the bottom right is represented differently from "130.00" in the quantity column.

### The Three Input Streams

```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")

# Prepare inputs
encoding = processor(
    image,           # the document image (visual patches)
    words,           # list of words from OCR: ["Invoice", "#", "2024-03", ...]
    boxes,           # bounding boxes: [[x1,y1,x2,y2], ...] normalized to 0-1000
    word_labels=labels,  # token labels for training: [B-INV_NO, I-INV_NO, O, ...]
    return_tensors="pt",
    padding="max_length",
    truncation=True
)

# Three tensors go into the model simultaneously:
# encoding.input_ids      → text tokens
# encoding.bbox           → 2D position for each token
# encoding.pixel_values   → image patches
```

### Few-Shot LoRA Fine-Tuning on SageMaker

The challenge: Only 200 labeled documents per class (12 classes = 2,400 total documents). This is far too few to fine-tune a full model.

```python
from peft import get_peft_model, LoraConfig, TaskType

# LoRA config: only update a tiny fraction of parameters
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=16,              # rank — controls adapter size
    lora_alpha=32,     # scaling factor
    lora_dropout=0.1,
    # Target only attention layers (where understanding lives)
    target_modules=["query", "value"],
)

# Wrap the pre-trained model with LoRA adapters
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label_list)
)
model = get_peft_model(model, lora_config)

# Parameter count comparison:
model.print_trainable_parameters()
# trainable params: 884,736 || all params: 125,884,672 || trainable%: 0.70%
#
# We only update 0.7% of parameters → no overfitting on 200 samples!
```

### AWS SageMaker Training Job

```python
from sagemaker.huggingface import HuggingFace

estimator = HuggingFace(
    entry_point="train.py",
    source_dir="./src",
    instance_type="ml.g4dn.xlarge",   # 1x T4 GPU, $0.73/hr
    instance_count=1,
    transformers_version="4.36",
    pytorch_version="2.1",
    py_version="py310",
    hyperparameters={
        "model_name": "microsoft/layoutlmv3-base",
        "num_train_epochs": 20,        # more epochs OK with LoRA (less overfitting)
        "per_device_train_batch_size": 4,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "num_labels": 12,
    }
)
estimator.fit({"training": "s3://docai-data/labeled-documents/"})
```

**Result: 96% accuracy across 12 document types with only 200 samples per class.**

Why it works: LayoutLMv3 was pre-trained on millions of documents. It already deeply understands document structure. Your LoRA adapters just need to teach it *which* of those structural patterns correspond to your 12 document types.

---

## 📦 Component 2: FastAPI Async Task Queue — 4x Throughput

### Synchronous Baseline (What's Wrong With It)

```
Worker thread 1: processing doc → GPU → waiting → done
Worker thread 2: waiting...
Worker thread 3: waiting...
GPU: 40% utilized (idle while CPU does pre/post processing)
Throughput: ~2,500 docs/hour
```

### Async Architecture (The Fix)

```python
from fastapi import FastAPI, BackgroundTasks
from celery import Celery
import asyncio

app = FastAPI()
celery = Celery("docai", broker="redis://redis:6379")

# GPU batch accumulator
pending_docs = []
BATCH_SIZE = 32
BATCH_TIMEOUT_MS = 50  # wait max 50ms to fill a batch

@app.post("/process")
async def process_document(file: UploadFile):
    """Non-blocking: immediately returns a job ID"""
    doc_id = uuid4()
    doc_bytes = await file.read()
    
    # Add to Celery queue (non-blocking)
    task = celery_process.delay(doc_id, doc_bytes)
    
    return {"job_id": str(doc_id), "task_id": task.id, "status": "queued"}

@celery.task
def celery_process(doc_id, doc_bytes):
    """Runs in worker process, batched GPU inference"""
    # Pre-process on CPU
    words, boxes, image = preprocess_document(doc_bytes)
    
    # Add to GPU batch
    with batch_lock:
        pending_docs.append((doc_id, words, boxes, image))
        
        if len(pending_docs) >= BATCH_SIZE:
            process_batch(pending_docs.copy())
            pending_docs.clear()
    
    return {"doc_id": doc_id, "status": "processed"}

def process_batch(batch):
    """Single GPU inference call for entire batch"""
    all_encodings = processor(
        [b[3] for b in batch],   # all images
        [b[1] for b in batch],   # all word lists
        [b[2] for b in batch],   # all box lists
        return_tensors="pt",
        padding=True
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model(**all_encodings)  # GPU processes all 32 at once
    
    # Distribute results
    for i, (doc_id, _, _, _) in enumerate(batch):
        store_result(doc_id, decode_predictions(outputs.logits[i]))
```

### Throughput Math

```
Synchronous:
  1 doc → preprocess (100ms CPU) + inference (350ms GPU) + postprocess (50ms CPU) = 500ms
  Throughput: 1 / 0.5s = 2 docs/sec = 7,200 docs/hour

Async batched (batch=32):
  32 docs → preprocess (parallel, ~100ms) + inference (400ms GPU for all 32) + postprocess (50ms) = 550ms
  Throughput: 32 / 0.55s = 58 docs/sec = ~10,000+ docs/hour ✓
```

---

## 📦 Component 3: Airflow Pipeline Orchestration

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.s3_key_sensor import S3KeySensor
from datetime import datetime

with DAG("docai_pipeline", schedule_interval="*/5 * * * *") as dag:
    
    # Sensor: wait for new documents in S3
    wait_for_docs = S3KeySensor(
        task_id="wait_for_docs",
        bucket_name="docai-incoming",
        bucket_key="incoming/*.pdf",
        wildcard_match=True,
        timeout=300,
    )
    
    # Step 1: Validate and pre-process
    preprocess = PythonOperator(
        task_id="preprocess_documents",
        python_callable=run_preprocessing_batch,
    )
    
    # Step 2: LayoutLMv3 inference
    inference = PythonOperator(
        task_id="run_layoutlmv3",
        python_callable=run_batched_inference,
    )
    
    # Step 3: Post-process + quality check
    postprocess = PythonOperator(
        task_id="postprocess_and_validate",
        python_callable=validate_extractions,
    )
    
    # Step 4: Store results
    store = PythonOperator(
        task_id="store_results",
        python_callable=store_to_database,
    )
    
    wait_for_docs >> preprocess >> inference >> postprocess >> store
```

---

## 📊 Cross-Project Summary

| | TradeBrains | txt2create.com | DocAI |
|---|---|---|---|
| **Core Problem** | Real-time credit card fraud detection | Scalable AI content generation | Document data extraction at scale |
| **Data Source** | TransUnion credit card pipeline | User text prompts | PDFs, scanned docs, forms |
| **Scale** | 500K+ transactions | 500+ concurrent users | 10K+ documents/hour |
| **Key ML Technique** | Graph community detection + sliding windows | GRPO fine-tuning + vLLM serving | Few-shot LoRA on LayoutLMv3 |
| **Streaming Tech** | Kafka + Spark Structured Streaming | Async job queue (BullMQ) | Celery + Redis |
| **Orchestration** | LangGraph multi-agent | TypeScript/Node.js + MLflow | Airflow DAG |
| **Infra** | Kubernetes + Feast + BigQuery | Kubernetes + GPU node pools | AWS SageMaker + FastAPI |
| **Key Achievement** | 89% recall on fraud rings | 38% better prompt adherence | 96% accuracy, 4x throughput |
| **Hardest Problem** | No labels for new fraud types | GPU memory efficiency at scale | Training with only 200 samples |
| **Solution** | Unsupervised + graph structure | PagedAttention + quantization | LoRA (0.7% params updated) |

---

*Each project addresses a different hard problem in production ML: real-time fraud signal at the data layer (TradeBrains), cost-efficient LLM serving at scale (txt2create), and low-data learning for document understanding (DocAI).*
