# 🎯 Gen AI Engineer — Interview Preparation
### Target: Deloitte | Experience: ~3.5 YOE | Profile: AI Engineer (Avneez Bhargkeshari)

---

## 📌 Section 1: RAG & Retrieval Systems

---

### Q1. Walk me through the enterprise RAG pipeline you built at [company]. What were the key design decisions?

**Answer:**

Sure. At [company], I architected an enterprise-grade RAG pipeline that was handling over a thousand documents. The core challenge wasn't just retrieval — it was ensuring that what the LLM received as context was actually relevant, because hallucinations were a serious problem in the early version.

Here's how it was structured:

- **Ingestion layer**: Documents were chunked with overlap-aware splitting — I didn't just use fixed-size chunks. I experimented with sentence-level and semantic chunking to maintain coherence within a chunk.
- **Embedding & indexing**: We used both Pinecone for managed vector storage and FAISS for local high-speed retrieval, depending on the deployment context. Embeddings were generated using domain-tuned models.
- **Retrieval**: I implemented a hybrid retrieval approach — dense vector search combined with BM25 sparse search — which significantly improved recall on keyword-heavy queries.
- **Reranking**: Retrieved chunks were passed through a cross-encoder reranker before being sent to the LLM, which filtered out low-relevance chunks.
- **AWS infra**: The whole thing ran on AWS — S3 for document storage, Lambda for async ingestion triggers, and EKS for serving.

The result was a 32% reduction in hallucinations and 45% improvement in retrieval latency. The latency win came mainly from pre-computed embeddings, batched indexing, and caching frequently retrieved chunks.

---

### Q2. In your alfred.capital project, you reduced hallucinations by 68% using a hybrid RAG approach. How exactly did that work?

**Answer:**

This was one of my most technically interesting projects. The financial domain is very unforgiving — a wrong number or incorrect entity in an answer can have real consequences.

The 68% hallucination reduction came from a combination of things:

**First**, the hybrid retrieval itself. Dense retrieval (FAISS/Pinecone) is great for semantic similarity, but in finance, you often query on exact tickers, dates, or entity names. BM25 handles those keyword-exact matches better. By combining both scores — using a weighted reciprocal rank fusion — we made sure neither type of query was left underserved.

**Second**, I used LangGraph to build a multi-agent system where one agent specifically monitors retrieval quality. If the retrieved context has low confidence scores or no documents above a certain relevance threshold, it reroutes the query — either broadening the search scope or switching to a different retrieval strategy entirely.

**Third**, I added a post-generation verification step where a smaller, fast model cross-checked cited facts against the retrieved context. If there was a mismatch, it flagged or regenerated the answer.

The combination of these three layers brought hallucinations down dramatically. The financial entity extraction F1 also went from 0.71 to 0.89, which was a big deal for the downstream analytics pipeline.

---

### Q3. How do you handle retrieval failures in production RAG systems?

**Answer:**

Retrieval failure is something most teams don't think about until it's causing silent wrong answers in production — and that's dangerous.

In my LangGraph multi-agent system at alfred.capital, I designed explicit failure detection into the agentic loop. Here's the pattern I used:

- After retrieval, an **evaluator agent** checks the confidence score and semantic similarity of the top-k retrieved chunks against the query. If no chunk exceeds a threshold, it's flagged as a retrieval failure.
- The system then **reroutes** — it might try a rephrased version of the query, switch from dense to sparse retrieval, or escalate to a broader keyword search.
- If rerouting still doesn't surface relevant context, the system returns a **graceful fallback** — something like "Insufficient information in the knowledge base" — rather than hallucinating an answer.

This autonomous rerouting was key to why the system performed well in production without constant human intervention. I also logged all retrieval failures to a monitoring dashboard so we could identify knowledge gaps and update the corpus proactively.

---

## 📌 Section 2: LLM Fine-Tuning & Training

---

### Q4. You fine-tuned 7B and 13B Qwen3-VL models using QLoRA. Can you explain the technical approach and why QLoRA?

**Answer:**

Absolutely. Fine-tuning large models from scratch is computationally expensive and often unnecessary when you have a strong base model. QLoRA — Quantized Low-Rank Adaptation — gives you the best of both worlds: you can fine-tune a 13B model on a fraction of the GPU memory.

Here's how I approached it:

- **Quantization**: First, the base model is loaded in 4-bit NF4 quantization using bitsandbytes. This reduces the memory footprint by roughly 4x without significant quality loss, because NF4 is information-theoretically optimal for normally distributed weights.
- **LoRA adapters**: Instead of updating all model weights, LoRA injects small trainable rank decomposition matrices into the attention layers — typically into Q and V projection matrices. These adapters have maybe 1-5% of the parameters of the full model.
- **Supervised Fine-Tuning (SFT)**: I ran SFT on domain-specific data with carefully constructed instruction-response pairs. Data quality here is critical — garbage in, garbage out.
- **AWS SageMaker**: I used SageMaker GPU clusters for the training runs, with multi-GPU setups.

The result was a 21% improvement in domain-specific accuracy and a 60% cut in inference cost, because the fine-tuned model needed fewer tokens and less prompting to get the right answer.

Why not full fine-tuning? Because at 13B parameters, full fine-tuning would require 8x or more the GPU memory. QLoRA made this practical on our available infrastructure.

---

### Q5. Tell me about your GRPO-based fine-tuning work on txt2create.com. How does GRPO differ from PPO-based RLHF?

**Answer:**

Great question. At txt2create.com, I used GRPO — Group Relative Policy Optimization — which is a recent advancement that addresses some of the practical pain points of PPO-based RLHF.

In traditional PPO-based RLHF, you need a separate reward model and a value network (critic). The critic estimates the baseline value for variance reduction, but it adds significant memory overhead and training complexity, especially with large models.

GRPO simplifies this by **replacing the critic with group-relative baselines**. Here's the intuition: instead of estimating an absolute value for each response, you generate a group of responses for the same prompt, score them all with your reward function, and use the group's average reward as the baseline. The policy is then updated to prefer responses that score above the group average.

In practice for txt2create.com:

- I defined **custom reward functions** for image generation — things like prompt adherence (does the output match what was asked), aesthetic quality scores from a CLIP-based model, and diversity metrics.
- Each prompt generated a group of candidate outputs, which were ranked by these reward functions.
- GRPO updated the model to shift probability mass toward higher-scoring outputs.

The result was a 38% improvement in prompt-adherence over SFT-only baselines. SFT alone just learns to mimic examples — GRPO actively optimizes for the quality signal you care about.

---

### Q6. How did you scale distributed training to 34B+ parameter models? Walk me through your DeepSpeed and FSDP setup.

**Answer:**

Scaling to 34B+ parameters requires careful parallelism strategy because no single GPU can fit the model in memory.

I used a combination of **DeepSpeed ZeRO Stage 3** and **FSDP** depending on the training scenario:

**DeepSpeed ZeRO-3** shards model parameters, gradients, and optimizer states across all GPUs. So if you have 8 GPUs, each holds only 1/8th of each. The communication overhead is higher, but memory savings are dramatic — it's what made 34B parameter training feasible.

**FSDP (Fully Sharded Data Parallelism)** is PyTorch's native equivalent. It's tighter integrated with the PyTorch ecosystem and easier to combine with custom training loops. I used this when I needed more control over the training logic.

Other key techniques I applied:

- **Gradient checkpointing**: Trading compute for memory by recomputing activations during backward pass instead of storing them.
- **Mixed precision training (BF16)**: BF16 is preferred over FP16 for LLMs because it has better numerical stability with the same memory footprint.
- **Flash Attention**: Replaced standard attention with Flash Attention 2, which is both faster and more memory-efficient.

The end result was a 65% reduction in training time. A lot of that came from eliminating redundant data copies across GPUs and optimizing the communication patterns.

---

## 📌 Section 3: MLOps & Infrastructure

---

### Q7. You reduced model deployment cycle from 3 weeks to 2 days. What changes drove that?

**Answer:**

This was a platform modernization initiative I led for 15+ ML teams at [company]. When I joined that effort, the deployment process was largely manual — model packaging, environment setup, infra provisioning, and monitoring configuration were all done ad hoc by each team.

Here's what I changed:

- **Standardized model packaging**: I built a common container spec using Docker that all teams had to conform to. This eliminated the "works on my machine" problem and made deployments reproducible.
- **Terraform for infra-as-code**: All Kubernetes resources — deployments, services, autoscalers — were defined in Terraform. Teams stopped manually clicking through cloud consoles.
- **Vertex AI Pipelines for orchestration**: Model training, evaluation, and deployment were wired into automated pipelines. A model that passes evaluation gates gets automatically promoted to staging, then production.
- **MLflow for experiment tracking and model registry**: This gave us a single source of truth for model versions, metrics, and artifacts. Before this, teams were sharing model files via S3 buckets with inconsistent naming.
- **CI/CD integration**: Pipeline runs were triggered on code merges, so model deployment became part of the engineering workflow rather than a separate ops process.

The 3-week-to-2-day compression mostly came from eliminating manual handoffs and approval steps that had no real quality gate — just bureaucratic waiting.

---

### Q8. How did your model monitoring system cut production incidents by 73%?

**Answer:**

Production incidents in ML systems often happen because of silent degradation — the model doesn't crash, it just gives worse answers, and nobody notices until a business metric drops.

The monitoring system I designed caught these early. Here's how:

- **Feature drift detection**: I used statistical tests — specifically KS-test and Population Stability Index — to compare the distribution of incoming feature values against the training distribution. If drift is detected, an alert fires.
- **Prediction drift monitoring**: Beyond features, I also tracked the distribution of model outputs. A sudden shift in predicted label distribution often signals a real-world change that the model isn't handling.
- **Data quality checks**: Null rates, schema validation, and value range checks ran on every inference batch. Upstream data issues were caught before they corrupted model behavior.
- **Automated retraining triggers**: When drift exceeded thresholds, a Vertex AI Pipeline was triggered automatically to retrain on fresh data. This closed the loop without requiring human intervention.
- **MTTR improvement**: Previously, incidents took 4 hours to resolve because nobody knew what had changed or when. With dashboards showing exactly when drift started and what features were affected, the investigation time collapsed to 23 minutes on average.

The 73% reduction in incidents was largely because we stopped treating monitoring as an afterthought and made it a first-class part of the deployment pipeline.

---

### Q9. You built MCP servers at [company] to expose internal tools to LLM agents. Can you explain the architecture?

**Answer:**

MCP — Model Context Protocol — is a standardized way to expose tools, APIs, and data sources to LLM agents. Think of it as a consistent interface layer between your agents and your organizational tooling.

At [company], we had 5+ agent workflows that each needed to call different internal services — databases, internal APIs, document stores. Before MCP, each agent had custom-written tool integrations that were brittle and hard to maintain.

Here's what I built:

- **MCP Server layer**: I deployed custom MCP servers that wrapped our internal tools — things like internal search APIs, project management systems, and data retrieval services. Each server exposes a set of standardized "tools" that any MCP-compatible agent can call.
- **Plug-and-play integration**: Because MCP is a standard protocol, adding a new tool to an agent became a configuration change rather than a code change. You declare which MCP servers the agent has access to, and it can discover available tools dynamically.
- **Security and access control**: The MCP servers acted as a controlled gateway — agents couldn't directly call internal services, they had to go through the MCP layer, which enforced authentication and rate limiting.

The result was a 70% reduction in integration overhead. New agent workflows that used to take days to wire up were up in hours because the tool layer was already built.

---

## 📌 Section 4: Agentic Systems & Orchestration

---

### Q10. You used LangGraph for multi-agent orchestration at alfred.capital. Why LangGraph over alternatives like CrewAI or AutoGen?

**Answer:**

Each framework makes different trade-offs, and the right choice depends on the use case.

For alfred.capital, the core requirement was **fine-grained control over the agent graph** — I needed to define exactly when agents hand off to each other, how failures are handled, and how state is persisted across steps. That's where LangGraph excels.

LangGraph models your agent workflow as a directed graph where nodes are agent actions and edges are conditional transitions. This gives you:

- **Explicit control flow**: I can define "if retrieval confidence < 0.7, go to the rerouting node" — this kind of conditional routing is first-class in LangGraph.
- **State management**: LangGraph has built-in state that flows through the graph. Every node reads and writes to a shared state object, which makes debugging much easier.
- **Human-in-the-loop support**: For financial use cases, sometimes you need a human approval step before an action. LangGraph's interrupt mechanism handles this natively.

CrewAI is great for role-based multi-agent setups where agents collaborate more loosely — good for autonomous research or content generation tasks. AutoGen is Microsoft's framework and is excellent for conversational multi-agent patterns.

For a production financial intelligence system where I needed deterministic routing and explicit failure handling, LangGraph was the right call.

---

### Q11. How do you think about the trade-off between agentic autonomy and reliability in production systems?

**Answer:**

This is something I think about a lot, especially after deploying agents in financial and enterprise contexts where mistakes are costly.

My mental model is: **autonomy and reliability exist on a spectrum, and you calibrate based on the cost of failure.**

For low-stakes, reversible actions — like searching for information, summarizing documents, or generating a draft — I'm comfortable with high autonomy. If the agent gets it wrong, a human can catch it and correct it.

For high-stakes or irreversible actions — like writing to a database, sending a communication, or making a financial recommendation — I always build in a **human-in-the-loop checkpoint** or at least a **confirmation gate** before execution.

In practice, I implement this through:

- **Tool categorization**: Read-only tools get called freely. Write tools require explicit confirmation.
- **Confidence thresholds**: If an agent's confidence in its plan drops below a threshold, it surfaces the plan for human review rather than executing.
- **Audit logging**: Every agent action is logged with its reasoning trace. This makes post-hoc review possible and builds trust over time.

The goal is to earn autonomy incrementally. Start constrained, prove the system works, then expand the autonomy envelope based on observed performance.

---

## 📌 Section 5: LLM Serving & Production Infrastructure

---

### Q12. You deployed 13B parameter models at sub-200ms latency on Vertex AI with vLLM. How did you achieve that?

**Answer:**

Sub-200ms for a 13B model is aggressive, and hitting that consistently required optimization at multiple layers.

**vLLM's core advantage** is PagedAttention — it manages the KV cache like virtual memory, which dramatically reduces memory waste and allows much higher throughput. Without PagedAttention, the KV cache for concurrent requests either overflows or has to be over-provisioned.

Beyond vLLM itself, I also applied:

- **TensorRT-LLM**: For the most latency-critical paths, I compiled the model with TensorRT-LLM, which generates CUDA kernels optimized for the specific hardware. This gave an additional 20-30% latency reduction over vLLM alone.
- **Model quantization**: Running in INT8 or FP8 reduced memory bandwidth requirements, which is often the bottleneck for LLM inference.
- **Request batching**: vLLM's continuous batching lets you group requests dynamically rather than waiting for a fixed batch to fill. This keeps GPU utilization high.
- **Vertex AI autoscaling**: I configured horizontal pod autoscaling on GKE with GPU node pools, so the system could scale out to handle traffic spikes without latency degradation.

The 40% cost reduction came from better GPU utilization — we were doing the same request volume with fewer GPUs because each GPU was handling more concurrent requests efficiently.

---

### Q13. How did you handle 500+ concurrent generation requests at p99 < 4s on txt2create.com without managed cloud ML services?

**Answer:**

This was a deliberate choice to self-host because managed ML services like SageMaker endpoints or Vertex AI endpoints add latency and cost at high volume. We were running a consumer product where unit economics mattered.

The architecture I built:

- **vLLM inference cluster on Kubernetes**: Multiple vLLM instances running behind a load balancer, with GPU-aware pod scheduling to ensure each pod was on a GPU node.
- **Async job queue**: I didn't serve generation requests synchronously. Incoming requests were enqueued into an async job queue (backed by Redis). This decoupled request acceptance from generation, so the API layer could respond immediately with a job ID, and users polled for results.
- **Model-load balancing**: Different model variants were loaded on different pods. A router layer directed requests to the appropriate backend based on the requested model and current queue depth.
- **GPU-aware scheduling**: Kubernetes node affinity rules ensured that inference pods always landed on GPU nodes, and resource limits prevented CPU-only nodes from getting inference workloads.

The p99 < 4s was achievable because the async queue absorbed traffic spikes gracefully — instead of requests timing out during peak load, they just waited a bit longer in the queue, which is acceptable for generation workloads.

---

## 📌 Section 6: Data Engineering & Feature Stores

---

### Q14. You implemented a Feature Store with Feast and BigQuery. What problem does training-serving skew cause, and how did this solve it?

**Answer:**

Training-serving skew is one of the most insidious problems in production ML, and it's surprisingly common.

Here's the core issue: during training, you compute features from your historical dataset. But in production, those same features are computed on live data — often by a different code path, sometimes written by a different team. Even small differences in how a feature is computed — different normalization, different handling of nulls, slightly different aggregation window — can cause the model to receive inputs at inference time that look different from what it was trained on. The model silently underperforms, and it's very hard to debug.

Feast solves this by providing a **single feature computation layer** that is used both for training data generation and for online inference. Features are defined once as code, and Feast handles serving them from appropriate stores — offline store (BigQuery for batch training) and online store (Redis or Bigtable for low-latency inference).

What I implemented specifically:

- Defined feature views in Feast that pulled from BigQuery — event-based features for user activity, aggregated market features, etc.
- Used Feast's materialization job to push features to the online store on a scheduled basis.
- Integrated Feast's Python SDK into both the training pipeline and the inference service, so both used the identical feature retrieval code path.

The 18% lift in production model accuracy after deploying this was almost entirely attributed to eliminating skew that had been silently degrading performance for months.

---

### Q15. Walk me through the Kafka + Spark Streaming pipeline you built on Databricks for alfred.capital.

**Answer:**

The core problem at alfred.capital was that financial data moves fast — market prices, news events, filings — and the RAG system's knowledge base needed to stay current. A batch-only approach would mean the system was always working with stale information.

Here's the architecture I built:

- **Kafka as the ingestion backbone**: All incoming data streams — market feeds, news APIs, regulatory filings — published to dedicated Kafka topics. Kafka gave us durable, replayable event streams, which is critical for financial data.
- **Spark Structured Streaming on Databricks**: Spark consumers read from Kafka topics and performed real-time transformations — entity extraction, normalization, deduplication, and embedding generation for new documents.
- **Delta Lake as the storage layer**: Processed data landed in Delta Lake tables on Databricks. Delta gives you ACID transactions on top of Parquet files, which means you can do concurrent reads and writes safely — important when Spark is writing while the RAG system is reading.
- **Airflow DAGs for orchestration**: Airflow managed the scheduling of batch jobs that complemented the streaming pipeline — periodic reindexing, model retraining triggers when market drift was detected, and data quality checks.
- **BigQuery as analytics sink**: Aggregated metrics and analytics data flowed to BigQuery for dashboards and reporting.

The retraining trigger was particularly interesting — when Spark detected significant distributional shift in incoming market data (using rolling statistical tests), it would automatically trigger an Airflow DAG that kicked off an LLM fine-tuning run on Vertex AI.

---

## 📌 Section 7: Prompt Engineering & Applied AI

---

### Q16. How do you approach prompt engineering for production systems? What's your process?

**Answer:**

Prompt engineering in production is very different from the quick iteration you do in a playground. At scale, a poorly structured prompt can cost you significantly in tokens, latency, and quality.

My process:

**1. Start with the task specification**: Before writing any prompt, I write down exactly what the model needs to do, what a good output looks like, and what failure modes I'm worried about. This prevents writing prompts that optimize for the wrong thing.

**2. System prompt design**: I separate instructions into the system prompt rather than cramming everything into the user turn. The system prompt establishes role, format constraints, and guardrails.

**3. Few-shot examples**: For tasks with specific output formats or domain nuances, I include 2-4 worked examples in the prompt. Few-shot prompting consistently outperforms zero-shot for structured tasks.

**4. Chain-of-thought for complex reasoning**: For tasks requiring multi-step reasoning — like financial analysis or code debugging — I prompt the model to think step by step before giving the final answer. This dramatically reduces errors.

**5. Evaluation-driven iteration**: I maintain a golden dataset of 50-100 examples with expected outputs. Every prompt change is evaluated against this dataset. I don't ship a prompt change that doesn't improve or maintain performance on the eval set.

**6. Token efficiency**: In high-volume production systems, I audit prompts for unnecessary verbosity. Every token costs money and adds latency.

---

### Q17. How do you evaluate a RAG system's quality in production?

**Answer:**

Evaluation is the hardest part of RAG because you're evaluating two things simultaneously — retrieval quality and generation quality — and they interact.

Here's my evaluation framework:

**Retrieval metrics:**
- **Recall@k**: Of all relevant documents, what fraction did the retriever surface in the top-k results?
- **MRR (Mean Reciprocal Rank)**: How highly ranked is the first relevant document?
- **Context relevance**: A separate LLM judge evaluates whether the retrieved chunks are actually relevant to the query.

**Generation metrics:**
- **Faithfulness**: Is the generated answer grounded in the retrieved context, or is the model hallucinating? I use an LLM judge that cross-checks claims against the context.
- **Answer relevance**: Does the answer actually address what was asked?
- **Completeness**: Are all aspects of a multi-part question addressed?

**End-to-end:**
- **Human evaluation**: For high-stakes domains like finance, I sample a percentage of production queries daily for human review. This catches failure modes that automated metrics miss.
- **A/B testing**: When comparing retrieval or generation strategies, I run controlled A/B tests and measure downstream business metrics.

I use a framework like RAGAS to automate a lot of these checks and integrate them into the CI pipeline so any component change gets evaluated before deployment.

---

## 📌 Section 8: Scenario & Design Questions

---

### Q18. Deloitte serves enterprise clients with strict data security requirements. How would you design a RAG system for a client that cannot send data to external LLM APIs?

**Answer:**

This is a very common requirement in consulting — especially for financial services, healthcare, and government clients. The answer is a fully on-premise or private cloud deployment.

Here's how I'd design it:

- **Model serving**: Deploy open-source models (Llama 3, Mistral, Qwen) on the client's own infrastructure using vLLM. All inference happens within the client's network — no data leaves.
- **Embedding models**: Use a locally deployed embedding model — something like `bge-large-en` or a fine-tuned variant — rather than OpenAI's embedding API. Embeddings are generated on-prem.
- **Vector store**: Deploy a self-hosted vector database — Qdrant or Milvus — within the client's network. No vectors are sent to external services.
- **Orchestration**: LangChain or LlamaIndex can be configured to use local models and local vector stores. The framework doesn't care whether the endpoint is external or internal.
- **Data access controls**: Layer role-based access control at the retrieval level — different users can only retrieve from document segments they're authorized to access. This is typically implemented as metadata filtering in the vector store query.
- **Audit logging**: Every query and retrieved context is logged to an internal audit store for compliance.

I've worked with Vertex AI private deployments on GCP that satisfy many enterprise security requirements — Google's private service connect keeps traffic within the customer's VPC. But for the most sensitive clients, fully air-gapped on-prem is the only option, and open-source models make that viable today.

---

### Q19. A client's LLM-based application is performing well in testing but degrading in production over time. How do you diagnose and fix this?

**Answer:**

This is a classic production ML problem, and it usually comes down to a few root causes. I'd approach it systematically:

**Step 1 — Characterize the degradation.** Is it happening uniformly across all query types, or only for certain topics? Is it gradual or sudden? Sudden degradation often points to an upstream data change. Gradual degradation often points to concept drift.

**Step 2 — Check the data pipeline.** In RAG systems, the knowledge base needs to stay current. If documents aren't being updated or re-indexed, the system is answering questions based on stale information. I'd check the ingestion logs and index freshness.

**Step 3 — Monitor feature/input drift.** I'd look at the distribution of incoming queries compared to the training/testing distribution. If users are now asking questions that are semantically different from what the system was designed for, the retriever may not be surfacing relevant context.

**Step 4 — Check the LLM itself.** If the underlying model was updated by the API provider, behavior can change subtly. I'd pin the model version and evaluate before upgrading.

**Step 5 — Retrieval quality audit.** Sample production queries and manually inspect what the retriever is returning. Sometimes the retriever is returning plausible-looking but irrelevant chunks, and the LLM is doing its best with bad input.

**Fix strategy:**
- Update the knowledge base and re-index.
- Retrain or fine-tune the embedding model on recent query-document pairs.
- Add or update few-shot examples in the prompt to handle new query patterns.
- Implement the drift detection and automated retraining pipeline I built at [company] — that prevents this from being a reactive fire-fighting exercise.

---

### Q20. How would you approach building a Gen AI solution for a Deloitte client in the professional services domain — say, automating parts of audit workflow?

**Answer:**

This is a great applied question. Audit is a domain I find genuinely interesting from an AI perspective because it combines structured data, unstructured documents, regulatory rules, and professional judgment.

Here's how I'd approach it:

**Discovery phase**: Work with auditors to understand which tasks are high-volume and rule-bound vs. which require genuine professional judgment. The former are good automation targets; the latter need AI-assist, not AI-replace.

**Document processing pipeline**: Auditors work with financial statements, contracts, invoices, and correspondence. I'd build a document intelligence layer — using vision-language models for document understanding, entity extraction for amounts/dates/parties, and a classification layer to route documents to the right processing workflow.

**RAG-based policy Q&A**: Audit firms have enormous internal knowledge bases — audit standards, client-specific procedures, regulatory guidance. A RAG system that lets auditors query this in natural language, with citations, would be high value.

**Anomaly detection agent**: An agent that runs automated checks against financial data — comparing ratios to industry benchmarks, flagging unusual transactions, checking journal entry patterns — surfaces exceptions for human review rather than replacing the auditor's judgment.

**Human-in-the-loop design**: Every AI output in an audit context needs to be reviewable and explainable. I'd design the system so every AI recommendation is accompanied by its reasoning and source documents. Auditors sign off on findings; AI accelerates the identification.

**Compliance and data security**: Given Deloitte's client data obligations, everything would run in a private cloud deployment with strict access controls and audit logging — exactly the architecture I described earlier.

The goal isn't to automate audit — it's to let auditors focus on judgment-intensive work by offloading the mechanical, time-consuming parts.

---

## 📌 Section 9: Behavioral & Situational

---

### Q21. Tell me about a time a model you deployed failed in production and how you handled it.

**Answer:**

Yes, this happened early in my time at [company] with one of the first RAG deployments. We had a retrieval pipeline that was performing well in testing, but about two weeks after go-live, users started reporting that the system was giving confidently wrong answers on a specific category of questions.

When I investigated, I found a few issues working together:

The embedding model we'd chosen had weak performance on technical domain-specific terms — it was treating semantically different technical concepts as similar because it was trained on general web text. So the retriever was surfacing plausible-looking but incorrect context.

Additionally, we hadn't implemented a retrieval confidence threshold, so the LLM was receiving low-quality context and generating answers anyway rather than declining.

How I handled it:

- Short-term: I added a confidence threshold to the retriever so that queries with low-confidence retrievals returned a "cannot answer confidently" response instead of a hallucination.
- Medium-term: I fine-tuned the embedding model on domain-specific query-document pairs we collected from the initial deployment.
- Long-term: I built the monitoring and drift detection system that I now apply to all productions deployments — so this type of silent degradation gets caught before users notice it.

The experience reinforced something I now treat as a core principle: retrieval quality gates are not optional in production RAG systems.

---

### Q22. How do you stay current in a field that moves as fast as Gen AI?

**Answer:**

Honestly, it requires deliberate effort because the volume of new work is enormous.

My approach is a mix of depth and breadth:

**For breadth**, I follow Hugging Face's daily paper digest and Twitter/X — it surfaces what's being talked about across the community. I spend maybe 20-30 minutes in the morning scanning titles and abstracts.

**For depth**, I pick 3-4 papers per month that are directly relevant to what I'm building and actually read them carefully — not just the abstract. I work through the math where it matters for implementation.

**For applied learning**, I believe you only really understand something when you've built it. I maintain personal projects — txt2create.com is an example — where I implement new techniques. GRPO was something I learned by reading the DeepSeek-Math paper and then actually implementing it.

**For community**: I'm active in technical communities — I maintain connections on Telegram and with people working on similar problems. Peer learning is underrated; often someone has already hit the problem you're about to face.

I also find that working at the intersection of research and production — which is what my role at [company] involves — naturally keeps me current, because production problems often point you to exactly the right papers to read.

---

### Q23. You've worked across backend engineering, MLOps, and applied AI. How do you see your role as a Gen AI Engineer at Deloitte?

**Answer:**

I see it as a full-stack ownership role for AI systems in production — not just writing prompts or fine-tuning models, but being responsible for the entire chain from data ingestion through model deployment to monitoring and continuous improvement.

What I bring that I think is genuinely differentiating is that I can operate across all layers. I can architect the data pipeline that feeds the RAG system, build the serving infrastructure that runs the model efficiently, write the agentic orchestration logic, and also do the MLOps work to keep it reliable in production.

At Deloitte specifically, I think the value is in translating cutting-edge AI capabilities into solutions that work reliably in enterprise environments — where data security, explainability, and compliance matter as much as model quality. That requires someone who understands both the AI deeply and the engineering rigor that enterprise deployment demands.

I'm also genuinely excited about the consulting model — getting exposure to multiple industries and problem types accelerates learning in a way that staying in one domain doesn't. I want to build a broad repertoire of where Gen AI creates real leverage, and Deloitte is one of the best places to do that.

---

*Prepared for Deloitte Gen AI Engineer Interview | Avneez Bhargkeshari | May 2026*
