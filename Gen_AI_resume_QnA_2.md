# Technical Interview Q&A — Resume Bullet Points Deep Dive

---

## 🧠 Project 1: txt2create.com — Multimodal AI Generation Platform

*Stack: React, Python, QLoRA, vLLM, TypeScript, MLflow*

---

### 🔷 Bullet 1: React Components with Memoization

> *"Optimized and developed React components for user interaction on the frontend side using memoization techniques."*

---

**Q1. What memoization techniques did you use in React, and when do you reach for each one?**

**A:** The three main tools are `React.memo`, `useMemo`, and `useCallback`.

- `React.memo` wraps a functional component and prevents re-renders if props haven't changed (shallow comparison by default). I used this for expensive display components like image/video preview cards that received the same props across many parent re-renders.
- `useMemo` caches the *result* of an expensive computation between renders. I used it to avoid re-deriving filtered/sorted lists on every render.
- `useCallback` caches a *function reference* so that child components receiving it as a prop don't unnecessarily re-render. It's especially useful when passing handlers down to memoized children.

---

**Q2. What problem were you actually solving with memoization on this platform? Why was it needed?**

**A:** On a multimodal generation platform, the UI handles concurrent async jobs — each with its own status, progress indicator, and preview. Without memoization, any state update (e.g., one job completing) would trigger re-renders across all job cards, even unaffected ones. This caused noticeable jank. Memoizing individual job card components meant only the relevant card re-rendered on status change.

---

**Q3. How does `React.memo` decide whether to re-render? What are its limitations?**

**A:** It does a shallow equality check on props — it compares object references, not deep values. So if you pass a new object literal `{ config: {...} }` each render, `React.memo` will still re-render even if the values inside are identical. That's why `useMemo` and `useCallback` often need to accompany it — to stabilize object/function references.

---

**Q4. Can memoization ever hurt performance?**

**A:** Yes. Every memoized value has a memory and bookkeeping cost. If a component re-renders almost every time anyway (because its props change frequently), the overhead of the comparison can exceed the savings. React's own docs advise treating memoization as an optimization of last resort, not a default. You should profile first (React DevTools Profiler) and memoize only what's actually expensive.

---

**Q5. What's the difference between memoization and caching?**

**A:** Conceptually similar, but scoped differently. Memoization is function-level — it caches the output of a pure function for a given set of inputs, within a single process or component lifecycle. Caching is broader — it can span network layers, databases, CDNs, and persists beyond a single function call. In React, `useMemo` is memoization; a service worker or Redis layer would be caching.

---

**Q6. How would you debug a component that isn't memoizing correctly?**

**A:** I'd use the React DevTools Profiler to record interactions and inspect which components re-rendered and why. The "why did this render?" feature shows which prop or state changed. Then I'd trace whether object/function references are being stabilized with `useMemo`/`useCallback` on the parent side.

---

**Q7. Have you used `React.lazy` or `Suspense` alongside memoization?**

**A:** Yes — `React.lazy` is for code-splitting (deferred loading of a component bundle), while memoization is about avoiding re-computation once a component is loaded. On a platform with heavy media generation UIs, I used `React.lazy` to defer loading the video preview panel until needed, and `React.memo` to avoid re-renders once it was mounted.

---

### 🔷 Bullet 2: Fine-tuning with GRPO-based RL and Custom Reward Functions

> *"Fine-tuned open-source generative models using GRPO-based reinforcement learning with custom reward functions for human preference alignment, improving prompt-adherence by 30% over SFT-only baselines."*

---

**Q8. What is GRPO and how does it differ from PPO for LLM fine-tuning?**

**A:** GRPO (Group Relative Policy Optimization) is a variant of PPO designed for language model training. The key difference: PPO requires a separate critic/value network to estimate baselines for variance reduction, which doubles memory and compute. GRPO replaces this with a *group-based* baseline — it samples multiple outputs for the same prompt and uses the mean reward across that group as the baseline. This eliminates the value network entirely, making it more memory-efficient while maintaining low-variance gradient estimates.

---

**Q9. Walk me through the RLHF pipeline. What are the stages?**

**A:**
1. **Supervised Fine-Tuning (SFT):** Train the base model on high-quality demonstration data to produce the initial policy.
2. **Reward Model Training:** Train a separate model (or use a scoring function) to score outputs based on human preferences.
3. **RL Fine-Tuning:** Use the reward model's signal to fine-tune the SFT model via an RL algorithm (PPO, GRPO, REINFORCE, etc.), while applying a KL-divergence penalty to prevent the policy from drifting too far from the SFT baseline.

---

**Q10. What reward functions did you design, and how did you operationalize "prompt-adherence"?**

**A:** Prompt-adherence was measured by checking whether the generated content matched specified constraints in the prompt — things like style, modality, and subject fidelity. The reward function combined: (1) a CLIP-based similarity score between the generated image and the text prompt, (2) a rule-based structural check (did the output include all requested elements), and (3) a human-preference proxy score from a fine-tuned BERT classifier trained on pairwise preference data. The composite score was weighted and normalized before being fed to GRPO.

---

**Q11. What is the KL divergence penalty in RL fine-tuning, and why is it critical?**

**A:** KL divergence measures how much the current policy (RL-updated model) has drifted from the reference policy (SFT model). Without a KL penalty, the model can exploit the reward function in degenerate ways — producing gibberish that scores high on a narrow metric but is useless to users. The penalty keeps the model grounded in coherent language. The trade-off is that a too-large KL weight restricts learning; you tune it as a hyperparameter.

---

**Q12. How did you measure the 30% improvement over SFT-only baselines? What metrics did you use?**

**A:** The 30% figure was measured on a held-out evaluation set using an automated prompt-adherence scoring pipeline — the same CLIP + rule-based composite used during training, but applied to generation samples from both models. We also ran blind A/B human evaluations where raters preferred GRPO outputs significantly more. The 30% refers to the composite score delta, not just one metric.

---

**Q13. What open-source models did you fine-tune?**

**A:** We fine-tuned variants in the LLaMA and Mistral families, and for multimodal tasks, LLaVA-style architectures that combine a vision encoder with a language model backbone. The choice depended on the modality target and hardware budget.

---

**Q14. What is QLoRA and why did you use it instead of full fine-tuning?**

**A:** QLoRA (Quantized Low-Rank Adaptation) loads the base model in 4-bit precision to reduce memory, then injects trainable low-rank adapter matrices (LoRA) at selected layers. Only the adapters are trained — the base weights are frozen. This lets you fine-tune a 7B+ parameter model on a single 24GB GPU, which would be impossible with full fine-tuning. The quality trade-off is minimal for most downstream tasks.

---

**Q15. What is LoRA conceptually, and how does it reduce parameters?**

**A:** LoRA decomposes the weight update matrix ΔW into two low-rank matrices: ΔW = A × B, where A is (d × r) and B is (r × k), and r is the rank (e.g., 8 or 16) — much smaller than d or k. So instead of updating millions of parameters in a weight matrix, you're only training r × (d + k) parameters. At inference, you can merge the adapters back into the base weights with zero added latency.

---

**Q16. How do you prevent reward hacking during RL fine-tuning?**

**A:** Several strategies: (1) Use diverse, multi-component reward functions that are hard to game simultaneously. (2) Apply a KL penalty to keep the policy near the SFT baseline. (3) Periodically audit samples that receive high rewards to catch degenerate patterns. (4) Use reward model ensembling — average over multiple reward models to reduce individual model exploitability.

---

**Q17. What's the difference between RLHF and RLAIF?**

**A:** RLHF (RL from Human Feedback) uses human annotators to generate preference labels for training the reward model. RLAIF (RL from AI Feedback) uses another LLM (often a stronger "teacher" model like GPT-4 or Claude) as the annotator. RLAIF is cheaper and faster to scale but introduces the teacher model's biases. Constitutional AI (Anthropic) is a notable example of RLAIF.

---

### 🔷 Bullet 3: vLLM Inference Stack on Kubernetes

> *"Built self-hosted vLLM inference stack on Kubernetes with GPU-aware scheduling, handling 500+ concurrent generation requests."*

---

**Q18. What is vLLM and what makes it fast compared to naive Hugging Face inference?**

**A:** vLLM is a high-throughput inference engine for LLMs, built around two core innovations: **PagedAttention** and **continuous batching**. PagedAttention manages the KV cache in non-contiguous memory pages (like virtual memory in an OS), dramatically reducing KV cache fragmentation and enabling much larger effective batch sizes. Continuous batching means the engine processes requests dynamically as they arrive, rather than waiting to fill a static batch — so GPU utilization stays high even with variable request rates.

---

**Q19. Explain PagedAttention. Why does KV cache management matter?**

**A:** During autoregressive generation, each token attends to all previous tokens — their keys and values are cached (the KV cache). With naive allocation, you pre-allocate a contiguous memory block for the max sequence length upfront, even if the actual sequence is short. This wastes GPU memory and limits the number of concurrent sequences. PagedAttention allocates KV cache in fixed-size pages, only allocating on demand, and allows non-contiguous layouts. This typically enables 2–4x more concurrent sequences for the same GPU memory.

---

**Q20. What does "GPU-aware scheduling" in Kubernetes mean? How did you implement it?**

**A:** Kubernetes by default doesn't understand GPU resources. GPU-aware scheduling involves: (1) Installing the NVIDIA device plugin for Kubernetes so the kubelet can report available GPUs as a resource (`nvidia.com/gpu`). (2) Adding resource requests/limits to pod specs (`resources: limits: nvidia.com/gpu: 1`). (3) Optionally using the NVIDIA GPU Operator for driver/runtime management. For more advanced scheduling (e.g., gang scheduling for multi-GPU jobs), tools like Volcano or the NVIDIA MIG Manager can be added.

---

**Q21. How did you scale to 500+ concurrent requests? What was the bottleneck?**

**A:** The primary bottleneck was GPU memory for the KV cache. We addressed this by: (1) Using vLLM's PagedAttention to maximize concurrent sequences per GPU. (2) Horizontal pod autoscaling — adding more vLLM replicas under load. (3) Load balancing at the service layer with a custom Node.js orchestration layer distributing requests across replicas. (4) Tuning `max-num-seqs` and `gpu-memory-utilization` in vLLM to squeeze out more concurrency per GPU.

---

**Q22. What's the difference between throughput and latency in an inference server context? Which did you optimize for?**

**A:** Throughput is total tokens generated per second across all users. Latency is the time for one user's request to complete (specifically time-to-first-token, TTFT, and inter-token latency). They trade off against each other: larger batches improve throughput but increase TTFT for individual users. For a generation platform where users tolerate some wait for quality output, we optimized for throughput while keeping TTFT under a soft SLA of ~3 seconds.

---

**Q23. How did you handle model loading time in Kubernetes? Pods are slow to start if they need to load large models.**

**A:** We pre-loaded models by keeping vLLM pods warm (min replicas > 0 in HPA config), so there's always at least one ready instance. For scale-out, we used a shared model store (mounted NFS or S3-backed persistent volume) so new pods pulled weights from a fast local cache rather than downloading from the internet. We also explored model quantization (4-bit/8-bit) to reduce load time and memory footprint.

---

**Q24. What are the trade-offs of self-hosting vLLM vs. using a managed API like OpenAI?**

**A:** Self-hosting gives you data privacy, cost control at scale, the ability to serve custom fine-tuned models, and no per-token API markup. But you take on operational burden: GPU provisioning, infrastructure reliability, model updates, and on-call. Managed APIs are zero-ops and always up-to-date but are expensive at scale, offer no custom model support, and involve sending data to a third party. For a platform serving fine-tuned proprietary models, self-hosting was the only viable path.

---

### 🔷 Bullet 4: TypeScript Orchestration Layer + MLflow Experiment Tracking

> *"Designed TypeScript/Node.js orchestration layer with async job queues and model-load balancing across backends; tracked 20+ fine-tuning runs with MLflow for reproducible experiment management."*

---

**Q25. What did the orchestration layer do, and why TypeScript/Node.js rather than Python?**

**A:** The orchestration layer sat between the React frontend and the vLLM/Python backends. It handled: receiving generation requests, enqueuing them, routing to the appropriate backend based on load, and streaming results back to the client. Node.js was chosen because it's event-driven and non-blocking — ideal for I/O-heavy coordination work where you're mostly waiting on GPU backends, not doing CPU computation. TypeScript added type safety for the job/result schemas shared across services.

---

**Q26. What async job queue implementation did you use? BullMQ? Custom?**

**A:** We used BullMQ (backed by Redis) for durable job queuing. Each generation request became a job with a unique ID. Workers pulled jobs from the queue and dispatched to vLLM backends. The Redis backend gave us persistence (jobs survive restarts), visibility into queue depth for autoscaling signals, and support for priorities and retries. The frontend polled or used SSE to get job status updates.

---

**Q27. How did you implement load balancing across multiple vLLM backends?**

**A:** We maintained a registry of available vLLM backend addresses and their current queue depths (queried via vLLM's `/metrics` endpoint). The orchestration layer implemented a weighted least-connections algorithm — routing new jobs to the backend with the lowest current load. We also implemented circuit breaking: if a backend stopped responding, it was removed from the pool until a health check passed.

---

**Q28. What does MLflow track, and how did you use it for reproducibility?**

**A:** MLflow tracks: (1) **Parameters** — hyperparameters like learning rate, LoRA rank, batch size, KL coefficient. (2) **Metrics** — loss curves, reward scores, eval benchmark results logged per step. (3) **Artifacts** — model checkpoints, tokenizer configs, sample generations. (4) **Tags** — metadata like base model name, dataset version. With 20+ runs, we could filter by metric, compare runs side-by-side, and reproduce any past training configuration exactly by querying its logged params.

---

**Q29. How do you ensure true reproducibility in ML experiments?**

**A:** Beyond MLflow parameter logging, you need: (1) Fixed random seeds for PyTorch, NumPy, and Python. (2) Dataset versioning (we used DVC or logged dataset hash to MLflow). (3) Pinned dependency versions (`requirements.txt` or Docker image digest). (4) Deterministic CUDA ops where possible (though some ops are non-deterministic on GPU). (5) Logging the Git commit hash alongside each run so you can check out the exact code state.

---

**Q30. What is the difference between an MLflow Run, Experiment, and Model Registry?**

**A:** An **Experiment** is a named container grouping related runs (e.g., "GRPO-LLaMA-3-8B"). A **Run** is one training execution with its own parameters, metrics, and artifacts. The **Model Registry** is a governed store for promoting model versions through stages (Staging → Production) with lineage tracking. We used the Registry to track which checkpoint was deployed to the vLLM server at any given time.

---

---

## 🧠 Project 2: alfred.capital — Financial Intelligence System

*Stack: React, LangChain, Python, FastAPI, Docker*

---

### 🔷 Bullet 5: LLM Pipelines with LangChain + RAG

> *"Built enterprise-grade LLM pipelines using LangChain and RAG architectures, processing 10K+ documents with 30% reduction in hallucinations and 50% improvement in retrieval latency."*

---

**Q31. What is RAG (Retrieval-Augmented Generation) and why is it better than pure LLM prompting for financial data?**

**A:** RAG grounds LLM responses in retrieved, up-to-date documents rather than relying solely on the model's parametric knowledge. The pipeline: (1) Embed a user query. (2) Retrieve semantically similar documents from a vector store. (3) Inject those documents into the LLM context as grounding. For financial data this is critical — stock prices, earnings, and news change daily. A model's training data is stale; RAG gives it access to current information and reduces hallucination because the model can cite retrieved facts rather than inventing them.

---

**Q32. How did you measure a 30% reduction in hallucinations?**

**A:** We used a combination of: (1) **Faithfulness scoring** — for each LLM response, we checked whether every factual claim was supported by the retrieved context using an NLI (Natural Language Inference) model. (2) **Human eval** on a labeled test set of 200 financial queries, where annotators flagged unsupported claims. The 30% refers to the drop in faithfulness violation rate between the pure-LLM baseline and the RAG pipeline.

---

**Q33. What caused the 50% improvement in retrieval latency?**

**A:** Several optimizations contributed: (1) Switching from exact search to approximate nearest neighbor (ANN) with FAISS's IVF index, which trades marginal accuracy for much faster search at scale. (2) Caching embeddings for frequently queried documents so they didn't need to be re-embedded on every request. (3) Batching embedding computations. (4) Co-locating the FAISS index in memory on the FastAPI server rather than making network calls to a remote vector DB.

---

**Q34. How do you handle document chunking for RAG? What's your strategy?**

**A:** Chunking strategy significantly impacts retrieval quality. I used a hybrid approach: (1) **Sentence-aware splitting** — break at sentence boundaries, not arbitrary character counts, to keep semantic units intact. (2) **Overlapping chunks** (e.g., 20% overlap) so context at chunk boundaries isn't lost. (3) **Metadata-aware chunking** — financial news articles were chunked with the headline and publication date prepended to each chunk, so retrieved chunks always carried temporal context. Chunk size was tuned to ~512 tokens as a balance between specificity and context.

---

**Q35. What are the failure modes of RAG? How did you address them?**

**A:**
- **Retrieval failure:** The right document isn't retrieved (low recall). Addressed by hybrid search (vector + BM25) and query expansion.
- **Context overflow:** Too many retrieved chunks exceed the context window. Addressed by re-ranking and truncating to top-k.
- **Irrelevant retrieval:** Retrieved docs are semantically similar but not actually relevant. Addressed by cross-encoder re-ranking.
- **Faithfulness failure:** The LLM ignores retrieved context and still halluculates. Addressed by prompt engineering (explicit instruction to only use provided context) and the KL-grounded reward signal from RLHF.

---

**Q36. What is LangChain and why use it over writing pipeline code manually?**

**A:** LangChain is a framework for composing LLM applications. It provides abstractions for chains (sequences of LLM calls), retrievers, memory, and agents. The benefit is speed of iteration — connecting a retriever, a prompt template, and an LLM model is a few lines rather than custom integration code. The downside is abstraction overhead: debugging is harder, and some LangChain abstractions add latency. For complex custom logic, I sometimes dropped down to raw API calls instead of fighting the framework.

---

### 🔷 Bullet 6: Custom Retriever with Vector Similarity + Sentiment Analysis

> *"Implemented a custom retriever that combines vector similarity search with sentiment analysis to prioritize recent and relevant financial news, improving recommendation accuracy by 15%."*

---

**Q37. How does your custom retriever work? Walk me through the end-to-end flow.**

**A:** 
1. User submits a query (e.g., "Is AAPL a good buy?").
2. The query is embedded into a dense vector.
3. FAISS retrieves the top-50 candidate documents by cosine similarity.
4. Each candidate is scored by a sentiment model (FINBERT) — bullish documents about the queried entity get a sentiment boost score.
5. Each candidate is also scored by recency — documents from the last 7 days get a higher weight than older ones.
6. The composite score (similarity + sentiment weight + recency weight) re-ranks the top-50, and the top-5 are injected into the LLM context.

---

**Q38. Why use FINBERT specifically for sentiment instead of a general sentiment model?**

**A:** FINBERT is pre-trained on financial corpora (Reuters news, financial reports), so it understands domain-specific language. "The company missed estimates" is negative in finance but might confuse a general model. General sentiment models trained on social media or movie reviews don't understand financial jargon and tend to misclassify neutral or domain-specific financial language. FINBERT's domain alignment gave us better sentiment signal with less noise.

---

**Q39. How did you tune the weights between similarity, sentiment, and recency?**

**A:** We treated weight tuning as a hyperparameter search. We had a labeled evaluation set of queries with ground-truth "relevant" document lists, and we optimized the weights (α for similarity, β for sentiment, γ for recency) using grid search to maximize Mean Reciprocal Rank (MRR) on the eval set. The best weights were found to emphasize recency heavily for time-sensitive queries (breaking news) and similarity more for fundamental analysis queries.

---

**Q40. What is the difference between dense retrieval and sparse retrieval (BM25)?**

**A:** Dense retrieval embeds queries and documents into continuous vector spaces (using transformer models like BERT/sentence-transformers) and finds nearest neighbors by cosine or dot-product similarity. It captures semantic meaning — "earnings beat" and "better than expected profit" are close in embedding space. BM25 is a sparse term-frequency-based method — it scores documents by exact keyword overlap. Dense retrieval generalizes better to paraphrase but may miss rare exact terms. BM25 is more reliable for precise identifiers (ticker symbols, proper names). Hybrid search combines both.

---

**Q41. How does FAISS work? What index type did you use and why?**

**A:** FAISS (Facebook AI Similarity Search) is a library for efficient approximate nearest neighbor search over dense vectors. For 10K documents, I used the `IVFFlat` index: it partitions the vector space into Voronoi cells (nlist clusters), and at query time searches only a subset of clusters (nprobe). For larger scales, `IVFPQ` adds product quantization to compress vectors and reduce memory. `IVFFlat` offered a good recall/speed trade-off for our dataset size.

---

**Q42. How do you keep the vector store up-to-date with new financial news arriving constantly?**

**A:** We implemented an incremental indexing pipeline: new documents are embedded and inserted into the FAISS index in append mode. Since FAISS doesn't natively support deletion, we maintained a "tombstone" filter — deleted or outdated document IDs were tracked in Redis, and retrieved results were filtered against this set before returning. For a more scalable solution, a vector DB like Pinecone or Weaviate (which support real-time upserts and deletes natively) would be preferable.

---

### 🔷 Bullet 7: React UI with Real-Time Data, FINBERT, and BM25/FAISS

> *"Designed a user-friendly interface in React to display stock recommendations, incorporating real-time data updates combining FINBERT and BM25/FAISS sparse search for enhanced user engagement."*

---

**Q43. How did you implement real-time data updates in the React UI?**

**A:** We used Server-Sent Events (SSE) for streaming updates from the FastAPI backend. SSE is simpler than WebSockets for one-directional server-to-client streaming — the server pushes updates as new news is indexed or a new recommendation is generated. On the React side, an `EventSource` hook subscribed to the SSE endpoint and dispatched updates to local state. For the stock price ticker (truly high-frequency), we used a WebSocket connection to a market data provider.

---

**Q44. What does BM25 add on top of FAISS vector search? Why combine them?**

**A:** FAISS (dense) captures semantic similarity but can struggle with exact term matches — ticker symbols, company names, specific financial metrics. BM25 (sparse) excels at exact keyword matching but misses semantic paraphrase. Combining them via Reciprocal Rank Fusion (RRF) or a learned linear combination produces a retriever that handles both cases. In practice, this "hybrid" retriever significantly improves recall on financial queries that mix semantic intent with exact entity names.

---

**Q45. How do you handle stale UI state in real-time dashboards?**

**A:** Several strategies: (1) **Optimistic updates** — update the UI immediately on user action, then confirm with the backend. (2) **Version tokens / ETags** — each data response carries a version; the client rejects updates older than what it currently holds. (3) **Reconnection logic** — SSE connections drop; the `EventSource` API automatically reconnects, but on reconnect we re-fetch the latest state snapshot rather than replaying all missed events.

---

**Q46. What were your performance considerations for rendering large lists of stock recommendations?**

**A:** We used React virtualization (`react-window` or `react-virtual`) to render only the visible rows of long lists, avoiding mounting thousands of DOM nodes. Recommendation cards were memoized to prevent re-renders on unrelated state changes. We also debounced search/filter inputs to avoid triggering API calls on every keystroke.

---

### 🔷 Bullet 8: LangGraph Multi-Agent System for Retrieval Failure Recovery

> *"Designed a LangGraph multi-agent system where agents detect retrieval failures and reroute autonomously — improving financial entity extraction F1 from 0.71 to 0.85."*

---

**Q47. What is LangGraph, and how is it different from a standard LangChain chain?**

**A:** LangGraph extends LangChain to support **stateful, cyclical** workflows modeled as a directed graph. A standard LangChain chain is a linear DAG — steps execute in order. LangGraph lets you define conditional edges and loops — an agent can retry, reroute, or call a different tool based on intermediate results. This is essential for agentic systems where the next action depends on what the previous action returned.

---

**Q48. What does a "retrieval failure" look like in your system, and how do agents detect it?**

**A:** A retrieval failure occurs when the retrieved documents are insufficient to answer the query — e.g., the vector search returns documents about the wrong company, or no recent documents exist for a niche entity. Detection heuristics: (1) The LLM's response includes uncertainty markers ("I don't have information on..."). (2) The retrieved documents' similarity scores fall below a confidence threshold. (3) A cross-encoder re-ranker assigns low relevance to all top-k results. When any condition fires, the router agent activates a fallback strategy.

---

**Q49. What rerouting strategies did the agents use on retrieval failure?**

**A:** The fallback graph included: (1) **Query rewriting agent** — rephrases the original query (e.g., using synonyms or the full company name instead of a ticker) and retries retrieval. (2) **Web search agent** — falls back to a live web search tool if the internal knowledge base doesn't have sufficient coverage. (3) **Decomposition agent** — breaks a complex query into sub-queries and retrieves independently before synthesizing. The routing decision was made by a lightweight classifier trained on (query, retrieval scores) pairs.

---

**Q50. How did you measure the improvement in F1 from 0.71 to 0.85 for financial entity extraction?**

**A:** We maintained a labeled test set of 500 financial documents with annotated entities (company names, tickers, financial metrics, dates). The extraction pipeline ran over this set, and F1 was computed as the harmonic mean of precision (fraction of extracted entities that were correct) and recall (fraction of gold entities that were extracted). The improvement came from the multi-agent rerouting reducing cases where the extraction model received insufficient context and defaulted to low-confidence or wrong extractions.

---

**Q51. What are the main challenges of building multi-agent systems?**

**A:**
- **Coordination complexity:** Agents can enter infinite loops or deadlock; you need loop detection and max-retry limits.
- **Observability:** Tracing which agent made which decision is hard. We used LangSmith for tracing agent execution paths.
- **Latency accumulation:** Each agent hop adds latency. You must bound the number of retries.
- **State consistency:** Multiple agents modifying shared state can cause race conditions. LangGraph's state management helps, but careful schema design is required.
- **Cost:** Multi-agent systems make more LLM calls. For financial latency-sensitive queries, this is a real concern.

---

**Q52. How do you prevent a multi-agent system from entering an infinite loop?**

**A:** (1) **Max iteration counter** — hard limit on the number of steps a workflow can take before returning an error response. (2) **Visited state tracking** — hash the (query, retrieval strategy) combination; if we've tried the same combination before, skip it. (3) **Confidence gates** — only retry if the confidence delta from the previous attempt was above a threshold; if repeated retries yield no improvement, terminate. (4) LangGraph supports conditional edge guards where you can explicitly check a retry counter in the graph state.

---

---

## 🔧 Cross-Cutting Technical Questions

---

**Q53. How do you approach system design for a platform that integrates both a heavy ML backend and a real-time frontend?**

**A:** The core principle is decoupling. The ML backend is CPU/GPU-intensive and slow; the frontend is latency-sensitive and I/O-bound. I separate them with an async job queue (e.g., BullMQ): frontend submits a job and polls or listens via SSE for completion; the ML backend processes independently. This prevents frontend threads from blocking on slow inference, enables horizontal scaling of ML workers independently of web servers, and provides natural retry/failure handling through the queue.

---

**Q54. What is the CAP theorem, and how did it influence decisions in this system?**

**A:** The CAP theorem states a distributed system can guarantee at most two of: Consistency, Availability, and Partition tolerance. In practice, partition tolerance is non-negotiable for any network system, so the choice is between CP (consistent but potentially unavailable under partition) and AP (available but potentially stale). For financial recommendations, we chose CP for the retrieval index (stale recommendations are worse than a brief unavailability) but AP for the job queue (we'd rather accept a job and potentially retry it than refuse requests under load).

---

**Q55. How do you monitor ML models in production?**

**A:** Monitoring has two layers: (1) **Infrastructure metrics** — GPU utilization, inference latency (P50/P95/P99), throughput, queue depth, error rates. Scraped by Prometheus, visualized in Grafana. (2) **Model quality metrics** — output distribution shift (are the embeddings drifting?), hallucination rate on sampled outputs, downstream task performance on a rolling evaluation set. Significant drift triggers a fine-tuning or re-indexing pipeline automatically.

---

**Q56. What are embeddings, and how do you choose an embedding model?**

**A:** Embeddings are dense vector representations of text where semantically similar texts are close in vector space. Selection criteria: (1) Domain fit — a model fine-tuned on financial text (e.g., FinBERT embeddings) outperforms a general model for financial RAG. (2) Dimensionality — higher dimensions capture more nuance but cost more memory and compute. (3) Context length — make sure the model can embed your chunk size without truncation. (4) Speed — embedding 10K documents needs to be fast enough for your indexing SLA.

---

**Q57. What is FastAPI and why is it preferred over Flask for ML serving?**

**A:** FastAPI is built on Starlette and Pydantic, with native async support and automatic OpenAPI schema generation from Python type hints. For ML serving, the async support is critical — you can handle many concurrent requests without threads blocking on slow model inference (using `asyncio` with background tasks or thread pools for CPU-bound work). Flask is synchronous by default, making it less suited for high-concurrency inference endpoints without additional tooling like Gunicorn workers.

---

**Q58. Walk me through how Docker helped you in these projects.**

**A:** Docker ensured environment reproducibility — the CUDA driver, Python version, and all dependencies were locked in an image, eliminating "works on my machine" issues. For deployment: each service (FastAPI backend, vLLM server, Node.js orchestrator) had its own Dockerfile, built and pushed to a registry, and pulled by Kubernetes. Docker Compose was used for local development to spin up all services together. Docker also enabled incremental layer caching, speeding up CI builds by only rebuilding changed layers.

---

**Q59. What is Docker vs. Kubernetes, and when do you need Kubernetes?**

**A:** Docker is a container runtime — it builds and runs containers on a single machine. Kubernetes is a container *orchestration* platform — it manages containers across a cluster of machines, handling scheduling, scaling, self-healing (restarting failed pods), service discovery, and rolling deployments. You need Kubernetes when you outgrow a single server: multiple replicas for high availability, auto-scaling under load, or coordinating multiple interdependent services at scale.

---

**Q60. If you had to pick one thing you'd do differently across both projects, what would it be?**

**A:** On txt2create, I'd invest earlier in observability for the vLLM cluster — we retrofitted Prometheus metrics and distributed tracing after hitting production bottlenecks, which slowed diagnosis. On alfred.capital, I'd use a purpose-built vector database (Pinecone or Weaviate) from the start rather than FAISS, which required significant custom code for real-time document updates and deletions. The operational overhead of maintaining our own FAISS lifecycle wasn't worth the cost savings at our scale.

---

*End of Interview Q&A — 60 Questions Across 8 Resume Bullets*
